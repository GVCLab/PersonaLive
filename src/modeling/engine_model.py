import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import traceback
import os
from PIL import Image

TRT_LOGGER = trt.Logger()
SKIP_ENGINE_MODEL_CHECK = True

def get_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        print(f"Loading engine from file {engine_file_path}...")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print(f"No file named {engine_file_path}! Please check the input.")
        return None 
    

def numpy_to_torch_dtype(np_dtype):
    mapping = {
        np.float32: torch.float,
        np.float64: torch.double,
        np.float16: torch.half,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.int16: torch.int16,
        np.int8: torch.int8,
        np.uint8: torch.uint8,
        np.bool_: torch.bool
    }
    return mapping.get(np_dtype, None)

def match_shape(a, b):
    if(len(a) == len(b)):
        return tuple(a) == tuple(b)
    elif len(a) > len(b):
        if(a[0] == 1):
            return match_shape(a[1:], b)
    else:
        if(b[0] == 1):
            return match_shape(a, b[1:])
    return False

def match_dtype(a, b):
    if(a.__class__ == torch.dtype):
        a = torch.tensor(0,dtype=a).numpy().dtype
    return a == b


class EngineModel:
    def __init__(self, engine_file_path, stream = None, device_int = 0, extra_lock = None):
        self.device_int = device_int
        self.extra_lock = extra_lock
        if not(self.extra_lock is None):
            self.extra_lock.acquire()
        assert os.path.exists(engine_file_path), "Engine model path not exists!"
        self.ctx = cuda.Device(self.device_int).make_context()
        try:
            self.engine = get_engine(engine_file_path) # 载入TensorRT引擎
            input_nvars = 0
            output_nvars = 0
            self.input_names = []
            self.output_names = []

            # 【辅助函数】用于获取安全的 Shape (消除 -1)
            def get_safe_shape(engine, name):
                shape = engine.get_tensor_shape(name)
                # 如果形状里包含 -1 (动态维度)
                if -1 in shape:
                    # 获取 Profile 0 的 (min, opt, max)
                    # 取下标 [2] 即 Max Shape，确保分配足够大的显存
                    profile = engine.get_tensor_profile_shape(name, 0)
                    if profile:
                        print(f"[EngineModel] Detected dynamic shape for {name}: {shape} -> Using Max Profile: {profile[2]}")
                        return profile[2]
                    else:
                        # 如果获取不到 Profile (通常发生在 Output)，这是一个风险点
                        # 这里为了防止报错，可以尝试打印警告
                        print(f"[EngineModel] Warning: Dynamic output {name} has no profile. Mem alloc might fail.")
                return shape

            for binding in self.engine: # 遍历所有tensor，区分Input/Output
                mode = self.engine.get_tensor_mode(binding)
                if(mode== trt.TensorIOMode.INPUT):
                    input_nvars += 1
                    self.input_names.append(binding)
                elif(mode == trt.TensorIOMode.OUTPUT):
                    output_nvars += 1
                    self.output_names.append(binding)

            self.input_nvars = input_nvars # input的数量
            self.output_nvars = output_nvars # output的数量
            
            self.input_shapes = {name : get_safe_shape(self.engine, name) for name in self.input_names} # 获取每个 I/O 的 shape 和 dtype
            self.input_dtypes = {name : self.engine.get_tensor_dtype(name) for name in self.input_names}
            self.input_nbytes = {
                name : trt.volume(self.input_shapes[name]) * trt.nptype(self.input_dtypes[name])().itemsize 
                for name in self.input_names
            } # nbytes = tensor 占多少 CUDA 内存（字节数）
            self.output_shapes = {name : get_safe_shape(self.engine, name) for name in self.output_names}
            self.output_dtypes = {name : self.engine.get_tensor_dtype(name) for name in self.output_names}
            self.output_nbytes = {
                name : trt.volume(self.output_shapes[name]) * trt.nptype(self.output_dtypes[name])().itemsize 
                for name in self.output_names
            }
            self.dinputs = {name : cuda.mem_alloc(self.input_nbytes[name]) for name in self.input_names} # 为每个输入/输出分配 CUDA 设备内存
            self.doutputs = {name :cuda.mem_alloc(self.output_nbytes[name]) for name in self.output_names}
            self.context = self.engine.create_execution_context() # 创建 ExecutionContext（执行上下文）
            if stream is None:
                self.stream = cuda.Stream()
            else:
                self.stream = stream
            for name in self.input_names: # 绑定 tensor 到 context
                self.context.set_tensor_address(name, int(self.dinputs[name]))
            for name in self.output_names:
                self.context.set_tensor_address(name, int(self.doutputs[name]))
            self.houtputs = {
                name :
                cuda.pagelocked_empty(
                    trt.volume(self.output_shapes[name]), dtype=trt.nptype(self.output_dtypes[name])
                ) for name in self.output_names
            } # 分配 page-locked host 内存以存储输出
        except Exception:
            self.ctx.pop()
            raise Exception("CUDA Initialization Failed!")
        self.ctx.pop()
        if not(self.extra_lock is None):
            self.extra_lock.release()

    def __call__(self, skip_check=SKIP_ENGINE_MODEL_CHECK, output_list=[], return_tensor=False, **inputs):
        if not skip_check:
            for name in inputs:
                assert name in self.input_names
                assert match_shape(inputs[name].shape, self.input_shapes[name])
                assert match_dtype(inputs[name].dtype, trt.nptype(self.input_dtypes[name]))
        if not(self.extra_lock is None):
            self.extra_lock.acquire()
        self.ctx.push()
        r = {}
        try:
            
            for name in inputs:
                hinput = inputs[name]
                if (isinstance(hinput,torch.Tensor) and hinput.device.type=="cuda" and hinput.device.index==self.device_int):
                    hinput_con = hinput.contiguous()
                    ptr = hinput_con.data_ptr()
                    cuda.memcpy_dtod_async(self.dinputs[name], ptr, self.input_nbytes[name], self.stream)
                else:
                    hinput_con = np.ascontiguousarray(hinput)
                    cuda.memcpy_htod_async(self.dinputs[name], hinput_con, self.stream)
            for name in self.input_names:
                if name not in inputs:
                    self.context.set_input_shape(name, self.input_shapes[name])
            self.context.execute_async_v3(self.stream.handle)
            if(return_tensor):
                for name in output_list:
                    t = torch.zeros(trt.volume(self.output_shapes[name]), device=f"cuda:{self.device_int}", dtype=numpy_to_torch_dtype(trt.nptype(self.output_dtypes[name])))
                    ptr = t.data_ptr()
                    cuda.memcpy_dtod_async(ptr, self.doutputs[name], self.output_nbytes[name], self.stream)
                    t = t.reshape(tuple(self.output_shapes[name]))
                    r[name] = t
            else:
                for name in output_list:
                    cuda.memcpy_dtoh_async(self.houtputs[name], self.doutputs[name], self.stream)
                    r[name] = self.houtputs[name]
            self.stream.synchronize()
        except Exception as e:
            print("TensorRT Execution Failed!")
            traceback.print_exc()
            self.ctx.pop()
            if not(self.extra_lock is None):
                self.extra_lock.release()
            return None
        self.ctx.pop()
        if not(self.extra_lock is None):
            self.extra_lock.release()
        return r
    

    def prefill(self, skip_check=SKIP_ENGINE_MODEL_CHECK, **inputs):
        if not (skip_check):
            for name in inputs:
                in_input = (name in self.input_names)
                assert in_input or (name in self.output_names)
                assert match_shape(inputs[name].shape, self.input_shapes[name] if in_input else self.output_shapes[name])
                assert match_dtype(inputs[name].dtype, trt.nptype(self.input_dtypes[name] if in_input else self.output_dtypes[name]))    
        if not(self.extra_lock is None):
            self.extra_lock.acquire()
        self.ctx.push()
        try:
            for name in inputs:
                in_input = (name in self.input_names)
                hinput = inputs[name]

                dst_ptr = self.dinputs[name] if in_input else self.doutputs[name]
                real_nbytes = 0
                if isinstance(hinput, torch.Tensor):
                    real_nbytes = hinput.numel() * hinput.element_size()
                else:
                    # 假设是 numpy
                    real_nbytes = hinput.nbytes

                if (isinstance(hinput,torch.Tensor) and hinput.device.type=="cuda" and hinput.device.index==self.device_int):
                    hinput_con = hinput.contiguous()
                    ptr = hinput_con.data_ptr()
                    cuda.memcpy_dtod_async(dst_ptr, ptr, real_nbytes, self.stream)
                else:
                    hinput_con = np.ascontiguousarray(hinput)
                    cuda.memcpy_htod_async(dst_ptr, hinput, self.stream)
            self.stream.synchronize()
        except Exception as e:
            traceback.print_exc()
            self.ctx.pop()
            if not(self.extra_lock is None):
                self.extra_lock.release()
            return False
        self.ctx.pop()
        if not(self.extra_lock is None):
            self.extra_lock.release()
        return True
    
    def __repr__(self):
        r = "TensorRTEngineModel(\n\tInput=[\n"
        for name in self.input_names:
            r += f"\t\t{name}: \t{trt.nptype(self.input_dtypes[name]).__name__}{self.input_shapes[name]},\n"
        r += "\t],Output=[\n"
        for name in self.output_names:
            r += f"\t\t{name}: \t{trt.nptype(self.output_dtypes[name]).__name__}{self.output_shapes[name]},\n"
        r+="\t]\n)"
        return r
    
    def link(self, other, var_map, skip_check=SKIP_ENGINE_MODEL_CHECK):
        assert self.device_int == other.device_int
        if not (skip_check):
            for source in var_map:
                assert source in other.output_names
                target = var_map[source]
                assert target in self.input_names
                assert match_shape(other.output_shapes[source], self.input_shapes[target]) 
                assert match_dtype(other.output_dtypes[source], self.input_dtypes[target])

        if not(self.extra_lock is None):
            self.extra_lock.acquire()
        self.ctx.push()
        try:
            for source in var_map:
                target = var_map[source]
                self.context.set_tensor_address(target, int(other.doutputs[source]))
        except Exception as e:
            traceback.print_exc()
            self.ctx.pop()
            if not(self.extra_lock is None):
                self.extra_lock.release()
            return False
        self.ctx.pop()
        if not(self.extra_lock is None):
            self.extra_lock.release()
        return True
    
    def bind(self, var_map, skip_check=SKIP_ENGINE_MODEL_CHECK):
        if not (skip_check):
            for source in var_map:
                assert source in self.output_names
                target = var_map[source]
                assert target in self.input_names
                assert match_shape(self.output_shapes[source], self.input_shapes[target]) 
                assert match_dtype(self.output_dtypes[source], self.input_dtypes[target])

        if not(self.extra_lock is None):
            self.extra_lock.acquire()
        self.ctx.push()
        try:
            for source in var_map:
                target = var_map[source]
                self.context.set_tensor_address(target, int(self.doutputs[source]))
        except Exception as e:
            traceback.print_exc()
            self.ctx.pop()
            if not(self.extra_lock is None):
                self.extra_lock.release()
            return False
        self.ctx.pop()
        if not(self.extra_lock is None):
            self.extra_lock.release()
        return True

    def unlink(self):

        if not(self.extra_lock is None):
            self.extra_lock.acquire()
        self.ctx.push()
        try:
            for name in self.input_names:
                self.context.set_tensor_address(name, int(self.dinputs[name]))
        except Exception:
            self.ctx.pop()
            if not(self.extra_lock is None):
                self.extra_lock.release()
            return False
        self.ctx.pop()
        if not(self.extra_lock is None):
            self.extra_lock.release()
        return True

