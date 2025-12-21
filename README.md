<div align="center">

<img src="assets/header.svg" alt="PersonaLive" width="100%">

<h2>Expressive Portrait Image Animation for Live Streaming</h2>

#### [Zhiyuan Li<sup>1,2,3</sup>](https://huai-chang.github.io/) · [Chi-Man Pun<sup>1,📪</sup>](https://cmpun.github.io/) · [Chen Fang<sup>2</sup>](http://fangchen.org/) · [Jue Wang<sup>2</sup>](https://scholar.google.com/citations?user=Bt4uDWMAAAAJ&hl=en) · [Xiaodong Cun<sup>3,📪</sup>](https://vinthony.github.io/academic/) 
<sup>1</sup> University of Macau  &nbsp;&nbsp; <sup>2</sup> [Dzine.ai](https://www.dzine.ai/)  &nbsp;&nbsp; <sup>3</sup> [GVC Lab, Great Bay University](https://gvclab.github.io/)

<a href='https://arxiv.org/abs/2512.11253'><img src='https://img.shields.io/badge/ArXiv-2512.11253-red'></a> <a href='https://huggingface.co/huaichang/PersonaLive'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107'></a> [![GitHub](https://img.shields.io/github/stars/GVCLab/PersonaLive?style=social)](https://github.com/GVCLab/PersonaLive)

<img src="assets/highlight.svg" alt="highlight" width="95%">

<img src="assets/demo_3.gif" width="46%"> &nbsp;&nbsp; <img src="assets/demo_2.gif" width="40.5%">
</div>

## 📋 TODO
- [ ] If you find PersonaLive useful or interesting, please give us a Star🌟! Your support drives us to keep improving.🍻
- [ ] Fix bugs (If you encounter any issues, please feel free to open an issue or contact me! 🙏)
- [ ] Enhance WebUI (Support reference image replacement).
- [x] **[2025.12.17]** 🔥 [ComfyUI-PersonaLive](https://github.com/okdalto/ComfyUI-PersonaLive) is now supported! (Thanks to [@okdalto](https://github.com/okdalto))
- [x] **[2025.12.15]** 🔥 Release `paper`!
- [x] **[2025.12.12]** 🔥 Release `inference code`, `config`, and `pretrained weights`!

## ⚙️ Framework
<img src="assets/overview.png" alt="Image 1" width="100%">


We present PersonaLive, a `real-time` and `streamable` diffusion framework capable of generating `infinite-length` portrait animations on a single `12GB GPU`.


## 🚀 Getting Started
### 🛠 Installation
```
# clone this repo
git clone https://github.com/GVCLab/PersonaLive
cd PersonaLive

# Create conda environment
conda create -n personalive python=3.10
conda activate personalive

# Install packages with pip
pip install -r requirements_base.txt
```

### ⏬ Download weights
Option 1: Download pre-trained weights of base models and other components ([sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) and [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)). You can run the following command to download weights automatically:
    
```bash
python tools/download_weights.py
```

Option 2: Download pre-trained weights into the `./pretrained_weights` folder from one of the below URLs:
    
<a href='https://drive.google.com/drive/folders/1GOhDBKIeowkMpBnKhGB8jgEhJt_--vbT?usp=drive_link'><img src='https://img.shields.io/badge/Google%20Drive-5B8DEF?style=for-the-badge&logo=googledrive&logoColor=white'></a> <a href='https://pan.baidu.com/s/1DCv4NvUy_z7Gj2xCGqRMkQ?pwd=gj64'><img src='https://img.shields.io/badge/Baidu%20Netdisk-3E4A89?style=for-the-badge&logo=baidu&logoColor=white'></a> <a href='https://www.alipan.com/s/jyJ9JBqS6Ty'><img src='https://img.shields.io/badge/Aliyun%20Drive-E67E22?style=for-the-badge&logo=alibabacloud&logoColor=white'></a> <a href='https://huggingface.co/huaichang/PersonaLive'><img src='https://img.shields.io/badge/HuggingFace-C8AC50?style=for-the-badge&logo=huggingface&logoColor=white'></a>

Finally, these weights should be organized as follows:
```
pretrained_weights
├── onnx
│   ├── unet_opt
│   │   ├── unet_opt.onnx
│   │   └── unet_opt.onnx.data
│   └── unet
├── personalive
│   ├── denoising_unet.pth
│   ├── motion_encoder.pth
│   ├── motion_extractor.pth
│   ├── pose_guider.pth
│   ├── reference_unet.pth
│   └── temporal_module.pth
├── sd-vae-ft-mse
│   ├── diffusion_pytorch_model.bin
│   └── config.json
├── sd-image-variations-diffusers
│   ├── image_encoder
│   │   ├── pytorch_model.bin
│   │   └── config.json
│   ├── unet
│   │   ├── diffusion_pytorch_model.bin
│   │   └── config.json
│   └── model_index.json
└── tensorrt
    └── unet_work.engine
```

### 🎞️ Offline Inference
```
python inference_offline.py
```
⚠️ Note for RTX 50-Series (Blackwell) Users: xformers is not yet fully compatible with the new architecture. To avoid crashes, please disable it by running:
```
python inference_offline.py --use_xformers False
```

To reduce video memory usage, use the `--batch_size` parameter:
```
python inference_offline.py --batch_size=16
```
**Recommended values:**  
`--batch_size=16` for GPUs with **8-12 GB** of VRAM  
`--batch_size=32` for GPUs with **12-16 GB** of VRAM  
The value must be a multiple of 4 (temporal_window_size)  

### 📸 Online Inference
#### 📦 Setup Web UI
```
# install Node.js 18+
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
nvm install 18

cd webcam
source start.sh
```

#### 🏎️ Acceleration (Optional)
Converting the model to TensorRT can significantly speed up inference (~ 2x ⚡️). Building the engine may take about `20 minutes` depending on your device. Note that TensorRT optimizations may lead to slight variations or a small drop in output quality.
```
pip install -r requirement_trt.txt

python torch2trt.py
```
*The provided TensorRT model is from an `H100`. We recommend `ALL users` (including H100 users) re-run `python torch2trt.py` locally to ensure best compatibility.*

#### ▶️ Start Streaming
```
python inference_online.py --acceleration none (for RTX 50-Series) or xformers or tensorrt
```
Then open `http://0.0.0.0:7860` in your browser. (*If `http://0.0.0.0:7860` does not work well, try `http://localhost:7860`)

**How to use**: Upload Image ➡️ Fuse Reference ➡️ Start Animation ➡️ Enjoy! 🎉
<div align="center">
  <img src="assets/guide.png" alt="PersonaLive" width="60%">
</div>

**Regarding Latency**: Latency varies depending on your device's computing power. You can try the following methods to optimize it:

1. Lower the "Driving FPS" setting in the WebUI to reduce the computational workload.
2. You can increase the multiplier (e.g., set to `num_frames_needed * 4` or higher) to better match your device's inference speed. https://github.com/GVCLab/PersonaLive/blob/6953d1a8b409f360a3ee1d7325093622b29f1e22/webcam/util.py#L73

## 📚 Community Contribution

Special thanks to the community for providing helpful setups! 🥂

* **Windows + RTX 50-Series Guide**: Thanks to [@dknos](https://github.com/dknos) for providing a [detailed guide](https://github.com/GVCLab/PersonaLive/issues/10#issuecomment-3662785532) on running this project on Windows with Blackwell GPUs.

* **TensorRT on Windows**: If you are trying to convert TensorRT models on Windows, [this discussion](https://github.com/GVCLab/PersonaLive/issues/8) might be helpful. Special thanks to [@MaraScott](https://github.com/MaraScott) and [@Jeremy8776](https://github.com/Jeremy8776) for their insights.
  
* **ComfyUI**: Thanks to [@okdalto](https://github.com/okdalto) for helping implement the [ComfyUI-PersonaLive](https://github.com/okdalto/ComfyUI-PersonaLive) support.

## 🎬 More Results
#### 👀 Visualization results

<table width="100%">
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/cdc885ef-5e1c-4139-987a-2fa50fefd6a4" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/014f7bae-74ce-4f56-8621-24bc76f3c123" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
  </tr>
</table>
<table width="100%">
  <tr>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/1e6a0809-15d2-4cab-ae8f-8cf1728c6281" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/d9cf265d-9db0-4f83-81da-be967bbd5f26" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/86235139-b63e-4f26-b09c-d218466e8e24" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/238785de-3b4c-484e-9ad0-9d90e7962fee" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
  </tr>
  <tr>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/c71c4717-d528-4a98-b132-2b0ec8cec22d" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/7e11fe71-fd16-4011-a6b2-2dbaf7e343fb" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/f62e2162-d239-4575-9514-34575c16301c" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/813e7fbd-37e9-47d7-a270-59887fafeca5" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
  </tr>
</table>

#### 🤺 Comparisons

<table width="100%">
  <tr>
    <td width="100%">
      <video src="https://github.com/user-attachments/assets/36407cf9-bf82-43ff-9508-a794d223d3f7" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
  </tr>
  <tr>
    <td width="100%">
      <video src="https://github.com/user-attachments/assets/3be99b91-c6a1-4ca4-89e9-8fad42bb9583" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
  </tr>
  <tr>
    <td width="100%">
      <video src="https://github.com/user-attachments/assets/5bd21fe4-96ae-4be6-bf06-a7c476b04ec9" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
  </tr>
</table>


## ⭐ Citation
If you find PersonaLive useful for your research, welcome to cite our work using the following BibTeX:
```bibtex
@article{li2025personalive,
  title={PersonaLive! Expressive Portrait Image Animation for Live Streaming},
  author={Li, Zhiyuan and Pun, Chi-Man and Fang, Chen and Wang, Jue and Cun, Xiaodong},
  journal={arXiv preprint arXiv:2512.11253},
  year={2025}
}
```

## ❤️ Acknowledgement
This code is mainly built upon [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [X-NeMo](https://byteaigc.github.io/X-Portrait2/), [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), [RAIN](https://pscgylotti.github.io/pages/RAIN/) and [LivePortrait](https://github.com/KlingTeam/LivePortrait), thanks to their invaluable contributions.
