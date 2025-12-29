import os
import signal
import sys
import json

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2
import threading
import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import mimetypes
import torch

from webcam.config import config, Args
from webcam.util import pil_to_frame, bytes_to_pil, is_firefox, bytes_to_tensor
from webcam.connection_manager import ConnectionManager, ServerFullException
import multiprocessing as mp

mimetypes.add_type("application/javascript", ".js")

THROTTLE = 0.001 


class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()

        self.produce_predictions_stop_event = None
        self.produce_predictions_task = None
        self.shutdown_event = asyncio.Event()

        self.init_app()

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )

                sender_task = asyncio.create_task(push_results_to_client(user_id, websocket))

                if self.produce_predictions_task is None or self.produce_predictions_task.done():
                    start_prediction_thread(user_id)

                await handle_websocket_input(user_id, websocket)

            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            except WebSocketDisconnect:
                logging.info(f"User disconnected: {user_id}")
            except Exception as e:
                logging.error(f"WS Error: {e}")
            finally:
                if 'sender_task' in locals():
                    sender_task.cancel()
                
                await self.conn_manager.disconnect(user_id, self.pipeline)
                
                if self.produce_predictions_stop_event is not None:
                    self.produce_predictions_stop_event.set()
                logging.info(f"Cleaned up user: {user_id}")

        async def handle_websocket_input(user_id: uuid.UUID, websocket: WebSocket):
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")
            
            try:
                while True:
                    message = await websocket.receive()

                    if "text" in message:
                        try:
                            text_data = message["text"]
                            data = json.loads(text_data)
                            status = data.get("status")

                            if status == "pause":
                                params = SimpleNamespace(**{"restart": True})
                                await self.conn_manager.update_data(user_id, params)
                            elif status == "resume":
                                await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                        except Exception as e:
                            logging.error(f"JSON Parse Error: {e}")

                    elif "bytes" in message:
                        image_data = message["bytes"]
                        if len(image_data) > 0:
                            input_tensor = bytes_to_tensor(image_data)
                            params = SimpleNamespace()
                            params.image = input_tensor
                            self.pipeline.accept_new_params(params)

            except WebSocketDisconnect:
                raise 
            except Exception as e:
                logging.error(f"Input Loop Error: {e}")
                raise

        async def push_results_to_client(user_id: uuid.UUID, websocket: WebSocket):
            MIN_FPS = 10
            MAX_FPS = 30
            SMOOTHING = 0.8  # EMA smoothing factor

            last_burst_time = time.time()
            last_queue_size = 0
            sleep_time = 1 / 40  # Initial guess
            
            last_frame_time = None 
            frame_time_list = []

            ema_frame_interval = sleep_time

            try:
                while True:
                    queue_size = await self.conn_manager.get_output_queue_size(user_id)
                    if queue_size > last_queue_size:
                        current_burst_time = time.time()
                        elapsed = current_burst_time - last_burst_time

                        if queue_size > 0 and elapsed > 0:
                            raw_interval = elapsed / queue_size
                            ema_frame_interval = SMOOTHING * ema_frame_interval + (1 - SMOOTHING) * raw_interval
                            sleep_time = min(max(ema_frame_interval, 1 / MAX_FPS), 1 / MIN_FPS)
                        
                        last_burst_time = current_burst_time
                    
                    last_queue_size = queue_size

                    frame = await self.conn_manager.get_frame(user_id)
                    if frame is None:
                        await asyncio.sleep(0.001)
                        continue

                    await websocket.send_bytes(frame)

                    if last_frame_time is None:
                        last_frame_time = time.time()
                    else:
                        frame_time_list.append(time.time() - last_frame_time)
                        if len(frame_time_list) > 100:
                            frame_time_list.pop(0)
                        last_frame_time = time.time()
                    
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.error(f"Push Result Error: {e}")

        def start_prediction_thread(user_id):
            self.produce_predictions_stop_event = threading.Event()
            
            def prediction_loop(uid, loop, stop_event):
                while not stop_event.is_set():
                    images = self.pipeline.produce_outputs()
                    if len(images) == 0:
                        time.sleep(THROTTLE)
                        continue
                    
                    frames = list(map(pil_to_frame, images))
                    asyncio.run_coroutine_threadsafe(
                        self.conn_manager.put_frames_to_output_queue(uid, frames),
                        loop
                    )

            self.produce_predictions_task = asyncio.create_task(asyncio.to_thread(
                prediction_loop, user_id, asyncio.get_running_loop(), self.produce_predictions_stop_event
            ))

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/settings")
        async def settings():
            info_schema = pipeline.Info.schema()
            info = pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = pipeline.InputParams.schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                }
            )
        
        @self.app.post("/api/upload_reference_image")
        async def upload_reference_image(ref_image: UploadFile = File(...)):
            try:
                data = await ref_image.read()
                img = bytes_to_pil(data)
                self.pipeline.fuse_reference(img)
                return {"status": "ok"}
            except Exception as e:
                logging.error(f"Reference image error: {e}")
                raise HTTPException(status_code=500, detail="Failed to process reference image")

        @self.app.post("/api/reset")
        async def reset():
            try:
                self.pipeline.reset()
            except Exception as e:
                print(f"Reset Error: {e}")
                raise HTTPException(status_code=500, detail='Failed to reset pipeline')

        if not os.path.exists("./webcam/frontend/public"):
            os.makedirs("./webcam/frontend/public")

        self.app.mount(
            "/", StaticFiles(directory="./webcam/frontend/public", html=True), name="public"
        )

        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.cleanup()

    async def cleanup(self):
        print("[App] Starting cleanup process...")
        self.shutdown_event.set()
        
        if self.produce_predictions_stop_event is not None:
            self.produce_predictions_stop_event.set()
        
        if self.produce_predictions_task is not None:
            self.produce_predictions_task.cancel()
            try:
                await self.produce_predictions_task
            except asyncio.CancelledError:
                pass
        
        try:
            await self.conn_manager.disconnect_all(self.pipeline)
        except Exception as e:
            print(f"[App] Error during disconnect_all: {e}")
        
        print("[App] Cleanup completed")

app_instance = None

def signal_handler(signum, frame):
    print(f"\n[Main] Received signal {signum}, shutting down gracefully...")
    if app_instance:
        import threading
        def trigger_cleanup():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(app_instance.cleanup())
                loop.close()
            except Exception as e:
                print(f"[Main] Error during cleanup: {e}")
        
        cleanup_thread = threading.Thread(target=trigger_cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        cleanup_thread.join(timeout=5)
    
    sys.exit(0)


if __name__ == "__main__":
    import uvicorn
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.acceleration == "tensorrt":
        from webcam.vid2vid_trt import Pipeline
    else:
        from webcam.vid2vid import Pipeline
    
    pipeline = Pipeline(config, device)
    
    app_obj = App(config, pipeline)
    app = app_obj.app
    app_instance = app_obj
    
    print('init done')

    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=config.reload,
            ssl_certfile=config.ssl_certfile,
            ssl_keyfile=config.ssl_keyfile,
        )
    except KeyboardInterrupt:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(app_obj.cleanup())
            loop.close()
        except Exception as e:
            print(f"[Main] Error during cleanup: {e}")
        sys.exit(0)
    except Exception as e:
        print(f"[Main] Error: {e}")
        sys.exit(1)