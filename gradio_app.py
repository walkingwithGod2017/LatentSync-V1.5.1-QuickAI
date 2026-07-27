import os
import torch
import gradio as gr
from pathlib import Path
from scripts.inference import build_pipeline, run_pipeline
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import threading
import time

CONFIG_PATH = Path("configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")
CONFIG = OmegaConf.load(CONFIG_PATH)
_PIPELINE = None
_PIPELINE_DTYPE = None
_PIPELINE_KEY = None
_PIPELINE_LOCK = threading.Lock()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


COMPILE_UNET = env_bool("LATENTSYNC_COMPILE_UNET", True)
COMPILE_MODE = os.getenv("LATENTSYNC_COMPILE_MODE", "default")
COMPILE_CACHE_DIR = os.getenv("LATENTSYNC_COMPILE_CACHE_DIR", "temp/compile_cache")
COMPILE_RECOMPILE_LIMIT = env_int("LATENTSYNC_COMPILE_RECOMPILE_LIMIT", 32)


def format_elapsed_time(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:.1f}s"
    return f"{seconds:.1f}s"


def get_pipeline(args):
    global _PIPELINE, _PIPELINE_DTYPE, _PIPELINE_KEY

    key = (
        CHECKPOINT_PATH.absolute().as_posix(),
        args.enable_deepcache,
        args.audio_embeds_cache_dir,
        args.compile_unet,
        args.compile_mode,
        args.compile_use_cudagraphs,
        args.compile_cache_dir,
        args.compile_recompile_limit,
    )
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        return _PIPELINE, _PIPELINE_DTYPE

    _PIPELINE, _PIPELINE_DTYPE = build_pipeline(CONFIG, args)
    _PIPELINE_KEY = key
    return _PIPELINE, _PIPELINE_DTYPE


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
):
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")  # Change the filename as needed

    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed)

    started_at = time.perf_counter()
    try:
        with _PIPELINE_LOCK:
            pipeline, dtype = get_pipeline(args)
            run_pipeline(pipeline, CONFIG, args, dtype)
        elapsed = time.perf_counter() - started_at
        elapsed_text = f"Generation time: {format_elapsed_time(elapsed)}"
        print(f"Processing completed successfully. {elapsed_text}")
        return output_path, elapsed_text
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    parser.add_argument("--audio_embeds_cache_dir", type=str, default="temp/cache/audio_embeds")
    parser.add_argument("--cache_dir", type=str, default="temp/cache")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile_unet", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default")
    parser.add_argument("--compile_use_cudagraphs", action="store_true")
    parser.add_argument("--compile_cache_dir", type=str, default=None)
    parser.add_argument("--compile_recompile_limit", type=int, default=None)

    argv = [
        "--inference_ckpt_path",
        CHECKPOINT_PATH.absolute().as_posix(),
        "--video_path",
        video_path,
        "--audio_path",
        audio_path,
        "--video_out_path",
        output_path,
        "--inference_steps",
        str(inference_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--seed",
        str(seed),
        "--temp_dir",
        "temp",
        "--audio_embeds_cache_dir",
        "temp/cache/audio_embeds",
        "--cache_dir",
        "temp/cache",
        "--enable_deepcache",
        "--compile_mode",
        COMPILE_MODE,
        "--compile_cache_dir",
        COMPILE_CACHE_DIR,
        "--compile_recompile_limit",
        str(COMPILE_RECOMPILE_LIMIT),
    ]
    if COMPILE_UNET:
        argv.append("--compile_unet")

    return parser.parse_args(argv)


# Create Gradio interface
with gr.Blocks(title="LatentSync demo") as demo:
    gr.Markdown(
        """
    <h1 align="center">LatentSync</h1>

    <div style="display:flex;justify-content:center;column-gap:4px;">
        <a href="https://github.com/bytedance/LatentSync">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
        </a> 
        <a href="https://arxiv.org/abs/2412.09262">
            <img src='https://img.shields.io/badge/arXiv-Paper-red'>
        </a>
    </div>
    """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video")
            audio_input = gr.Audio(label="Input Audio", type="filepath")

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=1.5,
                    step=0.1,
                    label="Guidance Scale",
                )
                inference_steps = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Inference Steps")

            with gr.Row():
                seed = gr.Number(value=1247, label="Random Seed", precision=0)

            process_btn = gr.Button("Process Video")

        with gr.Column():
            video_output = gr.Video(label="Output Video")
            generation_time_output = gr.Textbox(label="Generation Time", interactive=False)

            gr.Examples(
                examples=[
                    ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                    ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                    ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                ],
                inputs=[video_input, audio_input],
            )

    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            audio_input,
            guidance_scale,
            inference_steps,
            seed,
        ],
        outputs=[video_output, generation_time_output],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False)
