# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper


def get_inference_dtype():
    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    return torch.float16 if is_fp16_supported else torch.float32


def configure_compile_cache(args):
    compile_cache_dir = os.path.abspath(
        getattr(args, "compile_cache_dir", None) or os.path.join(args.temp_dir, "compile_cache")
    )
    torchinductor_cache_dir = os.path.join(compile_cache_dir, "torchinductor")
    triton_cache_dir = os.path.join(compile_cache_dir, "triton")
    os.makedirs(torchinductor_cache_dir, exist_ok=True)
    os.makedirs(triton_cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = torchinductor_cache_dir
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
    return compile_cache_dir


def maybe_compile_unet(pipeline, args):
    if not getattr(args, "compile_unet", False):
        return pipeline

    compile_cache_dir = configure_compile_cache(args)
    compile_mode = getattr(args, "compile_mode", "default")
    compile_use_cudagraphs = getattr(args, "compile_use_cudagraphs", False)
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
    print("Compiling UNet with torch.compile(fullgraph=False, dynamic=False)")
    if compile_use_cudagraphs:
        print(f"Compile mode: {compile_mode}")
        compile_kwargs = {"mode": compile_mode}
    else:
        print("Compile CUDAGraphs: disabled")
        if compile_mode != "default":
            print(f"Compile mode '{compile_mode}' ignored unless --compile_use_cudagraphs is set")
        compile_kwargs = {"options": {"triton.cudagraphs": False}}
    print(f"Compile cache dir: {compile_cache_dir}")
    pipeline.unet = torch.compile(pipeline.unet, fullgraph=False, dynamic=False, **compile_kwargs)
    return pipeline


def build_pipeline(config, args, dtype=None):
    dtype = dtype or get_inference_dtype()
    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        audio_embeds_cache_dir=getattr(args, "audio_embeds_cache_dir", "temp/cache/audio_embeds"),
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    pipeline = maybe_compile_unet(pipeline, args)

    # use DeepCache
    if args.enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
        pipeline._deepcache_helper = helper

    return pipeline, dtype


def run_pipeline(pipeline, config, args, dtype):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        video_fps=config.data.video_fps,
        audio_sample_rate=config.data.audio_sample_rate,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp_dir,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", False),
    )


def main(config, args):
    pipeline, dtype = build_pipeline(config, args)
    return run_pipeline(pipeline, config, args, dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    parser.add_argument("--audio_embeds_cache_dir", type=str, default="temp/cache/audio_embeds")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile_unet", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default")
    parser.add_argument("--compile_use_cudagraphs", action="store_true")
    parser.add_argument("--compile_cache_dir", type=str, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
