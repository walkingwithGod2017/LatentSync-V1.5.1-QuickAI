# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import shutil
import sys
import tempfile
import time
import subprocess
import uuid
from pathlib import Path as LocalPath

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/chunyu-li/LatentSync/model.tar"
VGG16_CHECKPOINT = "vgg16-397923af.pth"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    pget = shutil.which("pget")
    if pget is None:
        raise FileNotFoundError("pget not found")
    subprocess.check_call([pget, "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def ensure_auxiliary_checkpoint_link():
    checkpoint_dir = LocalPath.home() / ".cache" / "torch" / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source = LocalPath.cwd() / "checkpoints" / "auxiliary" / VGG16_CHECKPOINT
    target = checkpoint_dir / VGG16_CHECKPOINT
    if target.exists() or target.is_symlink():
        return

    try:
        target.symlink_to(source)
    except OSError as exc:
        if source.is_file():
            shutil.copyfile(source, target)
        else:
            print(f"Could not create auxiliary checkpoint link: {type(exc).__name__} - {exc}")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Soft links for the auxiliary models
        ensure_auxiliary_checkpoint_link()

    def predict(
        self,
        video: Path = Input(description="Input video", default=None),
        audio: Path = Input(description="Input audio to ", default=None),
        guidance_scale: float = Input(description="Guidance scale", ge=1, le=3, default=2.0),
        inference_steps: int = Input(description="Inference steps", ge=20, le=50, default=20),
        seed: int = Input(description="Set to 0 for Random seed", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if video is None:
            raise ValueError("Input video is required")
        if audio is None:
            raise ValueError("Input audio is required")

        video_path = str(video)
        audio_path = str(audio)
        config_path = "configs/unet/stage2.yaml"
        ckpt_path = "checkpoints/latentsync_unet.pt"
        output_path = LocalPath(tempfile.gettempdir()) / f"latentsync_{uuid.uuid4().hex}.mp4"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.inference",
                "--unet_config_path",
                config_path,
                "--inference_ckpt_path",
                ckpt_path,
                "--guidance_scale",
                str(guidance_scale),
                "--video_path",
                video_path,
                "--audio_path",
                audio_path,
                "--video_out_path",
                str(output_path),
                "--seed",
                str(seed),
                "--inference_steps",
                str(inference_steps),
            ],
            check=True,
        )
        return Path(str(output_path))
