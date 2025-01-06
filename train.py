import os
import shutil
import subprocess
import time
import torch
import sys
from typing import Optional
from zipfile import ZipFile, is_zipfile

from cog import BaseModel, Input, Path, Secret


# We return a path to our tarred LoRA weights at the end
class TrainingOutput(BaseModel):
    weights: Path


# Directories used for input/output data and cached model files
INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_CACHE = "ckpts"

# Hunyuan base model files, adjust as desired:
MODEL_FILES = ["hunyuan-video-t2v-720p.tar", "text_encoder.tar", "text_encoder_2.tar"]
BASE_URL = "https://weights.replicate.delivery/default/hunyuan-video/ckpts/"

# If your scripts are in a folder named "musubi-tuner", you can add:
sys.path.append("musubi-tuner")


def train(
    input_videos: Path = Input(
        description="A zip file containing videos and matching .txt captions.",
        default=None,
    ),
    epochs: int = Input(
        description="Number of training epochs (each approximately one pass over dataset).",
        default=16,
        ge=1,
        le=2000,
    ),
    rank: int = Input(
        description="LoRA rank for Hunyuan training.",
        default=32,
        ge=1,
        le=128,
    ),
    batch_size: int = Input(
        description="Batch size for training",
        default=4,
        ge=1,
        le=8,
    ),
    learning_rate: float = Input(
        description="Learning rate for training.",
        default=1e-3,
        ge=1e-5,
        le=1,
    ),
    optimizer: str = Input(
        description="Optimizer type",
        default="adamw8bit",
        choices=["adamw", "adamw8bit", "AdaFactor", "adamw16bit"],
    ),
    gradient_checkpointing: bool = Input(
        description="Enable gradient checkpointing to reduce memory usage.",
        default=True,
    ),
    timestep_sampling: str = Input(
        description="Method to sample timesteps for training.",
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift"],
    ),
    seed: int = Input(
        description="Random seed (use <=0 for a random pick).",
        default=42,
    ),
    hub_model_id: str = Input(
        description="(Optional) Hugging Face repo to upload trained LoRA.",
        default="",
    ),
    hf_token: Secret = Input(
        description="(Optional) Hugging Face token to upload the LoRA.",
        default=None,
    ),
) -> TrainingOutput:
    """
    Minimal Hunyuan LoRA training script using musubi-tuner. Mirrors logic from the predict.py.
    1. Ensure base weights are present (downloads if needed).
    2. Extract the provided zip of videos & .txt captions into 'input/videos'.
    3. Run musubi-tuner/cache_latents.py.
    4. Run musubi-tuner/cache_text_encoder_outputs.py.
    5. Train LoRA with musubi-tuner/hv_train_network.py.
    6. Archive the results as .tar.
    7. Optionally upload to Hugging Face if credentials are provided.
    """

    if not input_videos:
        raise ValueError(
            "You must provide input_videos (a zip with videos & .txt captions)."
        )

    # Clean up old run output
    clean_up()

    # Download base model weights if needed
    download_weights()

    # Handle seed
    if seed <= 0:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    # Ensure we have a default train.toml if it doesn't exist
    if not os.path.exists("train.toml"):
        print("No train.toml found; creating a default config.")
        with open("train.toml", "w") as f:
            f.write(
                """[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "./input/videos"
cache_directory = "./input/cache_directory"
target_frames = [1, 25, 45]
frame_extraction = "head"
"""
            )

    # Prepare directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract zip of videos + captions
    extract_zip(input_videos, INPUT_DIR)

    # Latent pre-caching
    print("Running latent pre-caching command...")
    latent_args = [
        "python",
        "musubi-tuner/cache_latents.py",  # or just "cache_latents.py" if local
        "--dataset_config",
        "train.toml",
        "--vae",
        os.path.join(MODEL_CACHE, "hunyuan-video-t2v-720p/vae/pytorch_model.pt"),
        "--vae_chunk_size",
        "32",
        "--vae_tiling",
    ]
    subprocess.run(latent_args, check=True)

    # Text encoder pre-caching
    print("Running text encoder pre-caching command...")
    text_encoder_args = [
        "python",
        "musubi-tuner/cache_text_encoder_outputs.py",  # or just "cache_text_encoder_outputs.py"
        "--dataset_config",
        "train.toml",
        "--text_encoder1",
        os.path.join(MODEL_CACHE, "text_encoder"),
        "--text_encoder2",
        os.path.join(MODEL_CACHE, "text_encoder_2"),
        "--batch_size",
        str(batch_size),
    ]
    subprocess.run(text_encoder_args, check=True)

    # LoRA training
    print("Running training command...")
    training_args = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "1",
        "--mixed_precision",
        "bf16",
        "musubi-tuner/hv_train_network.py",  # or "hv_train_network.py" local
        "--dit",
        os.path.join(
            MODEL_CACHE,
            "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        ),
        "--dataset_config",
        "train.toml",
        "--sdpa",
        "--mixed_precision",
        "bf16",
        "--fp8_base",
        "--optimizer_type",
        optimizer,
        "--learning_rate",
        str(learning_rate),
        "--max_data_loader_n_workers",
        "2",
        "--persistent_data_loader_workers",
        "--network_module",
        "networks.lora",
        "--network_dim",
        str(rank),
        "--timestep_sampling",
        timestep_sampling,
        "--discrete_flow_shift",
        "1.0",
        "--max_train_epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--output_dir",
        OUTPUT_DIR,
        "--output_name",
        "lora",
    ]
    if gradient_checkpointing:
        training_args.append("--gradient_checkpointing")

    subprocess.run(training_args, check=True)

    # Optionally upload to HF
    if hf_token and hub_model_id:
        handle_hf_upload(hub_model_id, hf_token)

    # Create tarball of the results
    output_path = "/tmp/trained_model.tar"
    print(f"Archiving LoRA outputs to {output_path}")
    os.system(f"tar -cvf {output_path} -C {OUTPUT_DIR} .")

    return TrainingOutput(weights=Path(output_path))


def clean_up():
    """Removes INPUT_DIR and OUTPUT_DIR if they exist."""
    if os.path.exists(INPUT_DIR):
        shutil.rmtree(INPUT_DIR)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)


def download_weights():
    """Download base Hunyuan model weights if not already cached."""
    os.makedirs(MODEL_CACHE, exist_ok=True)
    for model_file in MODEL_FILES:
        filename_no_ext = model_file.split(".")[0]
        dest_path = os.path.join(MODEL_CACHE, filename_no_ext)
        if not os.path.exists(dest_path):
            url = BASE_URL + model_file
            print(f"Downloading {url} to {MODEL_CACHE}")
            subprocess.check_call(["pget", "-xvf", url, MODEL_CACHE])


def extract_zip(zip_path: Path, extraction_dir: str):
    """Extract videos & .txt captions from the provided zip file."""
    if not is_zipfile(zip_path):
        raise ValueError("The provided input_videos must be a zip file.")

    os.makedirs(extraction_dir, exist_ok=True)
    final_videos_path = os.path.join(extraction_dir, "videos")
    os.makedirs(final_videos_path, exist_ok=True)

    file_count = 0
    with ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, final_videos_path)
                file_count += 1

    print(f"Extracted {file_count} files into: {final_videos_path}")


def handle_hf_upload(hub_model_id: str, hf_token: Secret):
    """Simple huggingface-cli upload."""
    token = hf_token.get_secret_value()
    print(f"Logging into Hugging Face with token and uploading to {hub_model_id}")
    os.system(f"huggingface-cli login --token {token}")
    # For new repos, you may need to create it first: huggingface-cli repo create <hub_model_id>.
    os.system(f"huggingface-cli upload {OUTPUT_DIR} --repo-id {hub_model_id} --folder")
