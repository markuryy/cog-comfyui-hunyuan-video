from cog import BaseModel, Input, Path, Secret
import os
import shutil
import subprocess
import time
import torch
from typing import Optional
from zipfile import ZipFile, is_zipfile


# We return a path to our tarred LoRA weights at the end
class TrainingOutput(BaseModel):
    weights: Path


# Paths for input data, output data, etc.
INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_CACHE = "ckpts"

# Hunyuan model files to download – adjust for your environment
MODEL_FILES = ["hunyuan-video-t2v-720p.tar", "text_encoder.tar", "text_encoder_2.tar"]
BASE_URL = "https://weights.replicate.delivery/default/hunyuan-video/ckpts/"


def train(
    input_videos: Path = Input(
        description="A zip file containing your video dataset plus .txt captions. Filenames must match, e.g. segment10.mp4 and segment10.txt.",
        default=None,
    ),
    epochs: int = Input(
        description="Number of training epochs (one epoch here = one pass over entire dataset).",
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
        description="Batch size for training.",
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
        description="Enable gradient checkpointing to save memory.",
        default=True,
    ),
    timestep_sampling: str = Input(
        description="Method to sample timesteps for training.",
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift"],
    ),
    seed: int = Input(
        description="Random seed (use <= 0 for a random choice).",
        default=42,
    ),
    hub_model_id: str = Input(
        description="Optional: Hugging Face repository name to upload trained LoRA.",
        default="",
    ),
    hf_token: Secret = Input(
        description="Optional: Hugging Face token to authenticate uploads.",
        default=None,
    ),
) -> TrainingOutput:
    """
    A simple train.py for Hunyuan LoRA training via musubi-tuner.

    1. Downloads base model weights if needed.
    2. Unpacks input videos & captions from a .zip.
    3. Runs latent caching & text encoder caching steps.
    4. Trains a LoRA model.
    5. Archives results as a .tar.
    6. Optionally uploads results to HF if a token/repo ID are provided.
    """
    if not input_videos:
        raise ValueError("input_videos must be provided.")

    # Clean up from any prior run
    clean_up()

    # Ensure base Hunyuan model weights are present
    download_weights()

    # Check/handle random seed
    if seed <= 0:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    # Create directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract the video & caption data
    extract_zip(input_videos, INPUT_DIR)

    # 1) Latent pre-caching
    print("Running latent pre-caching command...")
    latent_args = [
        "python",
        "cache_latents.py",
        "--dataset_config",
        "train.toml",  # Expects a 'train.toml' config in your project
        "--vae",
        f"{MODEL_CACHE}/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
        "--vae_chunk_size",
        "32",
        "--vae_tiling",
    ]
    subprocess.run(latent_args, check=True)

    # 2) Text encoder pre-caching
    print("Running text encoder output pre-caching command...")
    text_encoder_args = [
        "python",
        "cache_text_encoder_outputs.py",
        "--dataset_config",
        "train.toml",
        "--text_encoder1",
        f"{MODEL_CACHE}/text_encoder",
        "--text_encoder2",
        f"{MODEL_CACHE}/text_encoder_2",
        "--batch_size",
        str(batch_size),
    ]
    subprocess.run(text_encoder_args, check=True)

    # 3) Perform LoRA training
    print("Running training command...")
    training_args = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "1",
        "--mixed_precision",
        "bf16",
        "hv_train_network.py",
        "--dit",
        f"{MODEL_CACHE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
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

    # 4) Optionally upload to Hugging Face
    if hf_token and hub_model_id:
        handle_hf_upload(hub_model_id, hf_token)

    # 5) Archive results
    output_path = "/tmp/trained_model.tar"
    os.system(f"tar -cvf {output_path} -C {OUTPUT_DIR} .")

    # Return the path to the archived model
    return TrainingOutput(weights=Path(output_path))


def clean_up():
    """
    Removes INPUT_DIR and OUTPUT_DIR if they exist.
    """
    if os.path.exists(INPUT_DIR):
        shutil.rmtree(INPUT_DIR)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)


def download_weights():
    """
    Download base Hunyuan model weights if they aren't already present in MODEL_CACHE.
    """
    os.makedirs(MODEL_CACHE, exist_ok=True)
    for model_file in MODEL_FILES:
        filename_no_ext = model_file.split(".")[0]
        dest_path = os.path.join(MODEL_CACHE, filename_no_ext)
        if not os.path.exists(dest_path):
            url = BASE_URL + model_file
            print(f"Downloading {url} to {MODEL_CACHE}")
            subprocess.check_call(["pget", "-xvf", url, MODEL_CACHE])


def extract_zip(zip_path: Path, extraction_dir: str):
    """
    Extract videos and .txt captions from the provided zip into extraction_dir.
    """
    if not is_zipfile(zip_path):
        raise ValueError("The provided input_videos must be a zip file")

    os.makedirs(extraction_dir, exist_ok=True)
    final_videos_path = os.path.join(extraction_dir, "videos")
    os.makedirs(final_videos_path, exist_ok=True)

    file_count = 0
    with ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            # Skip Mac hidden system files
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, final_videos_path)
                file_count += 1

    print(f"Extracted {file_count} files from zip into: {final_videos_path}")


def handle_hf_upload(hub_model_id: str, hf_token: Secret):
    """
    Simple huggingface-cli upload. Adjust or replace with huggingface_hub if needed.
    """
    token = hf_token.get_secret_value()
    print(f"Logging into Hugging Face with token and uploading to {hub_model_id}")
    os.system(f"huggingface-cli login --token {token}")
    # The command below assumes hub_model_id is an existing model on HF.
    # For new repos, you may need to create it first, e.g. huggingface-cli repo create <repo>.
    os.system(f"huggingface-cli upload {OUTPUT_DIR} --repo-id {hub_model_id} --folder")
