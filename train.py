import os
import shutil
import subprocess
import time
import torch
import sys
from typing import Optional
from zipfile import ZipFile, is_zipfile
from huggingface_hub import HfApi
from cog import BaseModel, Input, Path, Secret


# We return a path to our tarred LoRA weights at the end
class TrainingOutput(BaseModel):
    weights: Path


# Directories used for input/output data and cached model files
INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_CACHE = "ckpts"

# Hunyuan model files to download
MODEL_FILES = ["hunyuan-video-t2v-720p.tar", "text_encoder.tar", "text_encoder_2.tar"]
BASE_URL = "https://weights.replicate.delivery/default/hunyuan-video/ckpts/"
JOB_DIR = Path("hunyuan-lora-for-hf")

# If your scripts are in "musubi-tuner", you can add it to the path:
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
    hf_repo_id: str = Input(
        description="Hugging Face repository ID, if you'd like to upload the trained LoRA to Hugging Face. For example, lucataco/flux-dev-lora. If the given repo does not exist, a new public repo will be created.",
        default=None,
    ),
    hf_token: Secret = Input(
        description="Hugging Face token, if you'd like to upload the trained LoRA to Hugging Face.",
        default=None,
    ),
) -> TrainingOutput:
    """
    Minimal Hunyuan LoRA training script using musubi-tuner.

    Steps:
    1. Clean up old run output if present.
    2. Download base weights if needed.
    3. Extract input videos & .txt files.
    4. Possibly create train.toml if not found. (We'll fix the default so it won't fail on short videos.)
    5. Cache latents, text encoder outputs.
    6. Train LoRA with musubi-tuner/hv_train_network.py.
    7. Convert to ComfyUI-compatible safetensors using musubi-tuner/convert_lora.py.
    8. Tar everything and return.
    9. Optionally push to HF if desired.
    """

    if not input_videos:
        raise ValueError("You must provide a zip with videos & .txt captions.")

    # 1. Clean up old outputs
    clean_up()

    # 2. Download base weights
    download_weights()

    # 3. Random seed logic
    if seed <= 0:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    # 4. Create train.toml if missing (FIX HERE to avoid zero video frames)
    if not os.path.exists("train.toml"):
        print("Creating default train.toml...")
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
# Instead of forcing frames 1,25,45, switch to uniform extraction of 3 frames
frame_extraction = "uniform"
frames_per_clip = 3
"""
            )

    # 5. Extract videos & text
    extract_zip(input_videos, INPUT_DIR)

    # 6. Cache latents
    latent_args = [
        "python",
        "musubi-tuner/cache_latents.py",
        "--dataset_config",
        "train.toml",
        "--vae",
        os.path.join(MODEL_CACHE, "hunyuan-video-t2v-720p/vae/pytorch_model.pt"),
        "--vae_chunk_size",
        "32",
        "--vae_tiling",
    ]
    subprocess.run(latent_args, check=True)

    # 7. Cache text encoder outputs
    text_encoder_args = [
        "python",
        "musubi-tuner/cache_text_encoder_outputs.py",
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

    # 8. Perform LoRA training
    print("Running LoRA training...")
    training_args = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "1",
        "--mixed_precision",
        "bf16",
        "musubi-tuner/hv_train_network.py",
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

    # 9. Convert the resulting LoRA to ComfyUI-compatible format
    original_lora_path = os.path.join(OUTPUT_DIR, "lora.safetensors")
    if os.path.exists(original_lora_path):
        converted_lora_path = os.path.join(OUTPUT_DIR, "lora_comfyui.safetensors")
        print(
            f"Converting from {original_lora_path} -> {converted_lora_path} (ComfyUI format)"
        )
        convert_args = [
            "python",
            "musubi-tuner/convert_lora.py",
            "--input",
            original_lora_path,
            "--output",
            converted_lora_path,
            "--target",
            "other",  # "other" -> diffusers style (ComfyUI)
        ]
        subprocess.run(convert_args, check=True)
    else:
        print("Warning: lora.safetensors not found, skipping conversion.")


    # 11. Archive final results
    output_path = "/tmp/trained_model.tar"
    print(f"Archiving LoRA outputs to {output_path}")
    os.system(f"tar -cvf {output_path} -C {OUTPUT_DIR} .")

    # 10. If we have HF token and ID, upload to HF
    if hf_token and hf_repo_id:
        shutil.move(os.path.join(OUTPUT_DIR, "lora.safetensors"), JOB_DIR / Path("lora.safetensors"))
        handle_hf_upload(hf_repo_id, hf_token)
        
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
    """Extract videos & .txt captions from the provided zip."""
    if not is_zipfile(zip_path):
        raise ValueError("The provided input_videos must be a zip file.")

    os.makedirs(extraction_dir, exist_ok=True)
    final_videos_path = os.path.join(extraction_dir, "videos")
    os.makedirs(final_videos_path, exist_ok=True)

    file_count = 0
    with ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            # skip Mac hidden system files
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, final_videos_path)
                file_count += 1

    print(f"Extracted {file_count} total files to: {final_videos_path}")


def handle_hf_upload(hf_repo_id: str, hf_token: Secret):
    """Simple huggingface-cli upload."""
    if hf_token is not None and hf_repo_id is not None:
        try:
            handle_hf_readme(hf_repo_id)
            print(f"Uploading to Hugging Face: {hf_repo_id}")
            api = HfApi()

            repo_url = api.create_repo(
                hf_repo_id,
                private=False,
                exist_ok=True,
                token=hf_token.get_secret_value(),
            )

            print(f"HF Repo URL: {repo_url}")

            api.upload_folder(
                repo_id=hf_repo_id,
                folder_path=JOB_DIR,
                repo_type="model",
                use_auth_token=hf_token.get_secret_value(),
            )
        except Exception as e:
            print(f"Error uploading to Hugging Face: {str(e)}")
            
def handle_hf_readme(hf_repo_id: str):
    readme_path = JOB_DIR / Path("README.md")
    license_path = Path("hf-lora-readme-template.md")
    shutil.copy(license_path, readme_path)

    content = readme_path.read_text()
    content = content.replace("[hf_repo_id]", hf_repo_id)

    repo_parts = hf_repo_id.split("/")
    if len(repo_parts) > 1:
        title = repo_parts[1].replace("-", " ").title()
        content = content.replace("[title]", title)
    else:
        content = content.replace("[title]", hf_repo_id)

    print(content)

    readme_path.write_text(content)
