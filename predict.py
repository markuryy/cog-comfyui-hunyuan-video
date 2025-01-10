import os
import json
import mimetypes
import shutil
import re
import requests
import tarfile
import tempfile
from typing import Any

from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import seed as seed_helper
from huggingface_hub import HfApi

# Directories for inputs/outputs
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/quicktime", ".mov")

api_json_file = "t2v-lora.json"

# Ensure HF Hub is online for LoRA downloads
if "HF_HUB_OFFLINE" in os.environ:
    del os.environ["HF_HUB_OFFLINE"]

class Predictor(BasePredictor):
    def setup(self):
        """
        Start ComfyUI, ensuring it doesn't attempt to download our local LoRA file
        before running. We do this by blanking out node 79's "lora_name" field so the
        weight downloader never sees it.
        """
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Manually download the LLAVA model if it doesn't exist
        llava_model_path = os.path.join("ComfyUI", "models", "clip_vision", "llava_llama3_fp8_scaled.safetensors")
        if not os.path.exists(llava_model_path):
            print("Downloading LLAVA text encoder...")
            os.makedirs(os.path.dirname(llava_model_path), exist_ok=True)
            llava_url = "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors"
            resp = requests.get(llava_url)
            resp.raise_for_status()
            with open(llava_model_path, "wb") as f:
                f.write(resp.content)
            print("✅ Downloaded LLAVA text encoder")

        # 1. Load the main workflow JSON
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        # 2. Update the LLAVA model path in the workflow
        if workflow.get("11"):
            workflow["11"]["inputs"]["clip_name2"] = "llava_llama3_fp8_scaled.safetensors"

        # 3. Blank node 79's "lora_name" so ComfyUI won't attempt to download it
        if workflow.get("79") and "lora_name" in workflow["79"]["inputs"]:
            workflow["79"]["inputs"]["lora_name"] = ""

        # 4. Save the updated workflow
        with open(api_json_file, "w") as f:
            json.dump(workflow, f, indent=2)

        # 5. Only handle the base model weights here
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "hunyuan_video_720_fp8_e4m3fn.safetensors",
                "hunyuan_video_vae_bf16.safetensors",
                "clip_l.safetensors",
            ],
        )

    def copy_lora_file(self, lora_url: str) -> str:
        """
        Download the user-provided LoRA file from either:
          1) A direct URL to a .safetensors file (http/https).
          2) A Hugging Face repo ID (e.g. "username/repo"), automatically finding
             the first .safetensors file and using its main branch URL.

        The file is placed into ComfyUI/models/loras/ and ensured to be a
        '.safetensors' file. Returns the final filename.
        """
        # Create/ensure our target folder exists
        lora_dir = os.path.join("ComfyUI", "models", "loras")
        os.makedirs(lora_dir, exist_ok=True)

        # If this looks like an http(s) link, handle direct download
        if re.match(r"^https?:\/\/", lora_url):
            # Attempt to derive a local filename from the URL
            filename = os.path.basename(lora_url)
            if not filename.lower().endswith(".safetensors"):
                filename += ".safetensors"

            dst_path = os.path.join(lora_dir, filename)

            # Download and write to the local destination
            resp = requests.get(lora_url)
            resp.raise_for_status()
            with open(dst_path, "wb") as f:
                f.write(resp.content)

            return filename

        else:
            # Otherwise, treat lora_url as a Hugging Face repo ID
            repo_id = lora_url.strip()
            if "/" not in repo_id:
                raise ValueError(
                    f"Invalid Hugging Face repo ID '{repo_id}', format should be 'user/repo'."
                )

            api = HfApi()
            try:
                files = api.list_repo_files(repo_id)
            except Exception as e:
                raise ValueError(
                    f"Failed to access Hugging Face repo '{repo_id}': {e}"
                ) from e

            # Find the first available .safetensors file
            safetensors_files = [f for f in files if f.endswith(".safetensors")]
            if not safetensors_files:
                raise ValueError(
                    f"No .safetensors files found in Hugging Face repo: {repo_id}"
                )

            # Take the first .safetensors file
            hf_filename = safetensors_files[0]
            # Build the direct download URL
            hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{hf_filename}"

            # Use the same logic as above to download
            filename = os.path.basename(hf_filename)
            dst_path = os.path.join(lora_dir, filename)

            resp = requests.get(hf_url)
            resp.raise_for_status()
            with open(dst_path, "wb") as f:
                f.write(resp.content)

            return filename

    def handle_replicate_weights(self, replicate_weights: Path) -> str:
        """
        Extract ONLY lora_comfyui.safetensors from the user-provided tar file
        and move it to ComfyUI/models/loras/.
        Return the final filename ("lora_comfyui.safetensors").
        """
        lora_dir = os.path.join("ComfyUI", "models", "loras")
        os.makedirs(lora_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(str(replicate_weights), "r:*") as tar:
                tar.extractall(path=temp_dir)

            # We specifically want the ComfyUI version
            comfy_lora_path = os.path.join(temp_dir, "lora_comfyui.safetensors")
            if not os.path.exists(comfy_lora_path):
                raise FileNotFoundError(
                    "No 'lora_comfyui.safetensors' found in the provided tar."
                )

            filename = "lora_comfyui.safetensors"
            dst_path = os.path.join(lora_dir, filename)
            shutil.copy2(comfy_lora_path, dst_path)

        return filename

    def update_workflow(
        self,
        workflow: dict[str, Any],
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        flow_shift: int,
        seed: int,
        denoise_strength: float,
        num_frames: int,
        lora_name: str,
        lora_strength: float,
        frame_rate: int,
        crf: int,
    ):
        """
        Update the t2v_lora.json workflow with user-selected parameters.
        """
        # Node 45/81: EmptyHunyuanLatentVideo
        for node_id in ["45", "81"]:
            if node_id in workflow:
                workflow[node_id]["inputs"]["width"] = width
                workflow[node_id]["inputs"]["height"] = height
                workflow[node_id]["inputs"]["length"] = num_frames

        # Node 17: BasicScheduler
        workflow["17"]["inputs"]["steps"] = steps
        workflow["17"]["inputs"]["denoise"] = denoise_strength

        # Node 26: FluxGuidance
        workflow["26"]["inputs"]["guidance"] = guidance_scale

        # Node 67: ModelSamplingSD3
        workflow["67"]["inputs"]["shift"] = flow_shift

        # Node 25: RandomNoise
        workflow["25"]["inputs"]["noise_seed"] = seed

        # Node 44: CLIPTextEncode
        workflow["44"]["inputs"]["text"] = prompt

        # Node 79: LoraLoaderModelOnly
        workflow["79"]["inputs"]["lora_name"] = lora_name
        workflow["79"]["inputs"]["strength_model"] = lora_strength

        # Node 79: VHS_VideoCombine
        workflow["79"]["inputs"]["frame_rate"] = frame_rate
        workflow["79"]["inputs"]["crf"] = crf
        workflow["79"]["inputs"]["save_output"] = True

    def predict(
        self,
        prompt: str = Input(
            default="A modern lounge in lush greenery.",
            description="The text prompt describing your video scene.",
        ),
        lora_url: str = Input(
            default="",
            description="A URL pointing to your LoRA .safetensors file or a Hugging Face repo (e.g. 'user/repo' - will use first .safetensors found).",
        ),
        replicate_weights: Path = Input(
            default=None,
            description="A tar file containing LoRA weights from replicate. (Optional)",
        ),
        lora_strength: float = Input(
            default=1.0, description="Scale/strength for your LoRA."
        ),
        width: int = Input(
            default=848, ge=64, le=1536, description="Width for the generated video."
        ),
        height: int = Input(
            default=480, ge=64, le=1024, description="Height for the generated video."
        ),
        steps: int = Input(
            default=20, ge=1, le=150, description="Number of diffusion steps."
        ),
        guidance_scale: float = Input(
            default=6.0, description="Overall influence of text vs. model."
        ),
        flow_shift: int = Input(
            default=7, ge=0, le=20, description="Video continuity factor (flow)."
        ),
        denoise_strength: float = Input(
            default=1.0, description="Controls how strongly noise is applied each step."
        ),
        num_frames: int = Input(
            default=73,
            ge=1,
            le=300,
            description="How many frames (duration) in the resulting video.",
        ),
        frame_rate: int = Input(
            default=24, ge=1, le=60, description="Video frame rate."
        ),
        crf: int = Input(
            default=19,
            ge=0,
            le=51,
            description="CRF (quality) for h264 video encoding, lower=better.",
        ),
        seed: int = seed_helper.predict_seed(),
    ) -> Path:
        """
        Create a video using HunyuanVideo with either:
         - replicate_weights tar (preferred if provided)
         - a direct lora_url (HTTP link or Hugging Face repo ID)
        """
        # Convert user seed to a valid integer
        seed = seed_helper.generate(seed)

        # 1. Clean up previous runs
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # 2. Decide how to obtain our LoRA file name
        if replicate_weights is not None:
            # Use replicate tar (prefer the comfyui version)
            lora_name = self.handle_replicate_weights(replicate_weights)
        else:
            # Use the remote url or huggingface repo
            if not lora_url:
                raise ValueError(
                    "No LoRA provided. Provide either replicate_weights tar or a lora_url."
                )
            lora_name = self.copy_lora_file(lora_url)

        # 3. Load the main workflow JSON
        with open(api_json_file, "r") as f:
            workflow = json.loads(f.read())

        # 3a. Zero out node 79 lora so handle_weights won't see it
        workflow["79"]["inputs"]["lora_name"] = ""

        # 4. Fill in user parameters
        self.update_workflow(
            workflow=workflow,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            seed=seed,
            denoise_strength=denoise_strength,
            num_frames=num_frames,
            lora_name="",  # intentionally blank
            lora_strength=0.0,  # skip real strength
            frame_rate=frame_rate,
            crf=crf,
        )

        # 5. Load the workflow -> handle_weights sees lora="", won't attempt a download
        wf = self.comfyUI.load_workflow(workflow)

        # 5a. Now set the real LoRA file with path
        wf["79"]["inputs"]["lora_name"] = lora_name
        wf["79"]["inputs"]["strength_model"] = lora_strength

        # 6. Run the workflow
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # 7. Retrieve final output
        output_files = self.comfyUI.get_files(OUTPUT_DIR)
        if not output_files:
            raise RuntimeError("No output video was generated.")
        return output_files[0]