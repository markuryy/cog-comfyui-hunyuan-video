import os
import json
import mimetypes
import shutil
import re
import requests
from typing import Any

from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import seed as seed_helper

# Directories for inputs/outputs
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/quicktime", ".mov")

api_json_file = "t2v-lora.json"

# Force offline mode
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        """
        Start ComfyUI, ensuring it doesn't attempt to download our local LoRA file
        before running. We do this by blanking out node 41's "lora" field so the
        weight downloader never sees "adapter_model_epoch172.safetensors".
        """
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # 1. Load the main workflow JSON
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        # 2. Blank node 41's "lora" so ComfyUI won't attempt to download it
        if workflow.get("41") and "lora" in workflow["41"]["inputs"]:
            workflow["41"]["inputs"]["lora"] = ""

        # 3. Only handle the base model weights here
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "hunyuan_video_720_fp8_e4m3fn.safetensors",
                "hunyuan_video_vae_bf16.safetensors",
                "clip-vit-large-patch14",
                "llava-llama-3-8b-text-encoder-tokenizer",
            ],
        )

    def copy_lora_file(self, lora_url: str) -> str:
        """
        Download the user-provided LoRA file from a URL, place it into ComfyUI/models/loras/,
        and ensure it has a .safetensors extension.
        Returns the final filename, e.g. "adapter_model_epoch172.safetensors".
        """

        # Quick check to ensure it's a URL (very basic)
        if not re.match(r"^https?:\/\/", lora_url):
            raise ValueError("Invalid LoRA URL. Please provide a valid https:// or http:// link.")

        lora_dir = os.path.join("ComfyUI", "models", "loras")
        os.makedirs(lora_dir, exist_ok=True)

        # Attempt to derive a filename from the URL
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
        force_offload: bool,
        denoise_strength: float,
        num_frames: int,
        lora_name: str,
        lora_strength: float,
        frame_rate: int,
        crf: int,
    ):
        """
        Update the t2v-lora.json workflow with user-selected parameters.
        """
        # Node 3: HyVideoSampler
        workflow["3"]["inputs"]["width"] = width
        workflow["3"]["inputs"]["height"] = height
        workflow["3"]["inputs"]["steps"] = steps
        workflow["3"]["inputs"]["embedded_guidance_scale"] = guidance_scale
        workflow["3"]["inputs"]["flow_shift"] = flow_shift
        workflow["3"]["inputs"]["seed"] = seed
        workflow["3"]["inputs"]["force_offload"] = 1 if force_offload else 0
        workflow["3"]["inputs"]["denoise_strength"] = denoise_strength
        workflow["3"]["inputs"]["num_frames"] = num_frames

        # Node 30: HyVideoTextEncode
        workflow["30"]["inputs"]["prompt"] = prompt
        workflow["30"]["inputs"]["force_offload"] = (
            "bad quality video" if force_offload else " "
        )

        # Node 41: HyVideoLoraSelect
        workflow["41"]["inputs"]["lora"] = lora_name
        workflow["41"]["inputs"]["strength"] = lora_strength

        # Node 34: VHS_VideoCombine
        workflow["34"]["inputs"]["frame_rate"] = frame_rate
        workflow["34"]["inputs"]["crf"] = crf
        workflow["34"]["inputs"]["save_output"] = True

    def predict(
        self,
        prompt: str = Input(
            default="A modern lounge in lush greenery.",
            description="The text prompt describing your video scene.",
        ),
        lora_url: str = Input(
            description="A URL pointing to your LoRA .safetensors file for fine-tuning."
        ),
        lora_strength: float = Input(
            default=1.0, description="Scale/strength for your LoRA."
        ),
        width: int = Input(
            default=640, ge=64, le=1536, description="Width for the generated video."
        ),
        height: int = Input(
            default=360, ge=64, le=1024, description="Height for the generated video."
        ),
        steps: int = Input(
            default=50, ge=1, le=150, description="Number of diffusion steps."
        ),
        guidance_scale: float = Input(
            default=6.0, description="Overall influence of text vs. model."
        ),
        flow_shift: int = Input(
            default=9, ge=0, le=20, description="Video continuity factor (flow)."
        ),
        force_offload: bool = Input(
            default=True, description="Whether to force model layers offloaded to CPU."
        ),
        denoise_strength: float = Input(
            default=1.0, description="Controls how strongly noise is applied each step."
        ),
        num_frames: int = Input(
            default=85,
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
        Create a video using HunyuanVideo with a remote LoRA (downloaded at runtime).
        """
        # Convert user seed to a valid integer
        seed = seed_helper.generate(seed)

        # 1. Clean up previous runs
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # 2. Download the LoRA file to ComfyUI/models/loras/
        lora_name = self.copy_lora_file(lora_url)

        # 3. Load the main workflow JSON
        with open(api_json_file, "r") as f:
            workflow = json.loads(f.read())

        # 3a. Zero out node 41 lora so handle_weights won't see it
        workflow["41"]["inputs"]["lora"] = ""

        # 4. Fill in user parameters, skipping the real LoRA name/strength for now
        self.update_workflow(
            workflow=workflow,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            seed=seed,
            force_offload=force_offload,
            denoise_strength=denoise_strength,
            num_frames=num_frames,
            lora_name="",  # intentionally blank
            lora_strength=0.0,  # skip real strength
            frame_rate=frame_rate,
            crf=crf,
        )

        # 5. Load the workflow -> handle_weights sees lora="", won't attempt a download
        wf = self.comfyUI.load_workflow(workflow)

        # 5a. Now that handle_weights is done, set the real local LoRA
        wf["41"]["inputs"]["lora"] = lora_name
        wf["41"]["inputs"]["strength"] = lora_strength

        # 6. Run the workflow
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # 7. Retrieve final output
        output_files = self.comfyUI.get_files(OUTPUT_DIR)
        if not output_files:
            raise RuntimeError("No output video was generated.")
        return output_files[0]
