# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import os
import time
import subprocess
from cog import BasePredictor, Input, Path, BaseModel
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import save_videos_grid
from src.audio_models.model import Audio2MeshModel
from src.utils.audio_util import prepare_audio_feature
from src.utils.mp_utils import LMKExtractor
from src.utils.draw_util import FaceMeshVisualizer
from src.utils.pose_util import project_points

# Weights are saved and loaded from replicate.delivery for faster booting
MODEL_URL = "https://weights.replicate.delivery/default/AniPortrait.tar"  # prepare the pre-trained models following https://github.com/Zejun-Yang/AniPortrait?tab=readme-ov-file#download-weights
MODEL_CACHE = "pretrained_model"


class ModelOutput(BaseModel):
    video: Path
    pose: Path


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        audio2video = "configs/prompts/animation_audio.yaml"
        self.config = OmegaConf.load(audio2video)
        weight_dtype = torch.float16

        self.audio_infer_config = OmegaConf.load(self.config.audio_inference_config)

        self.a2m_model = Audio2MeshModel(self.audio_infer_config["a2m_model"])
        self.a2m_model.load_state_dict(
            torch.load(self.audio_infer_config["pretrained_model"]["a2m_ckpt"]),
            strict=False,
        )
        self.a2m_model.cuda().eval()

        vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_vae_path,
        ).to("cuda", dtype=weight_dtype)

        reference_unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device="cuda")

        inference_config_path = self.config.inference_config
        infer_config = OmegaConf.load(inference_config_path)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            self.config.pretrained_base_model_path,
            self.config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")

        pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(
            device="cuda", dtype=weight_dtype
        )  # not use cross attention

        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            self.config.image_encoder_path
        ).to(dtype=weight_dtype, device="cuda")

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        # load pretrained weights
        denoising_unet.load_state_dict(
            torch.load(self.config.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(self.config.reference_unet_path, map_location="cpu"),
        )
        pose_guider.load_state_dict(
            torch.load(self.config.pose_guider_path, map_location="cpu"),
        )

        self.pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.pipe = self.pipe.to("cuda", dtype=weight_dtype)

        self.lmk_extractor = LMKExtractor()
        self.vis = FaceMeshVisualizer(forehead_edge=False)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        audio: Path = Input(description="Input audio"),
        width: int = Input(description="Width of output video", default=512),
        height: int = Input(description="Height of output video", default=512),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", default=3.5
        ),
        steps: int = Input(description="Inference steps", default=25),
        fps: int = Input(
            description="Frame per second in the output video", default=30
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.manual_seed(seed)

        ref_image_pil = Image.open(str(image)).convert("RGB")
        ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
        ref_image_np = cv2.resize(ref_image_np, (height, width))

        face_result = self.lmk_extractor(ref_image_np)
        assert face_result is not None, "No face detected."
        lmks = face_result["lmks"].astype(np.float32)
        ref_pose = self.vis.draw_landmarks(
            (ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True
        )

        sample = prepare_audio_feature(
            str(audio),
            wav2vec_model_path=self.audio_infer_config["a2m_model"]["model_path"],
        )
        sample["audio_feature"] = (
            torch.from_numpy(sample["audio_feature"]).float().cuda()
        )
        sample["audio_feature"] = sample["audio_feature"].unsqueeze(0)

        # inference
        pred = self.a2m_model.infer(sample["audio_feature"], sample["seq_len"])
        pred = pred.squeeze().detach().cpu().numpy()
        pred = pred.reshape(pred.shape[0], -1, 3)
        pred = pred + face_result["lmks3d"]

        pose_seq = np.load(self.config["pose_temp"])
        mirrored_pose_seq = np.concatenate((pose_seq, pose_seq[-2:0:-1]), axis=0)
        cycled_pose_seq = np.tile(
            mirrored_pose_seq, (sample["seq_len"] // len(mirrored_pose_seq) + 1, 1)
        )[: sample["seq_len"]]

        # project 3D mesh to 2D landmark
        projected_vertices = project_points(
            pred, face_result["trans_mat"], cycled_pose_seq, [height, width]
        )

        pose_images = []
        for i, verts in enumerate(projected_vertices):
            lmk_img = self.vis.draw_landmarks((width, height), verts, normed=False)
            pose_images.append(lmk_img)

        pose_list = []
        pose_tensor_list = []
        print(f"pose video has {len(pose_images)} frames, with {fps} fps")
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        args_L = len(pose_images)
        for pose_image_np in pose_images[:args_L]:
            pose_image_pil = Image.fromarray(
                cv2.cvtColor(pose_image_np, cv2.COLOR_BGR2RGB)
            )
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_image_np = cv2.resize(pose_image_np, (width, height))
            pose_list.append(pose_image_np)

        pose_list = np.array(pose_list)

        video_length = len(pose_tensor_list)

        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video_length
        )

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)

        video = self.pipe(
            ref_image_pil,
            pose_list,
            ref_pose,
            width,
            height,
            video_length,
            steps,
            guidance_scale,
            generator=generator,
        ).videos

        output_path = "/tmp/out.mp4"
        pose_path = "/tmp/pose.mp4"

        save_videos_grid(
            video,
            output_path,
            n_rows=1,
            fps=fps,
        )
        save_videos_grid(
            pose_tensor,
            pose_path,
            n_rows=1,
            fps=fps,
        )
        return ModelOutput(video=Path(output_path), pose=Path(pose_path))
