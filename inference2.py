import os
import torch
import numpy as np
import albumentations
from omegaconf import OmegaConf
from PIL import Image
from main import instantiate_from_config


class TextGuidedImageEditor:
    def __init__(self, vqgan_config, transformer_config, resume, device="cuda"):
        """
        Initialize the text-guided image editor model.

        Args:
            vqgan_config: Path to VQGAN config file
            transformer_config: Path to transformer config file
            resume: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Load configs
        self.decoder_config = OmegaConf.load(vqgan_config)
        self.config = OmegaConf.load(transformer_config)
        self.config.model.params.decoder_config = self.decoder_config.model
        self.config.model.params.ckpt_path = resume

        # Initialize model
        self.model = instantiate_from_config(self.config.model).to(device)
        self.model.eval()
        self.device = device

        # Get image size from config
        self.height = self.config.data.params.validation.params.size
        self.width = self.height * 4  # Using hw_ratio=4 from dataset.py

    def preprocess_image(self, image_np):
        """Preprocess input numpy image to model input format"""
        # Store original size
        ori_size = (image_np.shape[1], image_np.shape[0])  # (W, H)

        # Resize image
        resizer = albumentations.Resize(height=self.height, width=self.width)
        resized_img = resizer(image=image_np)["image"]

        # Get resized dimensions
        img_size = (resized_img.shape[1], resized_img.shape[0])  # (W, H)

        # Normalize to [-1, 1]
        normalized_img = (resized_img / 127.5 - 1.0).astype(np.float32)

        return normalized_img, ori_size, img_size

    def infer(self, image_np, text):
        """
        Process an input image according to the given text prompt

        Args:
            image_np: RGB numpy image array with shape [H, W, 3] (uint8, 0-255)
            text: String describing the desired edit

        Returns:
            edited_img: RGB numpy image array with shape [H, W, 3] (uint8, 0-255)
        """
        with torch.no_grad():
            # Preprocess the image
            preprocessed_img, ori_size, img_size = self.preprocess_image(image_np)

            # Convert to tensor and prepare for model
            img1 = torch.from_numpy(preprocessed_img).permute(2, 0, 1)[None].to(self.device)
            img1 = img1.to(memory_format=torch.contiguous_format)
            if img1.dtype == torch.double:
                img1 = img1.float()

            # Extract image features
            img1_quant_latent_ = self.model.conv(img1).flatten(2).permute(0, 2, 1)

            # Encode text
            rec2_indices, _ = self.model.str_converter.encode([text])
            rec2_indices = rec2_indices.to(img1_quant_latent_.device)

            # Prepare embeddings
            rec2_embd = self.model.str_embd(rec2_indices)
            rec1_mask = self.model.masked_rec_embd.weight[0][None, None, :].expand(
                rec2_indices.shape[0], rec2_indices.shape[1], -1
            )
            img2_mask = self.model.masked_img_imbd.weight[0][None, None, :].expand(
                img1_quant_latent_.shape[0], img1_quant_latent_.shape[1], -1
            )

            # Process through transformer
            inputs = torch.cat([rec2_embd, img1_quant_latent_, rec1_mask, img2_mask], dim=1)
            embeddings, logits = self.model.transformer(inputs)

            # Generate output image
            img2_rec_quant_latent = self.model.conv_o(
                embeddings[:, 320:576, :].permute(0, 2, 1).contiguous().view(embeddings.shape[0], -1, 8, 32)
            )
            img2_rec = self.model.decoder.decode(img2_rec_quant_latent)

            # Post-process the output image (similar to crop_and_resize function)
            edited_img = img2_rec.detach().cpu()
            edited_img = torch.clamp(edited_img, -1.0, 1.0)
            edited_img = (edited_img + 1.0) / 2.0
            edited_img = edited_img.permute(0, 2, 3, 1).numpy()[0]
            edited_img = (edited_img * 255).astype(np.uint8)

            return edited_img
