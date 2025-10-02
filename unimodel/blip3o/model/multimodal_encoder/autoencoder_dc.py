# Copyright 2025 Fu-Yun Wang
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

import torch
import torch.nn as nn
from diffusers import AutoencoderDC
from torchvision import transforms
from PIL import Image

class DiffusersImageProcessor:
    def __init__(self, image_size=1024):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size), 
            transforms.ToTensor(),  
            transforms.Lambda(lambda x: x * 2 - 1)  
        ])
        
    
    def preprocess(self, images, return_tensors="pt"):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)
        
        processed_images = [self.transform(image) for image in images]
        
        if return_tensors == "pt":
            process_images =  torch.stack(processed_images)
            
        return {"pixel_values": process_images}

    def __call__(self, item):
        return self.transform(item)

    @property
    def size(self):
        return {"height": self.image_size, "width": self.image_size}

class DiffusersAutoEncoderDCTower(nn.Module):
    def __init__(self, model_id="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", from_config=False):
        super().__init__()
        
        self.from_config = from_config
        self.model_id = model_id
        self.image_processor = DiffusersImageProcessor(image_size=1024)
        
        self.load_model()
    
    def load_model(self, device_map=None):
        if self.from_config:
            config = AutoencoderDC.load_config(self.model_id, subfolder="vae")
            self.autoencoder = AutoencoderDC.from_config(config, torch_dtype=torch.bfloat16, device_map=device_map)
        else:
            self.autoencoder = AutoencoderDC.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.bfloat16, variant="bf16", device_map=device_map)
        self.autoencoder.requires_grad_(False)
        print(f"Loaded Diffusers AutoEncoderDC: {self.model_id}")
    
    def forward(self, images):
        latents = self.autoencoder.encode(images.to(device=self.device, dtype=self.dtype)).latent * self.autoencoder.config.scaling_factor
        return latents

    @property
    def dtype(self):
        return self.autoencoder.dtype
    
    @property
    def device(self):
        return self.autoencoder.device
    
    @property
    def hidden_size(self):
        return self.autoencoder.config.latent_channels
    
    @property
    def image_size(self):
        return self.image_processor.size