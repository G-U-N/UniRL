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

import os

from diffusers.models.controlnets.controlnet import zero_module
from .autoencoder_dc import DiffusersAutoEncoderDCTower
from .sana_transformer import SanaTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import torch




def build_gen_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'gen_vision_tower')

    return DiffusersAutoEncoderDCTower(vision_tower, **kwargs)

def init_meta_params(model):
    for name, param in model.named_parameters():
        if param.is_meta:
            print(f"Initializing {name} (was meta)")
            new_tensor = torch.empty(param.shape, dtype=param.dtype, device='cpu')
            torch.nn.init.normal_(new_tensor, mean=0.0, std=0.02)  
            module = model
            for attr in name.split(".")[:-1]:
                module = getattr(module, attr)
            setattr(module, name.split(".")[-1], torch.nn.Parameter(new_tensor))


def build_dit(model_args, from_config=False):
    
    if from_config:
        config = SanaTransformer2DModel.load_config("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",subfolder="transformer", low_cpu_mem_usage=True)
        dit = SanaTransformer2DModel.from_config(config)
            
    else:
        dit = SanaTransformer2DModel.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",subfolder="transformer", ignore_mismatched_sizes=True)
        init_meta_params(dit)
        dit.ref_linear = zero_module(dit.ref_linear)
        dit.ref_patch_embed.load_state_dict(dit.patch_embed.state_dict())

    
    model_args.sana_embeds_hidden_size = 2304
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    return dit, noise_scheduler



