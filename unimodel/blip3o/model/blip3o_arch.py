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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import  build_gen_vision_tower, build_dit
from .multimodal_projector.builder import  build_down_projector



class blip3oMetaModel:

    def __init__(self, config):
        super(blip3oMetaModel, self).__init__(config)


        if hasattr(config, "gen_vision_tower"):
            self.gen_vision_tower = build_gen_vision_tower(config, from_config=True) # why use this will cause the training error
            self.down_projector = build_down_projector(config)

            self.t2i_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
            self.i2i_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))

            print(f" t2i query size {self.t2i_queries.shape}")
            print(f" i2i query size {self.i2i_queries.shape}")


            self.dit, self.noise_scheduler = build_dit(config, from_config=True)




    def get_gen_vision_tower(self):
        gen_vision_tower = getattr(self, 'gen_vision_tower', None)
        if type(gen_vision_tower) is list:
            gen_vision_tower = gen_vision_tower[0]
        return gen_vision_tower
    
    



    def initialize_vision_modules(self, model_args, fsdp=None):
        
        gen_vision_tower = model_args.gen_vision_tower


        self.config.gen_vision_tower = gen_vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")



        if getattr(self, 'dit', None) is None:
            print("random initiation the DiT !!!")
            self.dit, self.noise_scheduler = build_dit(model_args)
            print("self.dit.dtype", self.dit.dtype)
            # assert 0
        else:
            print("DiT load from checkpoint!!!")
            for p in self.dit.parameters():
                p.requires_grad = True
            print("self.dit.dtype", self.dit.dtype)
    

        if self.get_gen_vision_tower() is None:
            gen_vision_tower = build_gen_vision_tower(model_args, from_config=False) # why should we use this?
        else:
            gen_vision_tower = self.get_gen_vision_tower()


        if fsdp is not None and len(fsdp) > 0:
            self.gen_vision_tower = [gen_vision_tower]
        else:
            self.gen_vision_tower = gen_vision_tower
                

        self.config.down_projector_type = getattr(model_args, 'down_projector_type', 'linear')
        self.config.sana_embeds_hidden_size = getattr(model_args, 'sana_embeds_hidden_size', 2304)


        self.config.gen_hidden_size = gen_vision_tower.hidden_size
        self.config.n_query = model_args.n_query


        if getattr(self, 'down_projector', None) is None:
            print("random initiation the down_projector !!!")
            self.down_projector = build_down_projector(self.config)
        # else:
            # In case it is frozen by LoRA
        for p in self.down_projector.parameters():
            p.requires_grad = True


        if getattr(self, 't2i_queries', None) is None:
            print("random initiation the latent_queries !!!")
            self.t2i_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))
        else:
            print("latent_queries load from checkpoint!!!")
            self.t2i_queries.requires_grad = True
            
        if getattr(self, 'i2i_queries', None) is None:
            print("random initiation the latent_queries !!!")
            self.i2i_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))
        else:
            print("latent_queries load from checkpoint!!!")
            self.i2i_queries.requires_grad = True
        

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class blip3oMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_gen_vision_tower(self):
        return self.get_model().get_gen_vision_tower()

    def get_n_query(self):
        return self.get_model().config.n_query

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):

        sigma = timesteps.to(device, dtype) / 1000
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1
        return latents * mask
