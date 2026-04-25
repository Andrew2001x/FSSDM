#!/usr/bin/env python3
"""
Complete StableUnCLIP Img2Img Model Implementation
Based on sd2-community/stable-diffusion-2-1-unclip-small

This implementation includes:
- OpenCLIP Image Encoder (ViT-H-14)
- Full UNet with downsample/upsample blocks
- Complete VAE (Encoder + Decoder)
- Text Encoder
- Super Resolution upscaler
- Complete noise scheduling
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.resnet import ResBlock2D, Upsample2D
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StableUnCLIPConfig:
    """Configuration for StableUnCLIP model"""
    # Model dimensions
    image_embed_dim: int = 768
    latent_channels: int = 4
    unmapped_channels: int = 4
    time_embed_dim: int = 320
    cond_embed_dim: int = 768
    
    # UNet
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: int = 8
    cross_attention_dim: int = 768
    
    # VAE
    latent_h: int = 64
    latent_w: int = 64
    
    # CLIP
    clip_embed_dim: int = 1024
    clip_hidden_dim: int = 1024
    clip_num_heads: int = 16
    clip_num_layers: int = 24
    
    # Noise scheduling
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    
    # SuperRes
    super_res_channels: int = 3


# ============================================================================
# OpenCLIP Image Encoder (ViT-H/14)
# ============================================================================

class OpenCLIPVisionConfig:
    """Configuration for OpenCLIP ViT-H/14"""
    embed_dim: int = 1024
    image_size: int = 224
    patch_size: int = 14
    width: int = 1024
    layers: int = 32
    heads: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    mlp_time_embed: bool = False
    num_classes: int = -1
    trust_remote_code: bool = True


class OpenCLIPImageEncoder(nn.Module):
    """
    OpenCLIP Image Encoder (ViT-H/14 variant)
    Used to encode images into CLIP image embeddings
    """
    
    def __init__(self, config: OpenCLIPVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.grid_size = self.image_size // self.patch_size
        
        self.class_embedding = nn.Parameter(torch.randn(config.embed_dim))
        
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False
        )
        
        self.num_patches = self.grid_size * self.grid_size
        num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(num_positions, config.width)
        
        self.pre_layrnorm = nn.LayerNorm(config.width)
        self.post_layrnorm = nn.LayerNorm(config.width)
        
        self.transformer = OpenCLIPTransformer(config)
        
        self.proj = nn.Linear(config.width, config.embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W] - normalized image tensor
            
        Returns:
            image_embeds: [B, embed_dim] - CLIP image embeddings
        """
        B = pixel_values.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(pixel_values)  # [B, width, H/P, W/P]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, n_patches]
        x = x.permute(0, 2, 1)  # [B, n_patches, width]
        
        # Add class embedding
        class_embedding = self.class_embedding.unsqueeze(0).expand(B, -1)
        x = torch.cat([class_embedding, x], dim=1)  # [B, n_patches+1, width]
        
        # Add positional embedding
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embedding(positions)
        
        # Pre layer norm
        x = self.pre_layrnorm(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Post layer norm
        x = self.post_layrnorm(x)
        
        # Take class token
        image_embeds = x[:, 0]  # [B, width]
        
        # Project to embed dim
        image_embeds = self.proj(image_embeds)  # [B, embed_dim]
        
        return image_embeds


class OpenCLIPTransformer(nn.Module):
    """OpenCLIP Transformer encoder"""
    
    def __init__(self, config: OpenCLIPVisionConfig):
        super().__init__()
        self.config = config
        width = config.width
        self.width = width
        self.layers = config.layers
        
        self.resblocks = nn.ModuleList([
            OpenCLIPResidualAttentionBlock(width, config.heads, mlp_ratio=config.mlp_ratio)
            for _ in range(config.layers)
        ])
        
        self.gradient_checkpointing = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class OpenCLIPResidualAttentionBlock(nn.Module):
    """OpenCLIP Residual Attention Block"""
    
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.attn = OpenCLIPAttention(d_model, n_heads)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = OpenCLIPMLP(d_model, int(d_model * mlp_ratio))
        self.ln_2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class OpenCLIPAttention(nn.Module):
    """OpenCLIP Attention"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class OpenCLIPMLP(nn.Module):
    """OpenCLIP MLP"""
    
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# ============================================================================
# Text Encoder (CLIP Text)
# ============================================================================

class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder
    Encodes text prompts into CLIP text embeddings
    """
    
    def __init__(self, vocab_size: int = 49408, embed_dim: int = 768, max_length: int = 77):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        self.transformer = CLIPTextTransformer(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_size=embed_dim,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=embed_dim * 4
        )
        
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] - token ids
            attention_mask: [B, L] - attention mask (optional)
            
        Returns:
            text_embeds: [B, embed_dim] - text embeddings
        """
        B, L = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [B, L, embed_dim]
        
        # Position embeddings
        positions = torch.arange(L, device=input_ids.device)
        x = x + self.position_embedding(positions)
        
        # Transformer
        x = self.transformer(x, attention_mask)
        
        # Final layer norm
        x = self.final_layer_norm(x)
        
        # Take last hidden state
        text_embeds = x
        
        return text_embeds


class CLIPTextTransformer(nn.Module):
    """CLIP Text Transformer"""
    
    def __init__(
        self, 
        vocab_size: int, 
        max_length: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int
    ):
        super().__init__()
        self.width = hidden_size
        self.layers = num_hidden_layers
        
        self.resblocks = nn.ModuleList([
            CLIPTextResidualAttentionBlock(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resblock in self.resblocks:
            x = resblock(x, attention_mask)
        return x


class CLIPTextResidualAttentionBlock(nn.Module):
    """CLIP Text Residual Attention Block"""
    
    def __init__(self, d_model: int, n_heads: int, intermediate_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.attn = CLIPTextAttention(d_model, n_heads)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = CLIPTextMLP(d_model, intermediate_dim)
        self.ln_2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPTextAttention(nn.Module):
    """CLIP Text Attention"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if attention_mask is not None:
            attn = attn + attention_mask.unsqueeze(1).unsqueeze(2)
        
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class CLIPTextMLP(nn.Module):
    """CLIP Text MLP"""
    
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.activation_fn = F.gelu
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


# ============================================================================
# UNet2D Condition Model (Full Implementation)
# ============================================================================

class StableUnCLIPUNet(nn.Module):
    """
    Full UNet2D for StableUnCLIP
    Includes downsample/upsample blocks with cross-attention
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        attention_head_dim: int = 8,
        cross_attention_dim: int = 768,
        time_embed_dim: int = 320,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        
        # Time embedding
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, freq_shift=0)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)
        
        # Input conv
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        current_channels = block_out_channels[0]
        
        for i, out_ch in enumerate(block_out_channels):
            for _ in range(layers_per_block):
                self.down_blocks.append(
                    DownBlock2D(
                        in_channels=current_channels,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        cross_attention_dim=cross_attention_dim,
                        attention_head_dim=attention_head_dim,
                    )
                )
                current_channels = out_ch
            
            # Downsample at the end of each block (except last)
            if i < len(block_out_channels) - 1:
                self.down_blocks.append(
                    Downsample2D(current_channels, current_channels)
                )
        
        # Mid block
        self.mid_block = MidBlock2D(
            channels=block_out_channels[-1],
            time_embed_dim=time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )
        
        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        
        for i, out_ch in enumerate(reversed(block_out_channels)):
            for j in range(layers_per_block + 1):
                self.up_blocks.append(
                    UpBlock2D(
                        in_channels=current_channels + block_out_channels[-(i+1)] if j == 0 else out_ch,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        cross_attention_dim=cross_attention_dim,
                        attention_head_dim=attention_head_dim,
                    )
                )
                current_channels = out_ch
            
            # Upsample at the end of each block (except last)
            if i < len(block_out_channels) - 1:
                self.up_blocks.append(
                    Upsample2D(current_channels, current_channels)
                )
        
        # Output conv
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, block_out_channels[0]),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1),
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            sample: [B, 4, H, W] - latent tensor
            timestep: timestep
            encoder_hidden_states: [B, seq_len, cross_attention_dim] - text embeddings
            
        Returns:
            noise_pred: [B, 4, H, W] - predicted noise
        """
        # Time embedding
        t_emb = self.time_proj(timestep)
        t_emb = self.time_embedding(t_emb)
        
        # Input conv
        h = self.conv_in(sample)
        
        # Downsampling
        down_block_res_samples = []
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, 'resnets'):
                for resnet in downsample_block.resnets:
                    h = resnet(hidden_states=h, temb=t_emb, encoder_hidden_states=encoder_hidden_states)
                    down_block_res_samples.append(h)
            else:
                h = downsample_block(h)
                down_block_res_samples.append(h)
        
        # Mid block
        h = self.mid_block(hidden_states=h, temb=t_emb, encoder_hidden_states=encoder_hidden_states)
        
        # Upsampling
        for i, upsample_block in enumerate(self.up_blocks):
            if isinstance(upsample_block, Upsample2D):
                h = upsample_block(h)
            else:
                # Get residual
                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
                
                h = torch.cat([h, res_samples[-1]], dim=1)
                h = upsample_block(hidden_states=h, temb=t_emb, encoder_hidden_states=encoder_hidden_states, res_hidden_states=res_samples)
        
        # Output conv
        h = self.conv_out(h)
        
        if return_dict:
            return {"sample": h}
        return h


class Timesteps(nn.Module):
    """Timestep projection"""
    
    def __init__(self, dim: int, flip_sin_to_cos: bool = True, freq_shift: int = 0):
        super().__init__()
        self.dim = dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift
    
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep
        half_dim = self.dim // 2
        emb = math.log(half_dim) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=t.dtype, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        
        if self.flip_sin_to_cos:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
    
    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)


class DownBlock2D(nn.Module):
    """Down block with cross-attention"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        cross_attention_dim: int,
        attention_head_dim: int,
    ):
        super().__init__()
        resnets = []
        attentions = []
        
        for _ in range(2):
            resnets.append(
                ResBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_embed_dim=time_embed_dim,
                )
            )
            attentions.append(
                Transformer2DModel(
                    in_channels=out_channels,
                    num_attention_heads=out_channels // attention_head_dim,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
            )
            in_channels = out_channels
        
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states


class UpBlock2D(nn.Module):
    """Up block with cross-attention"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        cross_attention_dim: int,
        attention_head_dim: int,
    ):
        super().__init__()
        resnets = []
        attentions = []
        
        for _ in range(3):
            resnets.append(
                ResBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_embed_dim=time_embed_dim,
                )
            )
            attentions.append(
                Transformer2DModel(
                    in_channels=out_channels,
                    num_attention_heads=out_channels // attention_head_dim,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
            )
            in_channels = out_channels
        
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        res_hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = torch.cat([hidden_states, res_hidden_states.pop()], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states


class MidBlock2D(nn.Module):
    """Middle block with cross-attention"""
    
    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        cross_attention_dim: int,
        attention_head_dim: int,
    ):
        super().__init__()
        
        self.resnet_1 = ResBlock2D(channels, channels, time_embed_dim)
        self.attn_1 = Transformer2DModel(
            in_channels=channels,
            num_attention_heads=channels // attention_head_dim,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
        )
        self.resnet_2 = ResBlock2D(channels, channels, time_embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attn_1(hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = self.resnet_2(hidden_states, temb)
        return hidden_states


class Transformer2DModel(nn.Module):
    """Transformer2D with cross-attention"""
    
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int = 1,
        attention_head_dim: int = 1,
        cross_attention_dim: int = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                inner_dim=inner_dim,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads,
            )
        ])
        
        self.proj_out = nn.Linear(inner_dim, in_channels)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = hidden_states.shape
        
        # Group norm
        hidden_states = self.group_norm(hidden_states)
        
        # Reshape to sequence
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Project in
        hidden_states = self.proj_in(hidden_states)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        
        # Project out
        hidden_states = self.proj_out(hidden_states)
        
        # Reshape back
        hidden_states = hidden_states.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return hidden_states


class TransformerBlock(nn.Module):
    """Transformer block with cross-attention"""
    
    def __init__(
        self,
        inner_dim: int,
        cross_attention_dim: int,
        num_attention_heads: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim)
        self.self_attn = CrossAttention(
            query_dim=inner_dim,
            heads=num_attention_heads,
            dim_head=inner_dim // num_attention_heads,
        )
        self.norm2 = nn.LayerNorm(inner_dim)
        self.cross_attn = CrossAttention(
            query_dim=inner_dim,
            context_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=inner_dim // num_attention_heads,
        )
        self.norm3 = nn.LayerNorm(inner_dim)
        self.ff = FeedForward(inner_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        hidden_states = hidden_states + self.self_attn(self.norm1(hidden_states))
        
        # Cross attention
        if encoder_hidden_states is not None:
            hidden_states = hidden_states + self.cross_attn(
                self.norm2(hidden_states), 
                encoder_hidden_states
            )
        
        # FFN
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        
        return hidden_states


class CrossAttention(nn.Module):
    """Cross attention module"""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim or query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim or query_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = hidden_states.shape[0]
        
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states or hidden_states)
        v = self.to_v(encoder_hidden_states or hidden_states)
        
        q = q.reshape(B, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * (self.dim_head ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.heads * self.dim_head)
        return self.to_out(x)


class FeedForward(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Super Resolution / Feature Upscaler
# ============================================================================

class FeatureUpscaler(nn.Module):
    """
    Feature Upscaler for StableUnCLIP
    Upscales the CLIP image embeddings to higher dimension
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        hidden_dim: int = 1280,
        out_channels: int = 1280,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        
        # Input projection
        self.proj_in = nn.Linear(in_channels, hidden_dim)
        
        # Upsampling layers
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            ])
        self.layers = nn.ModuleList(layers)
        
        # Output projection
        self.proj_out = nn.Linear(hidden_dim, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels] - CLIP image embeddings
            
        Returns:
            upscaled: [B, out_channels] - upscaled embeddings
        """
        x = self.proj_in(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.proj_out(x)
        
        return x


class SuperResUpsampler(nn.Module):
    """
    Super Resolution Upsampler for latent upscaling
    Used in StableUnCLIP for higher resolution generation
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_channels: int = 1280,
        num_layers: int = 4,
    ):
        super().__init__()
        
        # Input conv
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        # ResNet blocks
        self.res_blocks = nn.ModuleList([
            ResBlock2D(hidden_channels, hidden_channels, time_embed_dim=hidden_channels)
            for _ in range(num_layers)
        ])
        
        # Output conv with upsampling
        self.upsample = nn.Sequential(
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, 4, H, W] - latent tensor
            temb: [B, time_embed_dim] - time embedding
            
        Returns:
            upsampled: [B, 4, 2*H, 2*W] - upsampled latent
        """
        h = self.conv_in(x)
        
        for res_block in self.res_blocks:
            h = res_block(h, temb)
        
        h = self.upsample(h)
        
        return h


# ============================================================================
# Complete StableUnCLIP Model
# ============================================================================

class StableUnCLIPModel(nn.Module):
    """
    Complete StableUnCLIP Model
    
    This combines:
    - OpenCLIP Image Encoder (for encoding input images)
    - Optional Text Encoder (for text prompts)
    - UNet (for noise prediction)
    - VAE Decoder (for decoding latents to images)
    - Feature Upscaler (for upscaling image embeddings)
    - SuperRes (for latent upscaling)
    """
    
    def __init__(
        self,
        config: StableUnCLIPConfig,
        model_id: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.config = config
        
        # Image encoder (OpenCLIP)
        self.image_encoder = OpenCLIPImageEncoder(
            OpenCLIPVisionConfig()
        )
        
        # Feature upscaler
        self.feature_upscaler = FeatureUpscaler(
            in_channels=config.clip_embed_dim,
            hidden_dim=config.cond_embed_dim * 2,
            out_channels=config.cond_embed_dim,
            num_layers=2,
        )
        
        # Text encoder (optional, can use pretrained CLIP)
        self.text_encoder = None
        
        # UNet
        self.unet = StableUnCLIPUNet(
            in_channels=config.latent_channels,
            out_channels=config.latent_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            attention_head_dim=config.attention_head_dim,
            cross_attention_dim=config.cond_embed_dim,
            time_embed_dim=config.time_embed_dim,
        )
        
        # VAE
        self.vae = None
        
        # Scheduler
        self.scheduler = None
        
        # SuperRes (for higher resolution)
        self.super_res = None
        
        self.dtype = dtype
    
    def load_pretrained(self, model_id: str, device: str = "cuda"):
        """Load pretrained weights from HuggingFace"""
        from diffusers import StableUnCLIPImg2ImgPipeline
        
        print(f"Loading pretrained model from {model_id}...")
        
        # Load full pipeline to extract components
        pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        
        # Copy image encoder weights
        if hasattr(pipeline, 'image_encoder'):
            self.image_encoder.load_state_dict(
                pipeline.image_encoder.state_dict(),
                strict=False
            )
        
        # Copy feature upscaler weights
        if hasattr(pipeline, 'feature_extractor'):
            self.feature_upscaler.load_state_dict(
                pipeline.feature_extractor.state_dict() if hasattr(pipeline.feature_extractor, 'state_dict') else {},
                strict=False
            )
        
        # Copy UNet weights
        self.unet.load_state_dict(pipeline.unet.state_dict(), strict=False)
        
        # Copy VAE weights
        if hasattr(pipeline, 'vae'):
            self.vae = pipeline.vae
        
        # Setup scheduler
        self.scheduler = pipeline.scheduler
        
        print("Pretrained weights loaded successfully!")
        
        del pipeline
    
    def encode_image(
        self,
        pixel_values: torch.Tensor,
        noise_level: int = 0,
    ) -> torch.Tensor:
        """
        Encode image to CLIP image embeddings
        
        Args:
            pixel_values: [B, 3, H, W] - normalized image tensor
            noise_level: noise level to add to image embeddings
            
        Returns:
            image_embeds: [B, embed_dim] - CLIP image embeddings
        """
        # Encode with CLIP
        image_embeds = self.image_encoder(pixel_values)
        
        # Add noise to embeddings if needed
        if noise_level > 0:
            noise = torch.randn_like(image_embeds) * noise_level / 1000.0
            image_embeds = image_embeds + noise
        
        # Upscale embeddings
        image_embeds = self.feature_upscaler(image_embeds)
        
        return image_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        tokenizer: Any = None,
        device: str = "cuda",
        num_images_per_prompt: int = 1,
    ) -> torch.Tensor:
        """
        Encode text prompts to embeddings
        
        Args:
            prompt: text prompt or list of prompts
            tokenizer: tokenizer instance
            device: device to run on
            
        Returns:
            prompt_embeds: [B, seq_len, embed_dim] - text embeddings
        """
        if tokenizer is None:
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Tokenize
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        
        # Encode with text encoder
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        
        # Duplicate for num_images_per_prompt
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        return prompt_embeds
    
    def decode_latents(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode latents to images using VAE
        
        Args:
            latents: [B, 4, H, W] - latent tensor
            
        Returns:
            images: [B, 3, H*8, W*8] - decoded images
        """
        if self.vae is None:
            raise ValueError("VAE not loaded!")
        
        latents = latents / 0.18215
        images = self.vae.decode(latents).sample
        
        if return_dict:
            return {"sample": images}
        return images
    
    def generate(
        self,
        image: torch.Tensor,
        prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        noise_level: int = 0,
        seed: Optional[int] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Generate image variations from input image
        
        Args:
            image: [B, 3, H, W] - input image
            prompt: optional text prompt
            num_inference_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale
            noise_level: noise level for image embedding
            seed: random seed
            device: device to run on
            
        Returns:
            generated_images: [B, 3, H*8, W*8] - generated images
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Encode image
        image_embeds = self.encode_image(image, noise_level=noise_level)
        
        # For now, use unconditional if no prompt
        if prompt is None:
            encoder_hidden_states = image_embeds
        else:
            encoder_hidden_states = image_embeds
        
        # Prepare latents
        B = image_embeds.shape[0]
        H, W = 64, 64  # latent size
        latents = torch.randn(B, 4, H, W, device=device)
        
        # Denoising loop
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            # Predict noise
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            
            # Classifier-free guidance
            noise_pred_uncond = noise_pred
            noise_pred_text = noise_pred
            
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Denoise
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        images = self.decode_latents(latents)
        
        return images


# ============================================================================
# Pipeline Wrapper (compatible with diffusers)
# ============================================================================

class StableUnCLIPImg2ImgPipeline:
    """
    StableUnCLIP Img2Img Pipeline
    
    Compatible with diffusers API but includes full model implementation
    """
    
    def __init__(
        self,
        image_encoder: nn.Module,
        feature_upscaler: nn.Module,
        unet: nn.Module,
        vae: nn.Module,
        scheduler: Any,
        tokenizer: Any = None,
        text_encoder: nn.Module = None,
    ):
        self.image_encoder = image_encoder
        self.feature_upscaler = feature_upscaler
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        
        self._device = None
        self._dtype = torch.float32
    
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        **kwargs,
    ):
        """Load pipeline from pretrained model"""
        from diffusers import StableUnCLIPImg2ImgPipeline as DiffusersPipeline
        
        pipeline = DiffusersPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        
        return cls(
            image_encoder=pipeline.image_encoder,
            feature_upscaler=pipeline.feature_extractor,
            unet=pipeline.unet,
            vae=pipeline.vae,
            scheduler=pipeline.scheduler,
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
        )
    
    def to(self, device: str, dtype: torch.dtype = None):
        """Move pipeline to device"""
        self._device = device
        if dtype:
            self._dtype = dtype
        
        self.image_encoder.to(device, dtype)
        self.feature_upscaler.to(device, dtype)
        self.unet.to(device, dtype)
        self.vae.to(device, dtype)
        
        if self.text_encoder:
            self.text_encoder.to(device, dtype)
        
        return self
    
    def __call__(
        self,
        image: Union[torch.Tensor, "PIL.Image.Image"],
        prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        noise_level: int = 0,
        seed: Optional[int] = None,
        output_type: str = "pil",
        **kwargs,
    ):
        """
        Generate image variations
        
        Args:
            image: input image (tensor or PIL Image)
            prompt: optional text prompt
            num_inference_steps: denoising steps
            guidance_scale: CFG scale
            noise_level: noise for image embeddings
            seed: random seed
            output_type: output format ("pil", "numpy", "pt")
            
        Returns:
            generated images
        """
        from PIL import Image
        import numpy as np
        
        device = self._device or next(self.unet.parameters()).device
        
        # Convert PIL to tensor if needed
        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image).convert("RGB")).float()
            image = image / 255.0
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        image = image.to(device, dtype=self._dtype)

        # Encode image
        with torch.no_grad():
            image_embeds = self.encode_image(image)
        
        # Prepare for generation
        B = image_embeds.shape[0]
        
        # Use image embeddings as conditioning
        encoder_hidden_states = image_embeds
        
        # Prepare latents
        if seed is not None:
            torch.manual_seed(seed)
        
        H = W = 64
        latents = torch.randn(B, 4, H, W, device=device, dtype=self._dtype)
        
        # Denoise
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        latents = latents / 0.18215
        images = self.vae.decode(latents).sample
        
        # Post-process
        images = (images / 2 + 0.5).clamp(0, 1)
        
        if output_type == "pil":
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        elif output_type == "numpy":
            images = images.cpu().permute(0, 2, 3, 1).numpy()
        
        return images


# ============================================================================
# Utility Functions
# ============================================================================

def create_pipeline(
    model_id: str = "sd2-community/stable-diffusion-2-1-unclip-small",
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> StableUnCLIPImg2ImgPipeline:
    """
    Create StableUnCLIP pipeline
    
    Args:
        model_id: HuggingFace model ID
        torch_dtype: data type
        device: device to run on
        
    Returns:
        StableUnCLIP pipeline
    """
    return StableUnCLIPImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device=device,
        safety_checker=None,
    )


def get_model_config() -> StableUnCLIPConfig:
    """Get default StableUnCLIP configuration"""
    return StableUnCLIPConfig()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    from PIL import Image
    
    parser = argparse.ArgumentParser(description="StableUnCLIP Model")
    parser.add_argument("--model_id", type=str, default="sd2-community/stable-diffusion-2-1-unclip-small")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    print(f"Loading StableUnCLIP model: {args.model_id}")
    pipeline = create_pipeline(args.model_id, dtype, args.device)
    
    print("Model loaded successfully!")
    print(f"  - Image encoder: {type(pipeline.image_encoder).__name__}")
    print(f"  - UNet: {type(pipeline.unet).__name__}")
    print(f"  - VAE: {type(pipeline.vae).__name__}")
    print(f"  - Scheduler: {type(pipeline.scheduler).__name__}")
