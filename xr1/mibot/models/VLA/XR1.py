# Copyright (C) 2026 Xiaomi Corporation.
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributions import Beta
from transformers import Qwen3VLConfig
from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, rotate_half
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from mibot.models import MIMODEL
from mibot.models.VLM.qwen3vl import (
    ACTION_END_ID,
    ACTION_START_ID,
    SCORE_ID,
    STATE_ID,
    Qwen3VLForConditionalGeneration,
)
from mibot.utils.model_utils import auto_cast


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def repeat_kv(x, repeats):
    return x.repeat_interleave(repeats, dim=1)


def repeat_batch(x, batch_size):
    if x.shape[0] == batch_size:
        return x
    if batch_size % x.shape[0] != 0:
        raise ValueError(f"Cannot repeat batch of size {x.shape[0]} to {batch_size}")
    return x.repeat_interleave(batch_size // x.shape[0], dim=0)


def apply_rotary_pos_emb(query, key, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, inter_dim=None, num_layers=1, bias=False):
        super().__init__()
        inter_dim = output_dim if inter_dim is None else inter_dim
        if num_layers == 1:
            layers = [nn.Linear(input_dim, output_dim, bias=bias)]
        else:
            layers = [nn.Linear(input_dim, inter_dim, bias=bias)]
            for _ in range(1, num_layers - 1):
                layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, inter_dim, bias=bias)])
            layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, output_dim, bias=bias)])
        self.layers = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, timestep):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / half
        )
        args = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
        return self.mlp(embedding)[:, None]


class Attention(nn.Module):
    def __init__(self, hidden_size=1024, head_dim=128, kv_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // kv_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(head_dim)
        self.k_norm = Qwen2RMSNorm(head_dim)

    def forward(self, hidden_states, past_key_values, position_embeds, attn_mask):
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeds
        if cos.ndim == 4:
            cos, sin = cos[0], sin[0]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        cache_key, cache_value = past_key_values
        cache_key = repeat_kv(repeat_batch(cache_key, batch_size), self.kv_group)
        cache_value = repeat_kv(repeat_batch(cache_value, batch_size), self.kv_group)
        key = torch.cat([cache_key, key], dim=-2)
        value = torch.cat([cache_value, value], dim=-2)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.attn = Attention(hidden_size=hidden_size)
        self.mlp = MLP(hidden_size=hidden_size)
        self.input_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.post_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, hidden_states, past_key_values, position_embeds, timestep, attn_mask):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_table[None] + timestep
        ).chunk(6, dim=1)
        residual = hidden_states
        hidden_states = modulate(self.input_layernorm(hidden_states), shift_attn, scale_attn)
        hidden_states = residual + gate_attn * self.attn(
            hidden_states, past_key_values, position_embeds, attn_mask
        )
        residual = hidden_states
        hidden_states = modulate(self.post_layernorm(hidden_states), shift_mlp, scale_mlp)
        return residual + gate_mlp * self.mlp(hidden_states)


class DiT(nn.Module):
    def __init__(self, layer_num=36, hidden_size=1024):
        super().__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList([DecoderLayer(hidden_size) for _ in range(layer_num)])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states, past_key_values, attn_mask, position_embeds, timestep):
        start = len(past_key_values) - self.layer_num
        if start < 0:
            raise ValueError("VLM cache has fewer layers than DiT")
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                past_key_values[start + index],
                position_embeds,
                timestep,
                attn_mask,
            )
        return hidden_states


@MIMODEL.register_module()
class xr1(nn.Module):
    def __init__(
        self,
        freq_coefficient=1.0,
        freq_excluded_dims=None,
        ffn_gradient_checkpointing=True,
        async_train=True,
    ):
        super().__init__()
        self.state_shape = (1, 60)
        self.action_shape = (30, 60)
        self.freq_coefficient = float(freq_coefficient)
        self.freq_excluded_dims = list([17, 18, 19] if freq_excluded_dims is None else freq_excluded_dims)
        self.ffn_gradient_checkpointing = bool(ffn_gradient_checkpointing)
        self.async_train = bool(async_train)
        self.training_repeat = 4
        self.num_steps = 5
        self.prefix_mask_prob = 0.5
        self.n_choices = 5
        self.beta = Beta(1.5, 1.0)
        self._build_model()

    def _build_model(self):
        config = Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = "flash_attention_2"
        if not torch.cuda.is_available() and torch.xpu.is_available():
            # Intel XPU: flash-attn is CUDA-only, so use PyTorch SDPA.
            attn_implementation = "sdpa"
        self.vlm = Qwen3VLForConditionalGeneration._from_config(
            config,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        ).train()
        self.vlm.model.get_input_embeddings().requires_grad_(False)
        self.vlm.model.visual.gradient_checkpointing_enable()
        if self.ffn_gradient_checkpointing:
            for layer in self.vlm.model.language_model.layers:
                mlp = layer.mlp
                original_forward = mlp.forward

                def checkpointed_forward(x, forward=original_forward, module=mlp):
                    if module.training and torch.is_grad_enabled():
                        return torch.utils.checkpoint.checkpoint(forward, x, use_reentrant=False)
                    return forward(x)

                mlp.forward = checkpointed_forward

        vlm_hidden_size = self.vlm.config.text_config.hidden_size
        self.state_projector_choice = Projector(self.state_shape[-1], vlm_hidden_size, num_layers=2)
        self.action_projector_choice = nn.Sequential(
            Projector(vlm_hidden_size, vlm_hidden_size, num_layers=4),
            Projector(vlm_hidden_size, self.action_shape[-1] * self.n_choices),
        )
        self.score_projector_choice = nn.Sequential(
            Projector(vlm_hidden_size, vlm_hidden_size, num_layers=4),
            Projector(vlm_hidden_size, self.n_choices),
        )
        self.dit = DiT()
        self.state_projector = Projector(self.state_shape[-1], 1024, num_layers=2)
        self.action_projector = Projector(self.action_shape[-1], 1024, num_layers=2)
        self.action_output_layer = Projector(1024, self.action_shape[-1], inter_dim=1024, num_layers=2)
        self.t_embedder = TimestepEmbedder(1024)
        self.t_projector = Projector(1024, 6 * 1024, bias=True)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(self.vlm.config.text_config)
        self.sink = nn.Embedding(1, 1024)
        self.to(torch.bfloat16)

    @staticmethod
    def _unpad(past_key_values, position_ids, attention_mask, segments):
        if segments is None:
            return past_key_values, position_ids, attention_mask
        keys = torch.stack([layer[0] for layer in past_key_values])
        values = torch.stack([layer[1] for layer in past_key_values])
        lengths = segments[:, 1] - segments[:, 0]
        max_length = int(lengths.max().item())
        layer_num, _, head_num, _, head_dim = keys.shape
        batch_size = len(segments)
        new_keys = keys.new_zeros(layer_num, batch_size, head_num, max_length, head_dim)
        new_values = values.new_zeros(new_keys.shape)
        new_positions = position_ids.new_zeros(3, batch_size, max_length)
        attention_mask = torch.zeros(batch_size, max_length, device=keys.device, dtype=torch.bool)
        for index, (start, end) in enumerate(segments.tolist()):
            length = end - start
            new_keys[:, index, :, :length] = keys[:, 0, :, start:end]
            new_values[:, index, :, :length] = values[:, 0, :, start:end]
            new_positions[:, index, :length] = position_ids[:, 0, start:end]
            attention_mask[index, :length] = True
        return list(zip(new_keys, new_values)), new_positions, attention_mask

    def _repeat(self, x, dim=0):
        if not self.training:
            return x
        return x.repeat_interleave(self.training_repeat, dim=dim)

    def _random_mask_prefix(self, mask, prefix_length, state_length, keep_last=2):
        if prefix_length <= keep_last:
            return mask
        action_start = 1 + state_length
        prefix_end = action_start + prefix_length - keep_last
        suffix_start = action_start + prefix_length
        random_mask = torch.rand(prefix_length - keep_last, device=mask.device) < self.prefix_mask_prob
        mask = mask.clone()
        mask[:, suffix_start:, action_start:prefix_end] *= (~random_mask).int()
        return mask

    def dit_forward(
        self,
        noisy_action,
        timestep,
        action_mask,
        state_embed,
        position_embeds,
        past_key_values,
        attn_mask,
        prefix_length,
    ):
        timestep = self.t_projector(self.t_embedder(timestep[:, 0, 0] * 1000)).view(-1, 6, 1024)
        noisy_action = self.action_projector(noisy_action * action_mask)
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()
        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, timestep)
        output = self.action_output_layer(hidden_states[:, -noisy_action.shape[1] :])
        if prefix_length:
            output[:, :prefix_length] = 0.0
        return output

    @torch.no_grad()
    def _generate(self, noise, kwargs):
        sample = noise.clone()
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            timestep = torch.ones(
                (sample.shape[0], 1, 1), device=sample.device, dtype=sample.dtype
            ) * step / self.num_steps
            sample = sample + self.dit_forward(sample, timestep, **kwargs) * dt
        return sample

    @torch.no_grad()
    def generate(self, batch):
        return self.forward(batch, return_loss=False)

    @auto_cast
    def forward(self, batch, return_loss=False):
        segments = batch.pop("action_vlm_condition_segments", None)
        batch.pop("action_segments", None)
        action = batch.pop("action")
        action_mask = batch.pop("action_mask")
        state = batch.pop("state")
        if self.training:
            prefix_length = random.randint(1, 6) if self.async_train and random.random() < 0.5 else 0
        else:
            prefix_length = batch.pop("prefix_length", 0)

        batch.pop("labels", None)
        if self.training:
            actual_lengths = batch.pop("vlm_action_actual_length")
            choice_target = batch.pop("vlm_action_target")
            choice_mask = batch.pop("vlm_action_mask")
            state_embeds = self.state_projector_choice(state.flatten(0, 1))
            vlm_outputs = self.vlm(
                **batch, state_embeds=state_embeds, use_cache=True, skip_logits=True
            )
            action_choice = self.action_projector_choice(
                vlm_outputs.hidden_states[
                    (batch["input_ids"] >= ACTION_START_ID) & (batch["input_ids"] < ACTION_END_ID)
                ]
            )
            score_choice = self.score_projector_choice(
                vlm_outputs.hidden_states[batch["input_ids"] == SCORE_ID]
            )
        else:
            vlm_outputs = self.vlm(**batch, use_cache=True, skip_logits=True)

        past_key_values, position_ids, cache_mask = self._unpad(
            list(vlm_outputs.past_key_values),
            vlm_outputs.position_ids,
            vlm_outputs.attention_mask,
            segments,
        )
        batch_size, action_length, _ = action.shape
        state_length = state.shape[1]
        query_length = 1 + state_length + action_length
        prefix = action[:, :prefix_length]

        dit_position_ids = (
            torch.arange(query_length, device=action.device).view(1, 1, -1).repeat(3, batch_size, 1)
            + position_ids.max(dim=-1).values[..., None]
            + 1
        )
        if action_length > prefix_length:
            dit_position_ids[:, :, -(action_length - prefix_length) :] += 10
        cache_mask = cache_mask[:, None, :].expand(-1, query_length, -1)
        causal_mask = torch.tril(torch.ones(batch_size, query_length, query_length, device=action.device))
        if self.training and prefix_length > 2:
            causal_mask = self._random_mask_prefix(causal_mask, prefix_length, state_length)
        attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None].bool()

        state_embed = self.state_projector(state)
        dit_position_ids = self._repeat(dit_position_ids, dim=1)
        action = self._repeat(action)
        action_mask = self._repeat(action_mask)
        prefix = self._repeat(prefix)
        state_embed = self._repeat(state_embed)
        attn_mask = self._repeat(attn_mask)
        position_embeds = self.rotary_emb(action, dit_position_ids)

        noise = torch.randn_like(action)
        kwargs = dict(
            action_mask=action_mask,
            state_embed=state_embed,
            position_embeds=position_embeds,
            past_key_values=past_key_values,
            attn_mask=attn_mask,
            prefix_length=prefix_length,
        )
        if self.training:
            timestep = ((1 - self.beta.sample((action.shape[0],)).to(action.device)) * 0.999).to(action.dtype)
            timestep = timestep[:, None, None]
            noisy_action = (1 - timestep) * noise + timestep * action
            target = action - noise
            pred = self.dit_forward(
                torch.cat([prefix, noisy_action[:, prefix_length:]], dim=1), timestep, **kwargs
            )[:, prefix_length:]
            target = target[:, prefix_length:]
            if prefix_length:
                prefix_pred = self._generate(torch.cat([prefix, noise[:, prefix_length:]], dim=1), kwargs)
                weight = (prefix_pred[:, prefix_length:] - action[:, prefix_length:]).abs()
            else:
                weight = torch.ones_like(pred)
            action_mask = action_mask[:, prefix_length:]
        else:
            return self._generate(torch.cat([prefix, noise[:, prefix_length:]], dim=1), kwargs)

        if not return_loss:
            return pred

        loss_mse, loss_freq = self.compute_flow_loss(pred, target, action_mask, weight)
        loss_choice, loss_score = self.compute_choice_loss(
            action_choice, score_choice, choice_target, choice_mask, actual_lengths
        )
        loss = 0.5 * loss_mse + self.freq_coefficient * loss_freq + 0.5 * loss_choice + 0.5 * loss_score
        return {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_freq": loss_freq,
            "loss_l1": loss_choice,
            "loss_score": loss_score,
        }

    def compute_flow_loss(self, pred, target, action_mask, weight):
        pred, target, weight = pred.float(), target.float(), weight.float()
        if not torch.any(action_mask):
            zero = (pred.sum() + target.sum()) * 0.0
            return zero, zero
        action_mask = action_mask.bool()
        with torch.no_grad():
            weight = weight.clone()
            weight[action_mask] /= weight[action_mask].mean()
            weight.clamp_(0.5, 5.0)
        loss_mse = (F.mse_loss(pred, target, reduction="none") * weight)[action_mask].mean()
        freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
        valid_batch = action_mask[:, -1].any(dim=1)
        if not torch.any(valid_batch):
            return loss_mse, freq.sum() * 0.0
        freq_mask = action_mask[valid_batch, : freq.shape[1]].clone()
        dims = [dim for dim in self.freq_excluded_dims if dim < freq_mask.shape[-1]]
        freq_mask[:, :, dims] = False
        freq_weight = weight.mean(dim=(1, 2)).view(-1, 1, 1)
        loss_freq = (freq * freq_weight)[valid_batch][freq_mask].mean()
        return loss_mse, loss_freq

    def compute_choice_loss(self, action_pred, score_pred, target, mask, actual_lengths):
        lengths = [int(length) for length in actual_lengths.detach().cpu().tolist()]
        if sum(lengths) != action_pred.shape[0] or len(lengths) != score_pred.shape[0]:
            raise ValueError("Choice token counts do not match targets")
        action_losses, score_losses = [], []
        start = 0
        for sample_index, length in enumerate(lengths):
            end = start + length
            predictions = action_pred[start:end].reshape(length, self.n_choices, -1).transpose(0, 1)
            sample_target = target[start:end].unsqueeze(0).repeat(self.n_choices, 1, 1)
            sample_mask = mask[start:end].bool().unsqueeze(0).repeat(self.n_choices, 1, 1)
            if not sample_mask.any():
                raise ValueError(f"Choice target {sample_index} has no valid action values")
            absolute_error = F.l1_loss(predictions, sample_target, reduction="none")
            choice_error = absolute_error[sample_mask].reshape(self.n_choices, -1).mean(dim=-1)
            choice_index = choice_error.argmin()
            action_losses.append(choice_error[choice_index])
            score_losses.append(((score_pred[sample_index] - choice_error.detach()) ** 2).mean())
            start = end
        return torch.stack(action_losses).mean(), torch.stack(score_losses).mean()
