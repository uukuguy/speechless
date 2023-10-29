# Derived from modeling_mistral

import inspect
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import transformers
from loguru import logger

from transformers.utils import is_flash_attn_available
if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from transformers.models.mistral.modeling_mistral import (
    repeat_kv,
    _get_unpad_data,
    apply_rotary_pos_emb,
)

def sliding_window_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
):
    # print(f"sliding_window_forward()")
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    use_sliding_windows = (
        _flash_supports_window_size
        and hasattr(self.config, "sliding_window") is not None
        and kv_seq_len > self.config.sliding_window
    )

    if not _flash_supports_window_size:
        logger.warning_once(
            "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
            " make sure to upgrade flash-attn library."
        )

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        if hasattr(self.config, "sliding_window") and kv_seq_len > self.config.sliding_window:
            slicing_tokens = kv_seq_len - self.config.sliding_window

            past_key = past_key_value[0]
            past_value = past_key_value[1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key much have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            past_key_value = (past_key, past_value)

            if padding_mask is not None:
                padding_mask = padding_mask[:, slicing_tokens:]
                padding_mask = torch.cat([padding_mask, torch.ones_like(padding_mask[:, -1:])], dim=-1)

        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # TODO: Mistral does not have dropout in the config??
    # It is recommended to use dropout with FA according to the docs
    # when training.
    dropout_rate = 0.0  # if not self.training else self.attn_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        logger.warning_once(
            "The input hidden states seems to be silently casted in float32, this might be related to"
            " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            " float16."
        )

        query_states = query_states.to(torch.float16)
        key_states = key_states.to(torch.float16)
        value_states = value_states.to(torch.float16)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _flash_attention_forward(self,
        query_states,
        key_states,
        value_states,
        padding_mask,
        q_len,
        dropout=dropout_rate,
        use_sliding_windows=use_sliding_windows,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def _flash_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    padding_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    use_sliding_windows=False,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        padding_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_sliding_windows (`bool`, *optional*):
            Whether to activate sliding window attention.
    """
    # Contains at least one padding token in the sequence
    if padding_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(self,
            query_states, key_states, value_states, padding_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        if not use_sliding_windows:
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )
        else:
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(self.config.sliding_window, self.config.sliding_window),
            )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        if not use_sliding_windows:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(self.config.sliding_window, self.config.sliding_window),
            )

    return attn_output

def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length):
    batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

    # On the first iteration we need to properly re-create the padding mask
    # by slicing it on the proper place
    if kv_seq_len != padding_mask.shape[-1]:
        padding_mask_num_tokens = padding_mask.shape[-1]
        padding_mask = padding_mask[:, padding_mask_num_tokens - kv_seq_len :]

    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
    value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        padding_mask = padding_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )

def replace_llama_attn():
    print(f"Replacing llama attention with sliding window attention from Mistral")
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = sliding_window_forward