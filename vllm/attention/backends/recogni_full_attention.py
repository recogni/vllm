"""Attention layer with Flash and PagedAttention.

NOTE(woosuk): At the moment, this file includes a lot of duplicated code from
XFormers backend. The duplicated code will be removed once we use flash-attn or
flashinfer for all the attention operations.
"""

from typing import Dict, List, Optional, Tuple, Type

import flash_attn
import torch
from recogni.torch.nn.modules.attention import FullAttention

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention


class RecogniFullAttentionBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls() -> Type["RecogniFullAttentionImpl"]:
        return RecogniFullAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "FlashAttentionMetadata":
        return FlashAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


class RecogniFullAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = (
            (sliding_window, sliding_window)
            if sliding_window is not None
            else (-1, -1)
        )
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}."
            )
        self.rfull_attn = FullAttention()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata[FlashAttentionMetadata],
        kv_scale: float,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                attn_metadata.kv_cache_dtype,
                kv_scale,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]

        if prefill_meta := attn_metadata.prefill_metadata:
            # QKV for prefill.
            query = query[:num_prefill_tokens]
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]

            assert query.shape[0] == num_prefill_tokens
            assert decode_query.shape[0] == num_decode_tokens
            # Prompt run.
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                if attn_metadata.is_init_run:
                    out = flash_attn.flash_attn_varlen_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=prefill_meta.seq_start_loc,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_q=prefill_meta.max_prompt_len,
                        max_seqlen_k=prefill_meta.max_prompt_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.sliding_window,
                        alibi_slopes=self.alibi_slopes,
                    )
                else:
                    _value = value[None, ...]
                    _value = _value.repeat_interleave(
                        self.num_queries_per_kv, dim=2
                    )
                    _value = _value.transpose(1, 2)

                    _key = key[None, ...]
                    _key = _key.repeat_interleave(
                        self.num_queries_per_kv, dim=2
                    )
                    _key = _key.transpose(1, 2)

                    _query = query[None, ...]
                    _query = _query.transpose(1, 2)
                    out = self.rfull_attn(
                        q=_query * self.scale**0.5,
                        k=_key * self.scale**0.5,
                        v=_value,
                        is_causal=True,
                    )[0, ...].movedim(0, 1)

                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                raise NotImplementedError("This should not happen.")
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.

            _value = (
                value_cache[decode_meta.block_tables]
                .movedim(-1, 2)  # Move block size to front
                .reshape(1, -1, self.num_kv_heads, self.head_size)
            )
            _key = (
                key_cache[decode_meta.block_tables]
                .movedim(-2, 2)  # Move block size to front
                .reshape(1, -1, self.num_kv_heads, self.head_size)
            )
            decode_query = decode_query.view(-1, self.num_heads, self.head_size)
            _key = _key.repeat_interleave(self.num_queries_per_kv, dim=2)
            _value = _value.repeat_interleave(self.num_queries_per_kv, dim=2)
            _decode_query = decode_query[..., None, :]
            _value = _value.transpose(1, 2)
            _key = _key.transpose(1, 2)

            context_len = decode_meta.context_lens[0]
            shape = _key.shape
            mask = (
                torch.arange(shape[2], device=_key.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
            )  # [1, 1, dim, 1]
            mask = mask < context_len

            # Expand the mask to match the tensor dimensions
            # mask = mask.expand(shape)
            _key = _key * mask
            _value = _value * mask
            # _key = _key[:, :, : decode_meta.context_lens[0], :]
            # _value = _value[:, :, : decode_meta.context_lens[0], :]
            output[num_prefill_tokens:] = self.rfull_attn(
                q=_decode_query * self.scale**0.5,
                k=_key * self.scale**0.5,
                v=_value,
                is_causal=False,
            ).squeeze(-2)

            # output[num_prefill_tokens:] = scaled_dot_product_attention(
            #     _decode_query,
            #     _key[:, :, : decode_meta.context_lens[0], :],
            #     _value[:, :, : decode_meta.context_lens[0], :],
            #     scale=self.scale,
            # ).squeeze(-2)

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)
