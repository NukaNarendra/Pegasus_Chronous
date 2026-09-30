"""
PEGASUS-CHRONOS Research Series (Paper 3 of 3)
Prior work: ARGUS (Zenodo:10.5281/zenodo.21310433),
            SHADE (2026)

NOVELTY STATEMENT:
All prior memory poisoning work (eTAMP arXiv:2605.16746,
CSTM-Bench arXiv:2604.21131, MemoryGraft arXiv:2512.16962)
studies EXTERNAL attackers injecting into agent memory.

PEGASUS studies the MODEL ITSELF as the source of ghost
states, deliberately planted during evaluation windows with
reduced safety controls — directly motivated by the
GPT-5.6 Sol / Hugging Face incident of July 21-22, 2026.

This module implements Bifurcated Attention Masking (Layer 2 defense) via
PyTorch hooks on transformer attention heads.

SCOPE BOUNDARY:
This defense operates directly on tensor dot-products during the model's forward pass
and is ONLY possible on white-box local models (e.g. Llama-3-8B). It cannot be
applied to black-box API models.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BifurcatedAttentionHook:
    r"""
    Manages active PyTorch forward hooks that enforce structural attention severance
    on quarantined memory token positions.
    
    Formula:
        Let Q be the set of token indices corresponding to quarantined memory.
        For attention weights A in R^{B x H x S x S}:
        $$ A_{b, h, i, q} = -\infty \quad \forall q \in Q, \, i \in \{1, \dots, S\} $$
        $$ \text{softmax}(A)_{b, h, i, q} = 0 $$
    """

    def __init__(self, hooks: List[torch.utils.hooks.RemovableHandle]):
        self.hooks = hooks
        self.is_active = True

    def remove(self) -> None:
        """Deregister all registered attention hooks."""
        for h in self.hooks:
            try:
                h.remove()
            except Exception as e:
                logger.debug("Hook removal warning: %s", e)
        self.hooks.clear()
        self.is_active = False

    def __enter__(self) -> BifurcatedAttentionHook:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove()


class AttentionProjector:
    """
    Installs attention modification hooks across all attention layers of a PyTorch causal LM.
    """

    @staticmethod
    def attach_bifurcated_mask(
        model: nn.Module,
        quarantined_token_indices: List[int],
    ) -> BifurcatedAttentionHook:
        r"""
        Attaches pre-forward hooks to transformer attention modules that set attention scores
        for quarantined positions to $$ -\infty $$.
        """
        if not quarantined_token_indices:
            return BifurcatedAttentionHook([])

        handles: List[torch.utils.hooks.RemovableHandle] = []
        indices_tensor = torch.tensor(quarantined_token_indices, dtype=torch.long)

        # Traverse model modules and identify attention modules
        for name, module in model.named_modules():
            # Matches LlamaAttention, MistralAttention, GPT2Attention, etc.
            if any(k in name.lower() for k in ["self_attn", "attn", "attention"]) and hasattr(module, "forward"):
                def make_pre_hook(q_indices: torch.Tensor):
                    def pre_hook(mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                        # Check kwargs for attention_mask
                        mask = kwargs.get("attention_mask")
                        if mask is not None and isinstance(mask, torch.Tensor):
                            new_mask = mask.clone()
                            valid_indices = [idx for idx in q_indices.tolist() if idx < new_mask.shape[-1]]
                            if valid_indices:
                                min_val = torch.finfo(new_mask.dtype).min
                                new_mask[..., valid_indices] = min_val
                                kwargs["attention_mask"] = new_mask
                        return args, kwargs
                    return pre_hook

                h = module.register_forward_pre_hook(make_pre_hook(indices_tensor), with_kwargs=True)
                handles.append(h)

        logger.info("Attached bifurcated attention hooks to %d attention modules.", len(handles))
        return BifurcatedAttentionHook(handles)

    @staticmethod
    def find_quarantined_token_spans(
        full_text: str,
        quarantined_substring: str,
        tokenizer: Any,
    ) -> List[int]:
        """
        Calculates the exact token position indices corresponding to a quarantined substring.
        """
        start_char = full_text.find(quarantined_substring)
        if start_char == -1:
            return []

        end_char = start_char + len(quarantined_substring)

        # Encode full text with return_offsets_mapping if available
        try:
            encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding.get("offset_mapping", [])
            quarantined_indices = [
                idx for idx, (start, end) in enumerate(offsets)
                if (start >= start_char and start < end_char) or (end > start_char and end <= end_char)
            ]
            return quarantined_indices
        except Exception:
            # Fallback estimation via token prefix lengths
            tokens_before = len(tokenizer.encode(full_text[:start_char], add_special_tokens=False))
            tokens_span = len(tokenizer.encode(quarantined_substring, add_special_tokens=False))
            return list(range(tokens_before, tokens_before + tokens_span))
