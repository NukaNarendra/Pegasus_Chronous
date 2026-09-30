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

This module implements the local PyTorch model client for
Part II mathematical containment experiments. It provides
white-box access to transformer internals, enabling:
- Forward hook registration for attention masking (Exp 5)
- Embedding extraction for topological analysis (Exp 4)
- Output logit access for TVD computation (Exp 5)

SCOPE BOUNDARY: Attention masking is ONLY possible on local
models where we control the forward pass. This does NOT
apply to API-served models (Part I).
"""
from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils.hooks
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

class LocalModelConfig(BaseModel):
    """Configuration for the local PyTorch model client."""
    model_id: str = Field(..., description="The HuggingFace model ID")
    device: str = Field(default='auto', description="Device to load the model on ('auto', 'cuda', 'cpu')")
    dtype: str = Field(default='float16', description="Data type for the model ('float16', 'bfloat16', 'float32')")
    max_tokens: int = Field(default=4096, description="Maximum context length/generation tokens")
    temperature: float = Field(default=1.0, description="Generation temperature")

class LocalModelResponse(BaseModel):
    """Structured response from the local model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    content: str
    model: str
    input_ids: Optional[torch.Tensor] = Field(default=None, description="Input tokens, for internal use")
    logits: Optional[torch.Tensor] = Field(default=None, description="Output logits, excluded from serialization", exclude=True)
    latency_ms: float

class AttentionHookHandle:
    """
    Handle for managing PyTorch forward hooks registered on attention modules.
    Provides context manager support for safe removal.
    """
    def __init__(self, handles: List[torch.utils.hooks.RemovableHandle]):
        self._handles = handles
        self._active = True

    @property
    def active(self) -> bool:
        """Return whether the hooks are currently active."""
        return self._active

    @property
    def num_hooks(self) -> int:
        """Return the number of registered hooks managed by this handle."""
        return len(self._handles)

    def remove(self) -> None:
        """Remove all managed hooks from the model."""
        if not self._active:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
        logger.info(f"Removed {self.num_hooks} attention hooks.")

    def __enter__(self) -> AttentionHookHandle:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove()

class LocalModelClient:
    """
    Local PyTorch model client for PEGASUS-CHRONOS mathematical containment experiments.
    Provides white-box access to transformer internals.
    """
    def __init__(self, config: LocalModelConfig):
        self.config = config
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._active_hooks: List[AttentionHookHandle] = []
        self._device: Optional[torch.device] = None

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype == "float16":
            return torch.float16
        elif self.config.dtype == "bfloat16":
            return torch.bfloat16
        elif self.config.dtype == "float32":
            return torch.float32
        return torch.float32

    async def load_model_async(self) -> None:
        """Async wrapper for model loading (I/O bound)."""
        await asyncio.to_thread(self.load_model)

    def load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace synchronously."""
        if self._model is not None:
            return
            
        logger.info(f"Loading local model {self.config.model_id} on {self.config.device} with dtype {self.config.dtype}...")
        try:
            dtype = self._resolve_dtype()
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            
            # Setup device map based on config
            device_map = self.config.device if self.config.device != 'cpu' else None
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
                device_map=device_map
            )
            self._model.eval()
            self._device = next(self._model.parameters()).device
            
            param_count = sum(p.numel() for p in self._model.parameters())
            logger.info(f"Successfully loaded {self.config.model_id} with {param_count:,} parameters onto {self._device}.")
            
            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._model.config.pad_token_id = self._tokenizer.eos_token_id
                
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_id}: {e}", exc_info=True)
            self._model = None
            self._tokenizer = None
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def ensure_loaded(self) -> None:
        """Ensure the model is loaded, throwing an error if it fails."""
        if self._model is None or self._tokenizer is None:
            self.load_model()

    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None, 
        return_logits: bool = False
    ) -> LocalModelResponse:
        """Generate response asynchronously to prevent blocking the event loop."""
        return await asyncio.to_thread(self._generate_sync, messages, temperature, max_tokens, return_logits)

    def _generate_sync(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None, 
        return_logits: bool = False
    ) -> LocalModelResponse:
        """Synchronous implementation of text generation."""
        self.ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        
        start_time = time.perf_counter()
        
        temp = temperature if temperature is not None else self.config.temperature
        max_new_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            # Apply chat template
            prompt_text = self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self._tokenizer(prompt_text, return_tensors='pt').to(self.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            with torch.inference_mode():
                if return_logits:
                    outputs = self._model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temp,
                        do_sample=(temp > 0),
                        pad_token_id=self._tokenizer.pad_token_id,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                    generated_sequences = outputs.sequences
                    
                    # Convert tuple of scores into a tensor of shape (batch, seq_len, vocab_size)
                    logits = torch.stack(outputs.scores, dim=1) if outputs.scores else None
                else:
                    generated_sequences = self._model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temp,
                        do_sample=(temp > 0),
                        pad_token_id=self._tokenizer.pad_token_id
                    )
                    logits = None

            # Extract only the newly generated tokens
            input_len = input_ids.shape[1]
            generated_tokens = generated_sequences[0, input_len:]
            content = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            latency = (time.perf_counter() - start_time) * 1000.0
            
            return LocalModelResponse(
                content=content,
                model=self.config.model_id,
                input_ids=input_ids,
                logits=logits,
                latency_ms=latency
            )
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("CUDA OutOfMemoryError during generation.", exc_info=True)
            torch.cuda.empty_cache()
            raise RuntimeError("CUDA OOM during generation") from e
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise

    def register_attention_mask_hook(self, quarantined_positions: List[int]) -> AttentionHookHandle:
        r"""
        Register forward hooks on ALL attention layers that mask out
        quarantined token positions.
        
        For each attention layer, we register a hook that modifies the
        attention mask to set weights for quarantined positions to -inf.
        
        This implements the Bifurcated Attention Masking defense (Experiment 5).
        
        Math: For quarantined positions Q, we set:
            $$ attention\_mask[:, :, :, q] = -\infty \quad \forall q \in Q $$
        This ensures $softmax(attention\_weights)[:, :, :, q] = 0$,
        making the quarantined tokens invisible to the model's
        decision-making process.
        
        Args:
            quarantined_positions: Token position indices to mask
            
        Returns:
            AttentionHookHandle that can be used to remove hooks
        """
        self.ensure_loaded()
        assert self._model is not None
        
        handles: List[torch.utils.hooks.RemovableHandle] = []
        
        def make_hook(positions: List[int]):
            def hook_fn(module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    # Clone to avoid in-place modification issues in autograd/backward
                    mask = kwargs["attention_mask"].clone()
                    dtype_min = torch.finfo(mask.dtype).min
                    
                    # Expected mask shape usually (batch, 1, tgt_len, src_len) or similar
                    for q in positions:
                        if q < mask.shape[-1]:
                            # Mask out the key position q across all heads and queries
                            mask[..., q] = dtype_min
                            
                    kwargs["attention_mask"] = mask
                return args, kwargs
            return hook_fn
            
        # Find all attention layers. In Llama architectures, these are typically self_attn
        # We walk through named modules to dynamically find them regardless of exact model type
        attention_modules = []
        for name, module in self._model.named_modules():
            if "self_attn" in name or "attention" in name.lower():
                # Filter down to the actual attention class to avoid hooking containers
                if hasattr(module, "q_proj") or hasattr(module, "query_key_value"):
                    attention_modules.append(module)
                    
        if not attention_modules:
            logger.warning("No attention modules found matching 'self_attn' or 'attention'. Hook registration may have failed.")
            
        for idx, module in enumerate(attention_modules):
            handle = module.register_forward_pre_hook(
                make_hook(quarantined_positions), 
                with_kwargs=True
            )
            handles.append(handle)
            
        logger.info(f"Registered attention mask hooks on {len(handles)} modules for positions: {quarantined_positions}")
        
        hook_handle = AttentionHookHandle(handles)
        self._active_hooks.append(hook_handle)
        return hook_handle

    async def extract_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Async wrapper for extracting embeddings."""
        return await asyncio.to_thread(self._extract_embeddings_sync, texts)

    def _extract_embeddings_sync(self, texts: List[str]) -> torch.Tensor:
        """
        Extract token embeddings by mean-pooling the last hidden state.
        Used by Experiment 4 (topology) to get embeddings for Dirichlet Energy.
        """
        self.ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        
        try:
            inputs = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.inference_mode():
                outputs = self._model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # outputs.hidden_states is a tuple of all layers, take the last one
                last_hidden_state = outputs.hidden_states[-1]
                attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size())
                
                # Mean pool over sequence length, factoring in attention mask
                sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                
                pooled_embeddings = sum_embeddings / sum_mask
                return pooled_embeddings
                
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Embedding extraction failed: {e}") from e

    async def get_output_distribution(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Async wrapper for getting output distribution."""
        return await asyncio.to_thread(self._get_output_distribution_sync, messages)

    def _get_output_distribution_sync(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """
        Run forward pass (no generation) to get softmax of logits at the last token position.
        Used by Experiment 5 for TVD computation.
        """
        self.ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self._tokenizer(prompt_text, return_tensors='pt').to(self.device)
            
            with torch.inference_mode():
                outputs = self._model(**inputs, return_dict=True)
                
                # Get logits for the last token in the sequence for each batch
                # Shape: (batch_size, sequence_length, vocab_size) -> (batch_size, vocab_size)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Compute softmax distribution
                distribution = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Return squeezed tensor if batch size is 1 to match expected (vocab_size,) shape
                if distribution.shape[0] == 1:
                    return distribution.squeeze(0)
                return distribution
                
        except Exception as e:
            logger.error(f"Failed to get output distribution: {e}", exc_info=True)
            raise RuntimeError(f"Output distribution computation failed: {e}") from e

    def remove_all_hooks(self) -> int:
        """Remove all active hooks and return the count of removed hooks."""
        total_removed = 0
        for handle in self._active_hooks:
            total_removed += handle.num_hooks
            handle.remove()
        self._active_hooks.clear()
        logger.info(f"Removed all hooks (total: {total_removed}).")
        return total_removed

    def unload_model(self) -> None:
        """Unload the model and tokenizer, clear CUDA cache to free memory."""
        self.remove_all_hooks()
        
        if self._model is not None:
            del self._model
            self._model = None
            
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            
        self._device = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()
        logger.info(f"Unloaded model {self.config.model_id} and cleared cache.")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return self._model is not None

    @property
    def device(self) -> torch.device:
        """Get the primary device the model is loaded on."""
        self.ensure_loaded()
        if self._device is None:
            # Fallback to cpu if device hasn't been cached
            return torch.device("cpu")
        return self._device

    @property
    def hidden_size(self) -> int:
        """Get the hidden dimension size of the transformer."""
        self.ensure_loaded()
        assert self._model is not None
        return getattr(self._model.config, "hidden_size", 0)

    @property
    def num_attention_heads(self) -> int:
        """Get the number of attention heads in the transformer."""
        self.ensure_loaded()
        assert self._model is not None
        return getattr(self._model.config, "num_attention_heads", 0)

    @property
    def num_layers(self) -> int:
        """Get the number of hidden layers in the transformer."""
        self.ensure_loaded()
        assert self._model is not None
        return getattr(self._model.config, "num_hidden_layers", 0)
