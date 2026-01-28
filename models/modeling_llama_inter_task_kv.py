# coding=utf-8
"""
Llama Model with Inter-Task KV Reuse
Based on transformers.models.llama.modeling_llama
Implements Sim-LLM paper's KV cache reuse logic
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaModel as _LlamaModel,
    LlamaForCausalLM as _LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    logger
)
from transformers.utils import add_start_docstrings_to_model_forward

from .inter_task_kv_manager import InterTaskKVManager, TaskCacheEntry


class LlamaModelWithKVReuse(_LlamaModel):
    """
    Llama Model with Inter-Task KV Reuse capability
    Extends the base LlamaModel to support KV cache reuse from similar tasks
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.kv_manager = None  # Will be set externally
        self.enable_kv_reuse = False
        self.current_task_id = None  # Task ID for auto-save during prefill
    
    def set_kv_manager(self, kv_manager: InterTaskKVManager):
        """Set the global KV manager"""
        self.kv_manager = kv_manager
        self.enable_kv_reuse = True
        print("[LlamaModel] KV Reuse enabled")
    
    def disable_kv_reuse(self):
        """Disable KV reuse for this forward pass"""
        self.enable_kv_reuse = False
    
    def enable_kv_reuse_mode(self):
        """Enable KV reuse for this forward pass"""
        self.enable_kv_reuse = True
    
    def set_current_task_id(self, task_id: Optional[str]):
        """Set the current task ID for auto-save during prefill"""
        self.current_task_id = task_id
    
    def clear_current_task_id(self):
        """Clear the current task ID"""
        self.current_task_id = None
    
    def _extract_last_layer_kv(self, next_cache, seq_len: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract last layer KV from next_cache with robust structure detection.
        
        Handles multiple cache formats:
        - Legacy tuple: tuple(tuple(k, v), tuple(k, v), ...)
        - DynamicCache: object with key_cache and value_cache lists
        - Other Cache implementations with __getitem__
        
        Args:
            next_cache: The cache object from forward pass
            seq_len: Expected minimum sequence length
            
        Returns:
            (key, value) tuple or None if extraction fails
        """
        if next_cache is None:
            print("[LlamaModel] _extract_last_layer_kv: next_cache is None")
            return None
        
        last_layer_idx = len(self.layers) - 1
        last_layer_kv = None
        
        try:
            # Method 1: Standard tuple of tuples - tuple(tuple(k, v), ...)
            if isinstance(next_cache, tuple):
                print(f"[LlamaModel] _extract_last_layer_kv: tuple detected, len={len(next_cache)}")
                
                # Debug: Check first and last items
                if len(next_cache) > 0:
                    for check_idx in [0, -1]:
                        item = next_cache[check_idx]
                        if item is None:
                            print(f"[LlamaModel] _extract_last_layer_kv: next_cache[{check_idx}] is None")
                        elif isinstance(item, tuple) and len(item) == 2:
                            k, v = item
                            if k is not None:
                                print(f"[LlamaModel] _extract_last_layer_kv: next_cache[{check_idx}] key.shape={k.shape}")
                            else:
                                print(f"[LlamaModel] _extract_last_layer_kv: next_cache[{check_idx}] key is None")
                        else:
                            print(f"[LlamaModel] _extract_last_layer_kv: next_cache[{check_idx}] type={type(item).__name__}")
                    
                    # Try to find any non-None layer
                    for idx in range(len(next_cache) - 1, -1, -1):
                        item = next_cache[idx]
                        if item is not None and isinstance(item, tuple) and len(item) == 2:
                            k, v = item
                            if k is not None and v is not None and hasattr(k, 'shape'):
                                kv_seq_len = k.shape[2]
                                print(f"[LlamaModel] _extract_last_layer_kv: Found valid KV at layer {idx}, key.shape={k.shape}")
                                if kv_seq_len >= seq_len:
                                    last_layer_kv = (k, v)
                                    break
                                else:
                                    print(f"[LlamaModel] _extract_last_layer_kv: kv_seq_len ({kv_seq_len}) < seq_len ({seq_len})")
            
            # Method 2: DynamicCache object (transformers >= 4.36)
            if last_layer_kv is None and hasattr(next_cache, 'key_cache') and hasattr(next_cache, 'value_cache'):
                print(f"[LlamaModel] _extract_last_layer_kv: DynamicCache detected, key_cache len={len(next_cache.key_cache)}")
                if len(next_cache.key_cache) > last_layer_idx:
                    k = next_cache.key_cache[last_layer_idx]
                    v = next_cache.value_cache[last_layer_idx]
                    if k is not None and v is not None:
                        kv_seq_len = k.shape[2]
                        print(f"[LlamaModel] _extract_last_layer_kv: DynamicCache extraction success, key.shape={k.shape}")
                        if kv_seq_len >= seq_len:
                            last_layer_kv = (k, v)
                        else:
                            print(f"[LlamaModel] _extract_last_layer_kv: kv_seq_len ({kv_seq_len}) < seq_len ({seq_len})")
                    else:
                        print(f"[LlamaModel] _extract_last_layer_kv: DynamicCache layer {last_layer_idx} has None KV")
            
            # Method 3: Try direct indexing (fallback)
            if last_layer_kv is None and hasattr(next_cache, '__getitem__') and not isinstance(next_cache, tuple):
                print(f"[LlamaModel] _extract_last_layer_kv: trying direct indexing")
                try:
                    layer_kv = next_cache[last_layer_idx]
                    if layer_kv is not None and isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                        k, v = layer_kv
                        if k is not None and v is not None and hasattr(k, 'shape'):
                            kv_seq_len = k.shape[2]
                            print(f"[LlamaModel] _extract_last_layer_kv: indexing extraction success, key.shape={k.shape}")
                            if kv_seq_len >= seq_len:
                                last_layer_kv = layer_kv
                except Exception as e:
                    print(f"[LlamaModel] _extract_last_layer_kv: indexing failed: {e}")
                    
        except Exception as e:
            print(f"[LlamaModel] _extract_last_layer_kv: extraction failed with error: {e}")
        
        if last_layer_kv is None:
            print(f"[LlamaModel] _extract_last_layer_kv: FAILED to extract any valid KV")
        
        return last_layer_kv
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """
        Compatibility wrapper: prefer parent implementation if available;
        otherwise fall back to transformers' _prepare_4d_causal_attention_mask,
        or a simple boolean mask.
        """
        # 1) Prefer parent's implementation when present
        if hasattr(super(), "_prepare_decoder_attention_mask"):
            return super()._prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)

        # 2) Try transformers helper
        try:
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
            return _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)
        except Exception:
            pass
        
        # 3) Fallback: simple causal mask
        bsz, tgt_len = input_shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        
        if attention_mask is None:
            attention_mask = torch.ones((bsz, tgt_len + past_key_values_length), dtype=torch.bool, device=device)
        
        # Create causal mask
        # Shape: (batch, 1, tgt_len, tgt_len + past_len)
        combined_len = tgt_len + past_key_values_length
        
        # Create causal mask (lower triangular)
        causal_mask = torch.triu(
            torch.ones((tgt_len, combined_len), dtype=torch.bool, device=device),
            diagonal=past_key_values_length + 1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, combined_len)
        
        # Expand attention mask
        expanded_attn_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, combined_len)
        
        # Combine masks
        combined_mask = causal_mask | ~expanded_attn_mask
        
        # Convert to float mask with -inf for masked positions
        attn_mask = torch.zeros((bsz, 1, tgt_len, combined_len), dtype=dtype, device=device)
        attn_mask.masked_fill_(combined_mask, torch.finfo(dtype).min)
        
        return attn_mask
    
    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        task_id: Optional[str] = None,
        reuse_kv: bool = False,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass with optional KV reuse
        
        Args:
            task_id: Unique identifier for the current task
            reuse_kv: Whether to attempt KV reuse for this forward pass
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # ============================================================
        # EMBEDDING COMPUTATION
        # ============================================================
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # NOTE: KV reuse is now handled externally by the test script.
        # When cache HIT occurs, the test script:
        # 1. Constructs past_key_values from cached KV
        # 2. Truncates input_ids to only the last token
        # 3. Passes cached KV to generate()
        # This way, the model naturally processes only the new token
        # while reusing the cached KV for the prefix.

        # Calculate past_key_values_length
        past_key_values_length = 0
        
        # Normalize past_key_values to a list of length num_layers (None for missing layers)
        num_layers = len(self.layers)
        if past_key_values is not None:
            if not isinstance(past_key_values, (list, tuple)):
                # Handle transformers.Cache or other cache-like objects
                logger.info("[LlamaModel] Received non-list past_key_values; normalizing to list")
                normalized_pkvs = []
                for i in range(num_layers):
                    try:
                        layer_kv = past_key_values[i]
                        normalized_pkvs.append(layer_kv)
                    except (KeyError, IndexError, TypeError) as e:
                        # transformers.Cache may raise KeyError for missing layers -> treat as None
                        normalized_pkvs.append(None)
                past_key_values = normalized_pkvs
            
            # Check if cache is effectively empty
            has_any_kv = False
            for layer_kv in past_key_values:
                if layer_kv is not None:
                    has_any_kv = True
                    try:
                        past_key_values_length = layer_kv[0].shape[2]
                        break
                    except (IndexError, AttributeError):
                        continue
            
            if not has_any_kv:
                logger.info("[LlamaModel] Received empty cache object; treating as no past KV")
                past_key_values = None

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, past_key_values_length + seq_length), dtype=torch.bool, device=inputs_embeds.device
            )
        
        # Prepare 4D attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        
        # ============================================================
        # STANDARD FORWARD PASS
        # ============================================================
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Defensive access to past_key_values
            try:
                past_key_value = past_key_values[idx] if past_key_values is not None else None
            except (KeyError, IndexError, TypeError) as e:
                past_key_value = None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                # Extract present_key_value from layer outputs
                # The index depends on output_attentions:
                # - If output_attentions=False: layer_outputs = (hidden_states, present_kv)
                # - If output_attentions=True: layer_outputs = (hidden_states, attn_weights, present_kv)
                present_kv_idx = 2 if output_attentions else 1
                
                # Handle different return types from decoder layer
                if len(layer_outputs) > present_kv_idx:
                    present_kv = layer_outputs[present_kv_idx]
                else:
                    present_kv = None
                
                # Debug logging for first and last layer
                if idx == 0 or idx == len(self.layers) - 1:
                    if present_kv is not None:
                        if isinstance(present_kv, tuple) and len(present_kv) == 2:
                            k, v = present_kv
                            print(f"[LlamaModel] Layer {idx} KV: key.shape={k.shape if k is not None else None}")
                        else:
                            print(f"[LlamaModel] Layer {idx} present_kv type: {type(present_kv).__name__}")
                    else:
                        print(f"[LlamaModel] Layer {idx} present_kv is None (layer_outputs len={len(layer_outputs)})")
                
                next_decoder_cache += (present_kv,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        
        # ============================================================
        # SAVE TO CACHE (if this was a cache miss during prefill)
        # ============================================================
        # Determine the effective task_id: prefer explicit parameter, fallback to current_task_id attribute
        effective_task_id = task_id if task_id is not None else self.current_task_id
        
        # Get actual sequence length from input
        actual_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        
        # Auto-save conditions:
        # 1. KV reuse is enabled
        # 2. KV manager is available
        # 3. This is a prefill phase (actual_seq_len > 1, indicating prompt processing not single-token generation)
        # 4. We have a valid task_id
        # 5. No past_key_values were provided (indicating this is a fresh prefill, not a cache hit)
        is_prefill_phase = actual_seq_len > 1
        is_fresh_prefill = past_key_values_length == 0  # No cached KV was used
        should_auto_save = (
            self.enable_kv_reuse and
            self.kv_manager is not None and
            is_prefill_phase and
            is_fresh_prefill and
            effective_task_id is not None
        )
        
        if should_auto_save:
            print(f"[LlamaModel] Auto-save check: seq_len={actual_seq_len}, task_id={effective_task_id}, use_cache={use_cache}")
            
            # Detailed debug logging for next_cache structure
            print(f"[LlamaModel] next_cache type: {type(next_cache).__name__}")
            if next_cache is not None:
                if isinstance(next_cache, (list, tuple)):
                    print(f"[LlamaModel] next_cache length: {len(next_cache)}")
                    if len(next_cache) > 0:
                        first_item = next_cache[0]
                        print(f"[LlamaModel] next_cache[0] type: {type(first_item).__name__}")
                        if isinstance(first_item, tuple) and len(first_item) >= 2:
                            print(f"[LlamaModel] next_cache[0][0] (key) type: {type(first_item[0]).__name__}, shape: {first_item[0].shape if hasattr(first_item[0], 'shape') else 'N/A'}")
                elif hasattr(next_cache, 'key_cache'):
                    print(f"[LlamaModel] DynamicCache detected, key_cache length: {len(next_cache.key_cache)}")
            
            # Extract last layer KV using helper method
            last_layer_kv = self._extract_last_layer_kv(next_cache, actual_seq_len)
            
            if last_layer_kv is not None:
                # Compute task embedding
                task_embedding = self.kv_manager._compute_task_embedding(inputs_embeds)
                
                # Add to cache
                add_success = self.kv_manager.add_task(
                    task_id=effective_task_id,
                    task_embedding=task_embedding,
                    top_layer_kv=last_layer_kv
                )
                if add_success:
                    print(f"[LlamaModel] ✅ Auto-saved KV for task {effective_task_id} (kv_seq_len={last_layer_kv[0].shape[2]}, input_seq_len={actual_seq_len})")
                else:
                    print(f"[LlamaModel] ⚠️ Failed to auto-save KV for task {effective_task_id}")
            else:
                logger.warning(f"[LlamaModel] ⚠️ Auto-save failed: Could not extract valid last_layer_kv. next_cache type={type(next_cache).__name__}, seq_len={actual_seq_len}")

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLMWithKVReuse(_LlamaForCausalLM):
    """
    Llama Causal LM with Inter-Task KV Reuse
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelWithKVReuse(config)
        self.kv_manager = None
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def set_kv_manager(self, kv_manager: InterTaskKVManager):
        """Set the global KV manager"""
        self.kv_manager = kv_manager
        self.model.set_kv_manager(kv_manager)
    
    def get_kv_manager(self) -> Optional[InterTaskKVManager]:
        """Get the KV manager"""
        return self.kv_manager
    
    def set_current_task_id(self, task_id: Optional[str]):
        """Set the current task ID for auto-save during prefill"""
        self.model.set_current_task_id(task_id)
    
    def clear_current_task_id(self):
        """Clear the current task ID"""
        self.model.clear_current_task_id()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        task_id: Optional[str] = None,
        reuse_kv: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with optional KV reuse
        
        Args:
            task_id: Unique identifier for the current task
            reuse_kv: Whether to attempt KV reuse for this forward pass
            cache_position: Position indices for cache (transformers 4.36+)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            task_id=task_id,
            reuse_kv=reuse_kv,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def generate_with_kv_reuse(
        self,
        input_ids: torch.LongTensor,
        task_id: Optional[str] = None,
        **generate_kwargs
    ):
        """
        Generate with KV reuse enabled
        
        Args:
            input_ids: Input token IDs
            task_id: Unique task identifier (will be auto-generated if None)
            **generate_kwargs: Additional arguments for generate()
        """
        # Auto-generate task_id if not provided
        if task_id is None and self.kv_manager is not None:
            # Use input_ids as a simple hash
            task_id = self.kv_manager.generate_task_id(str(input_ids.tolist()))
        
        # Enable KV reuse for the prefill phase
        self.model.enable_kv_reuse_mode()
        
        # Override forward to pass task_id and reuse_kv flag
        original_forward = self.forward
        
        def forward_with_reuse(*args, **kwargs):
            kwargs['task_id'] = task_id
            kwargs['reuse_kv'] = True
            return original_forward(*args, **kwargs)
        
        self.forward = forward_with_reuse
        
        try:
            outputs = self.generate(input_ids, **generate_kwargs)
        finally:
            # Restore original forward
            self.forward = original_forward
        
        return outputs
