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
        # INTER-TASK KV REUSE LOGIC (moved before past_key_values_length calculation)
        # ============================================================
        matched_entry = None
        skip_layers = False
        cached_kv_seq_len = 0
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if self.enable_kv_reuse and reuse_kv and self.kv_manager is not None and past_key_values is None:
            # This is a prefill phase - check for KV reuse opportunity
            
            # Compute task embedding from input embeddings
            task_embedding = self.kv_manager._compute_task_embedding(inputs_embeds)
            
            # Search for similar task
            matched_entry = self.kv_manager.search_similar_task(task_embedding)
            
            if matched_entry is not None:
                # Cache HIT - prepare to reuse KV from last layer
                skip_layers = True
                
                # Get cached KV sequence length
                cached_key, cached_value = matched_entry.top_layer_kv
                cached_kv_seq_len = cached_key.shape[2]  # (batch, num_heads, seq_len, head_dim)
                
                # Construct past_key_values with only the last layer populated
                num_layers = len(self.layers)
                past_key_values = []
                
                for layer_idx in range(num_layers):
                    if layer_idx == num_layers - 1:
                        # Last layer: use cached KV
                        past_key_values.append((cached_key, cached_value))
                    else:
                        # Other layers: None (will be skipped)
                        past_key_values.append(None)
                
                print(f"[LlamaModel] Using cached KV from task {matched_entry.task_id}, seq_len={cached_kv_seq_len}")

        # Calculate past_key_values_length
        past_key_values_length = 0
        if past_key_values is not None:
            # Find the first non-None layer to get the sequence length
            for layer_kv in past_key_values:
                if layer_kv is not None:
                    past_key_values_length = layer_kv[0].shape[2]
                    break

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
        # STANDARD FORWARD PASS (with potential layer skipping)
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

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Skip layers if we're reusing KV (except last layer)
            if skip_layers and idx < len(self.layers) - 1:
                # For skipped layers, just pass through hidden states
                if use_cache:
                    next_decoder_cache += (None,)
                continue

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
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        
        # ============================================================
        # SAVE TO CACHE (if this was a cache miss)
        # ============================================================
        if self.enable_kv_reuse and reuse_kv and self.kv_manager is not None and matched_entry is None:
            # Cache MISS - save the KV from last layer
            if use_cache and next_cache is not None and len(next_cache) > 0:
                last_layer_kv = next_cache[-1]  # (key, value) from last layer
                
                if last_layer_kv is not None and task_id is not None:
                    # Compute task embedding
                    task_embedding = self.kv_manager._compute_task_embedding(inputs_embeds)
                    
                    # Add to cache
                    self.kv_manager.add_task(
                        task_id=task_id,
                        task_embedding=task_embedding,
                        top_layer_kv=last_layer_kv
                    )
                    print(f"[LlamaModel] Saved KV for task {task_id}")

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
