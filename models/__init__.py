import os

from .configuration_llama import OptLlamaConfig
from .modeling_llama_opt import LlamaForCausalLM as OptLlamaForCausalLM
from .modeling_llama_lsh import LlamaForCausalLM as LshLlamaForCausalLM

# Inter-Task KV Reuse (Sim-LLM)
from .inter_task_kv_manager import InterTaskKVManager, LSHIndex, TaskCacheEntry
from .modeling_llama_inter_task_kv import (
    LlamaModelWithKVReuse,
    LlamaForCausalLMWithKVReuse
)
from .sim_llm_inference import SimLLMInference, create_sim_llm_inference

# Optional imports for modules that may not exist
try:
    from .wandb_callback import WandbCallback
except ImportError:
    WandbCallback = None

try:
    from .modeling_llama_cla import LlamaForCausalLM as ClaLlamaForCausalLM
    from .configuration_llama import ClaLlamaConfig
    _has_cla = True
except ImportError:
    ClaLlamaForCausalLM = None
    ClaLlamaConfig = None
    _has_cla = False

try:
    from .modeling_llama_opt_group import LlamaForCausalLM as GroupOptLlamaForCausalLM
    from .configuration_llama import GroupOptLlamaConfig
    _has_group_opt = True
except ImportError:
    GroupOptLlamaForCausalLM = None
    GroupOptLlamaConfig = None
    _has_group_opt = False


from transformers import AutoConfig, AutoModelForCausalLM

# Register OptLlama
AutoConfig.register("opt-llama", OptLlamaConfig)
AutoModelForCausalLM.register(OptLlamaConfig, OptLlamaForCausalLM)

# Register LshLlama (uses same config as OptLlama)
# Note: LshLlamaForCausalLM uses OptLlamaConfig

# Register ClaLlama if available
if _has_cla:
    AutoConfig.register("cla-llama", ClaLlamaConfig)
    AutoModelForCausalLM.register(ClaLlamaConfig, ClaLlamaForCausalLM)

# Register GroupOptLlama if available
if _has_group_opt:
    AutoConfig.register("group-opt-llama", GroupOptLlamaConfig)
    AutoModelForCausalLM.register(GroupOptLlamaConfig, GroupOptLlamaForCausalLM)


# Fused operations (optional, requires flash_attn)
if os.environ.get('LCKV_FUSED_RMSNORM', False):
    try:
        import transformers
        from flash_attn.ops.rms_norm import RMSNorm
        transformers.models.llama.modeling_llama.LlamaRMSNorm = RMSNorm
        from . import modeling_llama_opt
        modeling_llama_opt.LlamaRMSNorm = RMSNorm
        from . import modeling_llama_lsh
        modeling_llama_lsh.LlamaRMSNorm = RMSNorm
    except ImportError:
        pass
