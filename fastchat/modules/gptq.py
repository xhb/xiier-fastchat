import sys
from pathlib import Path
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from dataclasses import dataclass, field
from transformers import AutoTokenizer

try:
    sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))
    from modelutils import find_layers
    from quant import make_quant
except ImportError:
    print("Error: Failed to load GPTQ-for-LLaMa")
    sys.exit(-1)


def load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head'], kernel_switch_threshold=128):
    config = AutoConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize, faster=faster_kernel, kernel_switch_threshold=kernel_switch_threshold)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


@dataclass
class GptqConfig:
    ckpt: str = field(
        default=None,
        metadata={
            "help": "Load quantized model. The path to the local GPTQ checkpoint."
        },
    )
    wbits: int = field(default=16, metadata={"help": "#bits to use for quantization"})
    groupsize: int = field(
        default=-1,
        metadata={"help": "Groupsize to use for quantization; default uses full row."},
    )
    act_order: bool = field(
        default=True,
        metadata={"help": "Whether to apply the activation order GPTQ heuristic"},
    )


def load_gptq_quantized(model_name, gptq_config: GptqConfig, device):
    print("Loading GPTQ quantized model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    wbits = gptq_config.wbits
    groupsize = gptq_config.groupsize
    threshold = 128

    model_name = model_name.replace('/', '_')
    path_to_model = Path(f'./models/{model_name}')
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None

    if len(found_pts) == 1:
        pt_path = found_pts[0]
    elif len(found_safetensors) == 1:
        pt_path = found_safetensors[0]

    if not pt_path:
        print("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()

    model = load_quant(str(path_to_model), str(pt_path), wbits, groupsize, kernel_switch_threshold=threshold)
    model.to(device)

    return model, tokenizer


