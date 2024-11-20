from pathlib import Path
from typing import cast

import torch
from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.loss import ColbertPairwiseCELoss
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from torch import nn
from transformers import BitsAndBytesConfig, TrainerCallback, TrainingArguments

from config import get_colpali_config

colpali_config = get_colpali_config()

QUANTIZATION_STRATEGY = colpali_config["quantization-strategy"]

def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param}"
    )


device = get_torch_device("auto")

if QUANTIZATION_STRATEGY and device != "cuda:0":
    raise ValueError("This notebook requires a CUDA GPU to use quantization.")

# Prepare quantization config
if QUANTIZATION_STRATEGY is None:
    bnb_config = None
elif QUANTIZATION_STRATEGY == "8bit":
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
elif QUANTIZATION_STRATEGY == "4bit":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    raise ValueError(f"Invalid quantization strategy: {QUANTIZATION_STRATEGY}")


MODEL_NAME = colpali_config["colpali-model-name"] 

lora_config = LoraConfig.from_pretrained(MODEL_NAME)


model = cast(
    ColPali,
    ColPali.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ),
)

if not model.active_adapters():
    raise ValueError("No adapter found in the model.")

# The LoRA weights are frozen by default. We need to unfreeze them to fine-tune the model.
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

print_trainable_parameters(model)

if lora_config.base_model_name_or_path is None:
    raise ValueError("Base model name or path is required in the LoRA config.")

# load cloapli processor and name
processor = cast(
    ColPaliProcessor,
    ColPaliProcessor.from_pretrained(MODEL_NAME),
)
collator = VisualRetrieverCollator(processor=processor)

