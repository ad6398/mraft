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


