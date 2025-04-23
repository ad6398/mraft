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

dataset_name = colpali_config["dataset-name"]

train_ds = cast(DatasetDict, load_dataset(dataset_name))
# for time being
ds = train_ds.rename_column("page_image", "image")
ds["train"] = ds["train"].shuffle(seed=42)
print(train_ds)


checkpoints_dir = Path(colpali_config["output-dir"])
checkpoints_dir.mkdir(exist_ok=True, parents=True)

training_args = TrainingArguments(
    output_dir=str(checkpoints_dir),
    # hub_model_id=hf_pushed_model_name if hf_pushed_model_name else None,
    overwrite_output_dir=True,
    num_train_epochs=0.2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    eval_strategy="steps",
    save_steps=200,
    logging_steps=20,
    eval_steps=200,
    warmup_steps=100,
    learning_rate=5e-5,
    save_total_limit=1,
    # report_to=["wandb"] if wandb_experiment_name else [], TODO(AD)
)


class EvaluateFirstStepCallback(TrainerCallback):
    """
    Run eval after the first training step.
    Used to have a more precise evaluation learning curve.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


trainer = ContrastiveTrainer(
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    args=training_args,
    data_collator=collator,
    loss_func=ColbertPairwiseCELoss(),
    is_vision_model=True,
)

trainer.args.remove_unused_columns = False
trainer.add_callback(EvaluateFirstStepCallback())

train_results = trainer.train()

print("train results", train_results)

eval_results = trainer.evaluate()


print("eval results", eval_results)


