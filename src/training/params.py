from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=True)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = 2e-6
    merger_lr: Optional[float] = 1e-5
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger: bool = True

    # # TrainingArguments inherit arguments
    # remove_unused_columns: Optional[bool] = field(default=False)
    # bf16: bool = field(default=True)
    # fp16: bool = field(default=False)
    # output_dir: str = field(default="output/testing_lora")
    # num_train_epochs: float = field(default=1)
    # per_device_train_batch_size: int = field(default=1)
    # gradient_accumulation_steps: int = field(default=1)
    # learning_rate: float = field(default=1e-4)
    # weight_decay: float = field(default=0.1)
    # warmup_ratio: float = field(default=0.03)
    # lr_scheduler_type: str = field(default="cosine")
    # logging_steps: float = field(default=1)
    # tf32: Optional[bool] = field(default=True)
    # gradient_checkpointing: bool = field(default=True)
    # report_to: str = field(default="wandb")
    # save_strategy: str = field(default="steps")
    # save_steps: float = field(default=200)
    # save_total_limit: Optional[int] = field(default=10)
    # dataloader_num_workers: int = field(default=12)
    # # deepspeed: str = field(default="scripts/zero2.json")
    


@dataclass
class DataArguments:
    data_path: str = field(
        default="data/qwen_data.json"
    )
    lazy_preprocess: bool = True
    image_folder: Optional[str] = field(default='./')
    image_min_pixels: Optional[int] = field(default=((512 * 28 * 28))) # 3136)
    image_max_pixels: Optional[int] = field(default=((1280 * 28 * 28))) #12845056
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    fps: float = 1.0