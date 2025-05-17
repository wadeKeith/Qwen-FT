from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments
from trl import DPOConfig as DPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")


@dataclass
class TrainingArguments(HFTrainingArguments):
    ##########################################################################################
    # Dataloader parameters
    dataloader_num_workers: int = field(default=32)
    """Number of workers for data loading."""

    dataloader_pin_memory: bool = field(default=True)
    """Whether to pin memory for data loading."""

    dataloader_persistent_workers: bool = field(default=True)
    """Whether to use persistent workers for data loading."""

    ##########################################################################################
    # Training parameters
    per_device_train_batch_size: int = field(default=1)
    """Batch size per GPU for training."""

    auto_find_batch_size: bool = field(default=False)
    """Whether to automatically find the batch size."""

    # max_steps: int = field(default=3)
    """Total number of training steps to perform. If > 0, overrides num_train_epochs."""

    num_train_epochs: int = field(default=1)
    """Total number of training epochs to perform."""

    deepspeed: str = field(default="scripts/zero3_offload.json")
    """Path to deepspeed config file."""

    gradient_checkpointing: bool = field(default=False)
    """Whether to use gradient checkpointing. Save memory but slower training."""

    bf16: bool = field(default=True)
    """Whether to use bf16. Requires PyTorch >= 1.10 and NVIDIA A100 GPUs."""

    fp16: bool = field(default=False)
    """Whether to use fp16. Requires NVIDIA apex or PyTorch >= 1.6 and NVIDIA GPUs. when using bf16, set to False"""

    tf32: bool = field(default=True)
    """Whether to use tf32. Requires PyTorch >= 1.6 and NVIDIA A100 GPUs."""

    gradient_accumulation_steps: int = field(default=1)
    """Number of updates steps to accumulate before performing a backward/update pass."""

    seed: int = field(default=429)
    """Random seed for reproducibility."""

    max_grad_norm: float = field(default=1.0)
    """Max gradient norm. Used to clip gradients."""

    do_train: bool = field(default=True)
    """Whether to run training."""

    disable_flash_attn2: bool = field(default=False)
    """Whether to disable flash attention 2.0. If True, will use sdpa instead of flash attention 2.0."""

    ##########################################################################################
    # Save parameters
    save_strategy: str = field(default="steps")
    """The checkpoint save strategy to use."""

    # save_steps: float = field(default=2)
    """Number of updates steps before two checkpoint saves. If save_strategy is 'epoch', this is the number of epochs."""

    resume: bool = field(default=False)
    """Whether to resume training from the last checkpoint."""

    save_total_limit: Optional[int] = field(default=40)
    """Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir."""

    save_safetensors : bool = field(default=True)
    """Whether to save the model as a safetensors file. If False, saves as a pytorch file."""

    ##########################################################################################
    # Optimizer parameters
    learning_rate: float = field(default=1e-5)
    """Learning rate for training."""

    vision_lr: Optional[float] = field(default=2e-6)
    """Learning rate for vision tower. If None, will use the same learning rate as the rest of the model."""

    merger_lr: Optional[float] = field(default=1e-5)
    """Learning rate for merger. If None, will use the same learning rate as the rest of the model."""

    weight_decay: float = field(default=0.1)
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = field(default=0.03)
    """Ratio of total training steps to perform learning rate warmup for. See the documentation for more details."""

    # warmup_steps: int = field(default=1000)
    """Number of steps to perform learning rate warmup for. If warmup_ratio is set, this will be ignored."""

    optim: str = field(default="adamw_torch")
    """The optimizer to use. See the documentation for more details."""

    adam_beta1: float = field(default=0.9)
    """Beta1 for AdamW optimizer."""

    adam_beta2: float = field(default=0.999)
    """Beta2 for AdamW optimizer."""

    adam_epsilon: float = field(default=1e-8)
    """Epsilon for AdamW optimizer."""

    lr_scheduler_type: str = field(default="cosine")
    """The scheduler to use. See the documentation for more details."""

    ##########################################################################################
    # Logging parameters
    output_dir: str = field(default="./checkpoints/libero_cot")
    """Directory to save model checkpoints."""
    
    report_to: str = "wandb"
    """The integration to report the results and logs to. Supported platforms are `"tensorboard"` and `"wandb"`."""
    
    run_name: str = field(default="7b-libero-cot")
    """The name of the run. Used for logging and saving checkpoints."""

    logging_steps: int = field(default=1)
    """Number of updates steps before logging training metrics."""

    log_level: str = field(default="info") 
    """Logging level to use. 'debug', training recommend :'info'. 'warning', 'error' and 'critical', plus a 'passive' """

    ##########################################################################################
    # LORA arguments
    lora_enable: bool = False
    """Whether to use LoRA for training. If True, will use LoRA for training."""

    use_dora: bool = False
    """Option for using DoRA instead of LoRA. `lora_enable` should be `True` to use this option."""

    vision_lora: bool = False
    """Whether to use LoRA for vision tower. If True, will use LoRA for vision tower."""

    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    """List of namespan to exclude for LoRA. If None, will not exclude any namespan."""

    lora_weight_path: str = field(default="")
    """Path to LoRA weight file. If "", will not use LoRA weight file."""

    lora_rank: int = field(default=64)
    """Rank for LoRA. The higher the rank, the more parameters will be trained. If 0, will not use LoRA."""

    lora_alpha: int = field(default=16)
    """Alpha for LoRA. The higher the alpha, the more parameters will be trained. If 0, will not use LoRA."""

    lora_dropout: float = field(default=0.05)
    """Dropout for LoRA. The higher the dropout, the more parameters will be trained. If 0, will not use LoRA."""

    lora_bias: str = field(default="none")
    """Bias for LoRA. The higher the bias, the more parameters will be trained. If 0, will not use LoRA."""

    num_lora_modules: int = field(default=-1)
    """Number of LoRA modules to use. If -1, will use all LoRA modules. If 0, will not use LoRA."""

    freeze_llm: bool = field(default=False)
    """Whether to freeze the LLM. If True, will freeze the LLM."""

    freeze_vision_tower: bool = field(default=False)
    """Whether to freeze the vision tower. If True, will freeze the vision tower."""

    freeze_merger: bool = field(default=False)
    """Whether to freeze the merger. If True, will freeze the merger."""

    ##########################################################################################
    # Quantization parameters
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    """How many bits to use. 16, 8, or 4. If 16, will use fp16. If 8, will use bnb quantization. If 4, will use bnb quantization."""

    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    """Compress the quantization statistics through double quantization. If True, will use double quantization."""

    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    """Quantization data type to use. Should be one of `fp4` or `nf4`. If None, will use the default quantization type."""

    ##########################################################################################
    # Other parameters
    remove_unused_columns: bool = field(default=False)
    """Whether to remove unused columns from the dataset. If False, will keep all columns."""

    ##########################################################################################
    # Addition parameters
    cache_dir: Optional[str] = field(default=None)
    """Path to cache directory. If None, will use the default cache directory."""

    use_liger: bool = field(default=True)
    """Whether to use Liger for training. If True, will use Liger for training."""

    max_seq_length: int = field(default=32768) # This is the default value of the qwen2-vl model
    "Maximum sequence length. Sequences will be right padded (and possibly truncated)."



@dataclass
class DPOArguments(DPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
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
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger: bool = True
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta value for DPO."}
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute the reference log probabilities."}
    )
    dpo_loss:str = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="data/datasets/libero_cot.json", metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default="./")
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0