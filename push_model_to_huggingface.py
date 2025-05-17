import sys
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import torch


model_path = "checkpoints/7b/libero_cot/checkpoint-460"

# push base model to hub
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

processor = AutoProcessor.from_pretrained(
    model_path, use_fast=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,

)
# model.push_to_hub(
#     repo_id="yinchenghust/nora_libero_base",
#     commit_message="New model to hub",
#     tags = ["model", "pi0fast_base"],
#     token = "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",
#     private=False,
# )

# push trained model to hub
push_hf_name = "yinchenghust/qwen_2_5_libero_cot_7b"
model.push_to_hub(
    repo_id=push_hf_name,
    commit_message="Push model to hub",
    token = "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",
    private=False,
)

processor.push_to_hub(
    repo_id=push_hf_name,
    commit_message="Push model to hub",
    token = "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",
    private=False,
)

tokenizer.push_to_hub(
    repo_id=push_hf_name,
    commit_message="Push model to hub",
    token = "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",
    private=False,
)


