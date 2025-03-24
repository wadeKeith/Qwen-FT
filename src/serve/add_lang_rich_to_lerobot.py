from pathlib import Path
import jsonlines
import os
import sys
sys.path.append('./')
from PIL import Image
from src.serve.get_instruction_qwen import get_new_instruction
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from tqdm import tqdm

repo_id = "physical-intelligence/libero"
root_dir = "/home/yin/Documents/Github/openpi_data/completed"
max_new_tokens = 5120
temperature = 0
repetition_penalty = 1.0
model_path = "checkpoints/3b/libero/checkpoint-1320"
disable_flash_attention = False
model_base = "Qwen/Qwen2-VL-7B-Instruct"
device = 'cuda'
load_4bit = False
load_8bit = False

images_dir = Path(root_dir) / repo_id / "images/observation.images.image"
wrist_images_dir = Path(root_dir) / repo_id / "images/observation.images.wrist_image"
instructions_dir = Path(root_dir) / repo_id / "meta"
episode_task_save_dir = Path(root_dir) / repo_id / "meta/episodes_new.jsonl"
task_save_dir = Path(root_dir) / repo_id / "meta/tasks_new.jsonl"


with jsonlines.open(instructions_dir / "episodes.jsonl", "r") as reader:
    episode_instructions = list(reader)
task_idex_template = lambda idex, instruction: {"task_index": idex, "task": instruction}
task_json_file = []

image_episodes_dir = os.listdir(images_dir)
wrist_image_episodes_dir = os.listdir(wrist_images_dir)

assert len(image_episodes_dir) == len(wrist_image_episodes_dir) == len(episode_instructions)

generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
        "repetition_penalty": repetition_penalty,
    }

disable_torch_init()

use_flash_attn = True

model_name = get_model_name_from_path(model_path)

if disable_flash_attention:
    use_flash_attn = False

processor, model = load_pretrained_model(model_base = model_base, model_path = model_path, 
                                            device_map=device, model_name=model_name, 
                                            load_4bit=load_4bit, load_8bit=load_8bit,
                                            device=device, use_flash_attn=use_flash_attn
)

for i, (image_episode, wrist_image_episode, episode_instruction) in tqdm(enumerate(zip(image_episodes_dir, wrist_image_episodes_dir, episode_instructions))):
    images = [Image.open(images_dir / image_episode/"frame_000000.png"), Image.open(images_dir / wrist_image_episode/"frame_000000.png")]
    instruction = episode_instruction['tasks'][0]
    new_instruction = get_new_instruction(processor= processor, 
                                          model = model, 
                                          generation_args = generation_args, 
                                          old_instruction = instruction, 
                                          images = images, 
                                          device = 'cuda')
    episode_instructions[i]['tasks'][0] = new_instruction
    task_json_file.append(task_idex_template(i, new_instruction))

episode_task_save_dir.parent.mkdir(exist_ok=True, parents=True)
task_save_dir.parent.mkdir(exist_ok=True, parents=True)
with jsonlines.open(episode_task_save_dir, "a") as writer:
    writer.write_all(episode_instructions)
with jsonlines.open(task_save_dir, "a") as writer:
    writer.write_all(task_json_file)