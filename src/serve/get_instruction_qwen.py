from qwen_vl_utils import process_vision_info

def get_new_instruction(processor, model, generation_args, old_instruction, images, device = 'cuda'):

    conversation = []
    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image})
    user_text = old_instruction
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    conversation.append({"role": "user", "content": user_content})

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device) 
    generation_kwargs = dict(inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    output = model.generate(**generation_kwargs)
    new_instruction = processor.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).split('\nassistant\n')[-1].strip()
    return new_instruction