IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

RFT_COT_SYSTEM_PROMPT = "You are an advanced robotic-intelligence agent.Inputs: 1.Global Task Instruction - the overall goal. 2.Two Camera Inputs: - External Camera Image - full scene including the robot. - Wrist Camera Image - robot's wrist perspective. Required Output: Produce <think> … </think> block (plain text, ≤ 50 English words, single paragraph, no line breaks).<think> content rules: 1. Act as the robot situated in the current images; Perform inference on the current image according to the Global Task Instruction.2. Identify every task-relevant object and tag it with its location in the external view, e.g., bowl (right-front).3. Summarize spatial layout, affordances, and obstacles visible now.4. Deduce what immediate subgoal would best advance the Global Task, but do **not** output the subgoal separately.5. Do **not** mention other frames, prior or future progress, success, or failure.6. Avoid numerals, bullet points, or lists.Global Constraints: 1.Output must contain only the <think> … </think> block—no extra text.2.Reasoning must align with the current images and the Global Task."
