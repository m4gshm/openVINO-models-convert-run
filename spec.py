import openvino_genai as ov_genai

target_model_dir = "./models/Qwen2.5-Coder-7B-Instruct-int4-sym-g128-r1/1"
draft_model_dir = "./models/Qwen2.5-Coder-1.5B-Instruct-int4-sym-g128-r1/1"

# 1. Создаем ModelDesc с помощью функции draft_model (с маленькой буквы)
# Именно этот объект ожидает конструктор LLMPipeline
draft = ov_genai.draft_model(draft_model_dir, device="GPU")

# 2. Передаем объект draft в основной пайплайн
pipe = ov_genai.LLMPipeline(
    target_model_dir, 
    device="GPU", 
    draft_model=draft
)

config = ov_genai.GenerationConfig()
config.max_new_tokens = 100
config.num_assistant_tokens = 5

print(pipe.generate("def hello_world():", config))