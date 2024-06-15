import torch
from transformers import AutoModelForCausalLM

# 路径到原始和微调后的模型
original_model_path = "../model_weight/TAIDE-LX-7B-Chat"
finetuned_model_path = "../model_weight/TAIDE-LX-7B-Chat-finetuned"

# 加载模型
original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)

# 使用 state_dict 来获取模型参数和名称
original_state_dict = original_model.state_dict()
finetuned_state_dict = finetuned_model.state_dict()

# 只更新原始模型中存在的参数
for name in original_state_dict:
    if name in finetuned_state_dict:
        # 确保只复制形状相同的参数
        if original_state_dict[name].shape == finetuned_state_dict[name].shape:
            original_state_dict[name] = finetuned_state_dict[name]
        else:
            print(f"Skipping {name} due to shape mismatch.")

# 加载更新后的 state_dict 到原始模型
original_model.load_state_dict(original_state_dict)

# 保存合并后的模型
original_model.save_pretrained("./model_weight/TAIDE-LX-7B-Chat-merged")
