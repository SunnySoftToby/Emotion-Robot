import torch
from transformers import AutoModelForCausalLM

# 原始模型和 LoRA 權重的位置
original_model_path = "../model_weight/TAIDE-LX-7B-Chat"
finetuned_model_path = "../model_weight/TAIDE-LX-7B-Chat-finetuned"

# 載入原始模型和 LoRA 權重
original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)

# 獲取模型各層名稱
original_state_dict = original_model.state_dict()
finetuned_state_dict = finetuned_model.state_dict()

# 只更新原始模型中存在的参数(合併權重)
for name in original_state_dict:
    if name in finetuned_state_dict:
        # 確保模型層 size 相同
        if original_state_dict[name].shape == finetuned_state_dict[name].shape:
            original_state_dict[name] = finetuned_state_dict[name]
        else:
            print(f"Skipping {name} due to shape mismatch.")


# 合併保存權重
original_model.load_state_dict(original_state_dict)
original_model.save_pretrained("./model_weight/TAIDE-LX-7B-Chat-merged")
