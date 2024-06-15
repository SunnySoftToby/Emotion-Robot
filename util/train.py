import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import json
import os

# 添加 CUDA 库路径
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载多轮对话数据
with open('../EXTES/ExTES_translated_part1.json', 'r', encoding='utf-8') as f:
    dialogues = json.load(f)

# 将数据转换为 HuggingFace 数据集格式
contexts = []
responses = []
for dialogue in dialogues:
    context = ""
    for turn in dialogue['content'][:-1]:  # 忽略最后一个轮次
        if 'User' in turn:
            context += f"Human: {turn['User']}\n"
        elif 'AI' in turn:
            context += f"AI: {turn['AI']}\n"
    contexts.append(context.strip())

    # 检查对话是否包含最后一个 AI 的响应
    if 'AI' in dialogue['content'][-1]:
        responses.append("AI: " + dialogue['content'][-1]['AI'])
    else:
        responses.append("AI: [No response]")  # 添加一个默认的响应以防止键错误

data = {
    'context': contexts,
    'response': responses
}
dataset = Dataset.from_dict(data)

# 分割数据集
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model_weight/TAIDE-LX-7B-Chat", use_fast=False)

# 数据预处理函数
def preprocess_function(examples):
    inputs = examples['context']
    targets = examples['response']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(text_target=targets, max_length=512, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# 应用预处理函数
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 创建 BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# 加载模型，使用量化配置
model = AutoModelForCausalLM.from_pretrained(
    "./model_weight/TAIDE-LX-7B-Chat",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map="auto"
)

# 定义 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 指定任务类型
    inference_mode=False,  # 训练模式
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"] # 指定要应用 LoRA 的模块
)

# 将模型转换为 LoRA 模型
model = get_peft_model(model, lora_config)
model.to(device)


output_dir = os.path.join('..', 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 創建目錄


# 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,  # 批量大小
    per_device_eval_batch_size=1,  # 批量大小
    num_train_epochs=50,
    weight_decay=0.01,
    save_total_limit=1,
    save_strategy="no",  # 訓練完再存
    fp16=True,  # 启用混合精度训练
)

# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("./model_weight/TAIDE-LX-7B-Chat-finetuned")
tokenizer.save_pretrained("./model_weight/TAIDE-LX-7B-Chat-finetuned")
