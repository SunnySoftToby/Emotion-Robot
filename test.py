import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import gradio as gr
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./model_weight/TAIDE-LX-7B-Chat-finetuned", use_fast=False)
# 創建 BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
#加油 3q
# 加載模型，使用量化配置
# model = AutoModelForCausalLM.from_pretrained(
#     "./model_weight/TAIDE-LX-7B-Chat-merged",
#     torch_dtype=torch.float16,
#     quantization_config=quantization_config,
#     device_map="auto"
# )
model = AutoModelForCausalLM.from_pretrained("./model_weight/TAIDE-LX-7B-Chat-merged", torch_dtype=torch.float16)
model.to(device)


def respond(message, chat_history):
    emotion = "極度哀傷"
    sys = f"你是一位情感輔導師，對用戶進行心理諮商，聊天過程中要保持正向和積極。切記，回應要盡量簡潔，盡量不要超過三句話，全程都要使用繁體中文回答。根據使用者的綜合表現，我們判斷當下使用者情緒為: {emotion}，但使用者目前不知道這個訊息，請你不要透漏你已知道"

    chat = [
        {"role": "system", "content": sys},
        {"role": "user", "content": message}
    ]

    # Extend chat history with the new user message and prepare the chat template
    if chat_history:
        for msg, resp in chat_history[-5:]:
            chat.extend([
                {"role": "user", "content": msg},
                {"role": "assistant", "content": resp}
            ])

    prompt = tokenizer.apply_chat_template(chat)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=8192, temperature=0.7, top_k=50, do_sample=True)
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract and save the generated message
    bot_message = full_response.strip()
    chat_history.append((message, bot_message))
    return "", chat_history


def regenerate(chat_history):
    if not chat_history:
        return "", chat_history

    last_message = chat_history[-1][0]  # 獲取最後一條人類消息
    # 移除最後一條對話
    chat_history.pop()
    print(last_message)
    return respond(last_message, chat_history)  # 重新生成回應


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot], value="清除聊天紀錄")
    regenerate_btn = gr.Button("重新生成")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    regenerate_btn.click(regenerate, [chatbot], [msg, chatbot])

demo.launch()