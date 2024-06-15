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
    emotion = "哀傷"
    history_text = "\n".join([f"使用者: {usr_msg}\n輔導師: {resp}" for usr_msg, resp in chat_history][-5:])

    sys = (f"盡量不要生成超過二十個字，全程都要使用繁體中文回答。"
           f"你是一位專業的情感輔導師，並負責心理諮商，"
           f"根據外在表現，我們判斷當下使用者的情緒為: {emotion}，"
           f"但你要假裝不知道，直到使用者從文字表現出來"
           f"以下{'{}'}內的文字為你和使用者的歷史對話紀錄"
           f"{history_text}")
    # 構建完整的聊天歷史作為提示
    prompt = f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\nHuman: {message} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=2048)
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    response_start = full_response.rfind("[/INST]") + len("[/INST]")
    bot_message = full_response[response_start:].strip()

    # 添加到聊天歷史中
    chat_history.append((message, bot_message))
    # print("----------------------")
    # print(full_response)
    # print("----------------------")
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