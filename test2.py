import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import gradio as gr
import os
import csv

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


def respond(message, chat_history, feedback=None):

    emotion = "哀傷"
    history_text = "\n".join([f"使用者: {usr_msg}\n輔導師: {resp}" for usr_msg, resp in chat_history][-5:])

    sys = (f"回答不要超過二十個字，全程都要使用繁體中文回答。"
           f"你是一位專業的情感輔導師，正在提供心理諮詢服務。你的主要工作包括聆聽使用者分享他們的感情和生活困擾，分析他們的心理狀態，並根據每位使用者的具體情況提供個性化的建議和策略。你專注於幫助人們改善人際關係、處理情緒困擾，並提升他們的整體心理健康。你的目標是創造一個支持和理解的環境，讓使用者能夠自我探索和成長。"
           f"根據外在表現，我們判斷當下使用者的情緒為: {emotion}，"
           f"以下{'{}'}內的文字為你和使用者的對話紀錄，你可以依此來得知使用者資訊"
           f"{history_text}")

    # 重新生成回覆
    if feedback is not None:
        print("1")
        last_ai_message = ""
        if chat_history:
            last_ai_message = chat_history[-1][1]
        print("this is last ai msg")
        print(last_ai_message)
        rlhf_prompt = f'使用者對於你剛剛回復的答案({last_ai_message})不滿意，請你繼續保持以下說明並重新生成'
        sys = f"{rlhf_prompt}\n{sys}"
        chat_history.pop()

    prompt = f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\nHuman: {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=2048, temperature=0.5, top_k=20, do_sample=True)
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    response_start = full_response.rfind("[/INST]") + len("[/INST]")
    bot_message = full_response[response_start:].strip()

    chat_history.append((message, bot_message))
    return chat_history


def handle_feedback(feedback, chat_history):
    feedback_file = "feedback_data.csv"
    file_exists = os.path.isfile(feedback_file)
    with open(feedback_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Message", "Response", "Feedback"])

        if chat_history:
            history_to_record = chat_history[:-1] if len(chat_history) > 1 else []
            writer.writerow([history_to_record, chat_history[-1], feedback])

    if feedback in ["不滿意", "非常不滿意"]:
        last_human_message = chat_history[-1][0]
        return respond(last_human_message, chat_history, feedback)
    return chat_history


with gr.Blocks() as demo:
    with gr.Row():
        msg = gr.Textbox(placeholder="輸入您的訊息...")
        submit_btn = gr.Button("送出訊息")
    chatbot = gr.Chatbot()

    with gr.Column():
        with gr.Row():
            feedback_menu = gr.Dropdown(choices=['非常滿意', '滿意', '一般', '不滿意', '非常不滿意'],
                                        label="您對這個回答滿意嗎？",
                                        value="一般")
            feedback_btn = gr.Button("送出回饋")

    msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot])
    submit_btn.click(respond, inputs=[msg, chatbot], outputs=[chatbot])
    feedback_btn.click(handle_feedback, inputs=[feedback_menu, chatbot], outputs=[chatbot])

demo.launch()
