import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./model_weight/TAIDE-LX-7B-Chat", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("./model_weight/TAIDE-LX-7B-Chat", torch_dtype=torch.float16)
model.to(device)

def respond(message, chat_history):
    sys = "你是一位情感輔導師,開導使用者並給予正向鼓勵和引導"
    prompt = f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{message} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=2048)
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    response_start = full_response.find("[/INST]") + len("[/INST]")
    bot_message = full_response[response_start:].strip()  
    
    # Appending to chat history
    chat_history.append((message, bot_message))
    return None,chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot],value="清除聊天紀錄")
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
