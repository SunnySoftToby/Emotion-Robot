import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import os


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


class Translator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("./model_weight/TAIDE-LX-7B-Chat", use_fast=False, padding_side='left')
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained("./model_weight/TAIDE-LX-7B-Chat", torch_dtype=torch.float16)
        self.model.to(self.device)
        self.sys = "將英文句子翻譯成繁體中文，返回翻譯結果就可以，不要進行任何解釋"

    def translate(self, message):
        prompt = f"<s>[INST] <<SYS>>\n{self.sys}\n<</SYS>>\n\n{message} [/INST]</s>"
        self.tokenizer.padding_side = 'left'

        inputs = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=1024).to(
            self.device)
        output = self.model.generate(**inputs, max_new_tokens=1024, pad_token_id=self.tokenizer.pad_token_id)
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        response_start = full_response.rfind("[/INST]") + len("[/INST]")
        translation = full_response[response_start:].strip()

        # clean cache to avoid out-of-memory
        del inputs
        del output
        torch.cuda.empty_cache()
        gc.collect()

        return translation


