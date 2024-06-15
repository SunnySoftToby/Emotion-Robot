import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import gc
import os
import argparse

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

class Translator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("../model_weight/TAIDE-LX-7B-Chat", use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained("../model_weight/TAIDE-LX-7B-Chat", torch_dtype=torch.float16)
        self.model.to(self.device)

    def translate(self, message):
        sys = "將使用者輸入的英文句子翻譯成繁體中文，只需返回翻譯結果 ，不要進行超譯，只需要忠實地還原原文意思就可以，不要包括任何額外的訊息和無關的訊息。"
        prompt = f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{message} [/INST]"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_length=1024, pad_token_id=self.tokenizer.eos_token_id)
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        response_start = full_response.find("[/INST]") + len("[/INST]")
        translation = full_response[response_start:].strip()

        del inputs
        del output
        torch.cuda.empty_cache()
        gc.collect()

        return translation


def translate(dialogue, translator):
    try:
        translated_dialogue = dialogue.copy()
        translated_dialogue['description'] = translator.translate(dialogue.get('description', ''))
        for item in translated_dialogue['content']:
            for key in item:
                if key in ['User', 'AI']:
                    item[key] = translator.translate(item[key])
        return translated_dialogue
    except Exception as e:
        print(dialogue)
        return "FAILED"


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
2222

def write_json(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")


if __name__ == "__main__":

    # 創建解析器
    parser = argparse.ArgumentParser(description="讀取 start 和 end 參數")

    # 添加參數
    parser.add_argument('--start', type=int, required=True, help="起始值")
    parser.add_argument('--end', type=int, required=True, help="結束值")

    # 解析參數
    args = parser.parse_args()

    # 獲取參數值
    start = args.start
    end = args.end

    input_file_path = '../EXTES/ExTES.json'
    output_file_path = '../EXTES/ExTES_translated_part1.json'
    dialogues = read_json(input_file_path)

    dialogues = dialogues[start:end]  # part1

    print(len(dialogues))
    translator = Translator()

    start_time = time.time()

    if dialogues:
        translated_dialogues = [translate(dialogue, translator) for dialogue in dialogues]
        write_json(output_file_path, translated_dialogues)
        print("Translation complete. Translated dialogues written to:", output_file_path)

    else:
        print("Failed to read the JSON file.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
