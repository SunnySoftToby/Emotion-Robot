import time
import argparse

from translator import Translator
from json_io import read_json, write_json


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

    input_file_path = './EXTES/ExTES.json'
    output_file_path = f'./EXTES/ExTES_translated_{start}_{end}.json'

    dialogues = read_json(input_file_path)

    dialogues = dialogues[start:end]

    print(len(dialogues))
    en_to_zh_translator = Translator()

    start_time = time.time()

    if dialogues:
        translated_dialogues = [translate(dialogue, en_to_zh_translator) for dialogue in dialogues]
        write_json(output_file_path, translated_dialogues)
        print("Translation complete. Translated dialogues written to:", output_file_path)

    else:
        print("Failed to read the JSON file.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
