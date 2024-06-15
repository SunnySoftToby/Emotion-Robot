import csv
import json
import tiktoken

file_path = './MELD/modified_aba_modified_translated.csv'
jsonl_file_path = './MELD/MELD.jsonl'

with open(file_path, mode='r', newline='', encoding='utf-8') as file, \
     open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        messages = [
            {"role": "system", "content": "任務: 給定三句話，第一和第三句話是受訪者A，第二句話是輔導員B，請輸出受訪者A的情緒轉換狀況。"},
            {"role": "user", "content": f'A: {row[0]} \n B: {row[1]} \n A: {row[2]}'},
            {"role": "assistant", "content": f'情緒轉換: {row[3]} -> {row[4]}'}
        ]

        chat_format = {"messages": messages}
        jsonl_file.write(json.dumps(chat_format, ensure_ascii=False) + '\n')

# 设置编码器
enc = tiktoken.get_encoding("cl100k_base")
# 计算token数量
total_tokens = 0

with open(jsonl_file_path, mode='r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        for message in entry["messages"]:
            tokens = enc.encode(message["content"])
            total_tokens += len(tokens)

print(f"Total tokens: {total_tokens}")