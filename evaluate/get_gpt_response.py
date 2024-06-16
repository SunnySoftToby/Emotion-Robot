import openai
import pandas as pd
import time

count = 0

# 設定 API 金鑰
openai.api_key = 'your-key-here'


input_file_path = './dcard-crawler/csv/full_with_ai_response.csv'
dcard_df = pd.read_csv(input_file_path)
for index, row in dcard_df.iterrows():
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": "你是一個因壓力正在接受心理諮商的人，你要根據輔導師的內容表現出負面或中性或正面的回應"},
            {"role": "user", "content": "您好，歡迎來到我們的情感支持中心，我們致力於提供溫暖和專業的陪伴，幫助您度過每一個難關"},
            {"role": "assistant", "content": row["內容"]},
            {"role": "user", "content": row["AI回應"]}
        ]
    )
    gpt_content = response['choices'][0]['message']['content']
    dcard_df.at[index, 'GPT使用者回應'] = gpt_content
    print(gpt_content)
    count += 1
    if count % 5 == 0:
        print(f'waiting now: {count}')
        time.sleep(20)
    dcard_df.at[index, 'GPT使用者回應'] = dcard_df.at[index, 'GPT使用者回應'].replace('\n', ' ')

dcard_df.to_csv(input_file_path, index=False, encoding='utf-8-sig')
