# -*- coding: utf-8 -*-
import openai
import pandas as pd
import time

count = 0

openai.api_key = 'your-key-here'

input_file_path = './dcard-crawler/csv/full_with_ai_response.csv'
output_file_path = './dcard-crawler/csv/final.csv'
dcard_df = pd.read_csv(input_file_path)
try:
    for index, row in dcard_df.iterrows():

        a = row["內容"]
        b = row["AI回應"]
        c = row["GPT使用者回應"]
        response = openai.ChatCompletion.create(
            model='ft:gpt-3.5-turbo-0125:personal:mul-toby:9aUzalfF',
            messages=[
                {"role": "system",
                 "content": "任務: 給定三句話，第一和第三句話是受訪者A，第二句話是輔導員B，請輸出受訪者A的情緒轉換狀況。"},
                {"role": "user", "content": f'A: {a} \n B: {b} \n A: {c}'},
            ]
        )
        gpt_content = response['choices'][0]['message']['content']
        dcard_df.at[index, '情緒轉換'] = gpt_content
        print(gpt_content)
        count += 1
        if count % 5 == 0:
            print(f'waiting now: {count}')
            time.sleep(10)

finally:
    # Write the DataFrame to a CSV file
    dcard_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print('Data written to file.')
