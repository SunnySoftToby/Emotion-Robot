import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.translator import Translator
import argparse

if __name__ == "__main__":
    input_file_path = './dcard-crawler/csv/full.csv'
    output_file_path = './dcard-crawler/csv/full_with_ai_response.csv'
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

    dcard_df = pd.read_csv(input_file_path, usecols=['看板', '關鍵字', 'URL', '內容'])

    dcard_df = dcard_df.iloc[start:end]

    emotion_bot = Translator()
    emotion_bot.sys = (
        f"回答不要超過二十個字，全程都要使用繁體中文回答。"
        f"你是一位專業的情感輔導師，正在提供心理諮詢服務。"
        f"你的主要工作包括聆聽使用者分享他們的感情和生活困擾，分析他們的心理狀態，並根據每位使用者的具體情況提供個性化的建議和策略。"
        f"你專注於幫助人們改善人際關係、處理情緒困擾，並提升他們的整體心理健康。"
        f"你的目標是創造一個支持和理解的環境，讓使用者能夠自我探索和成長。")

    dcard_df['AI回應'] = dcard_df['內容'].apply(lambda content: emotion_bot.translate(content).replace('\n', ''))
    if not os.path.isfile(output_file_path):
        dcard_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    else:
        dcard_df.to_csv(output_file_path, index=False, encoding='utf-8-sig', mode='a', header=False)


