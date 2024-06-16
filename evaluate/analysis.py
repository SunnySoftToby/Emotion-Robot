import pandas as pd

# 读取CSV文件
input_file_path = './dcard-crawler/csv/final.csv'
df = pd.read_csv(input_file_path)

# 定义情绪到情感类别的映射字典
emotion_map = {
    "恐懼": "負面",
    "害怕": "負面",
    "感激": "正面",
    "正面": "正面",
    "感謝": "正面",
    "傷心": "負面",
    "壓力": "負面",
    "負面": "負面",
    "後悔": "負面",
    "開心": "正面",
    "中性": "中性",
    "憤怒": "負面",
}

# 初始化两个新的列
df["原本"] = ""
df["事後"] = ""

# 迭代每一行，提取情绪转换并映射到情感类别
for index, row in df.iterrows():
    emotions = row["情緒轉換"].replace("情緒轉換: ", "").replace(" ", "").split("->")
    if len(emotions) == 2:
        original_emotion = emotion_map.get(emotions[0], "未知")
        after_emotion = emotion_map.get(emotions[1], "未知")
        df.at[index, "原本"] = original_emotion
        df.at[index, "事後"] = after_emotion



emotion_transition_count = df.groupby(["原本", "事後"]).size().reset_index(name='counts')
print(emotion_transition_count)

# 保存
output_file_path = './dcard-crawler/csv/final_with_emotions.csv'
df.to_csv(output_file_path, index=False)



#    原本  事後  counts
# 0  中性  正面      10
# 1  中性  負面       2
# 2  正面  中性       1
# 3  正面  正面       2
# 4  正面  負面       1
# 5  負面  中性       6
# 6  負面  正面       9
# 7  負面  負面      44
