import pandas as pd

# 讀取 csv1 和 csv2
csv1 = pd.read_csv('modified_aba_modified.csv')
csv2 = pd.read_csv('meld_dev_sent_emo_modified_sentiment.csv')

# 合併 DataFrame，假設對照欄位是 'A0_id','B0_id','A1_id'
merged_df = pd.merge(csv1, csv2[['A0_id','B0_id','A1_id', 'A_sentiment_0','A_sentiment_1']], on=['A0_id','B0_id','A1_id'], how='left')

# 保留 csv1 的特定欄位
columns_to_keep1 = ['A_dialog_0']
columns_to_keep2 = ['B_dialog_0', 'A_dialog_1']
merged_df = merged_df[['A0_id','B0_id','A1_id']+columns_to_keep1+['A_sentiment_0']+columns_to_keep2+['A_sentiment_1'] ]

# 將合併後的 DataFrame 寫回到新的 CSV 文件
merged_df.to_csv('sentiment.csv', index=False)