import time
from translator import Translator
import pandas as pd


def process_except_selected_cols(df, function, excluded_cols):
    emotion_mapping = {
        'anger': '憤怒',
        'disgust': '噁心',
        'fear': '恐懼',
        'joy': '開心',
        'neutral': '中性',
        'sadness': '傷心',
        'surprise': '驚訝'
    }

    for column in df.columns:
        if column in excluded_cols:
            df[column] = df[column].map(emotion_mapping)
        else:
            df[column] = df[column].apply(function)
    return df


if __name__ == "__main__":

    input_file_path = './MELD/modified_aba_modified.csv'
    output_file_path = f'./MELD/modified_aba_modified_translated.csv'

    meld_df = pd.read_csv(input_file_path, usecols=['A_dialog_0', 'B_dialog_0', 'A_dialog_1', 'A_emotion_0', 'A_emotion_1'])
    excluded_cols = ['A_emotion_0', 'A_emotion_1']

    en_to_zh_translator = Translator()
    start_time = time.time()
    processed_df = process_except_selected_cols(meld_df, en_to_zh_translator.translate, excluded_cols)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

    processed_df.to_csv(output_file_path, index=False)
