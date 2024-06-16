import pandas as pd
import os

# Define data and label file paths
mfcc_data_path = 'mfcc/train/'
face_data_path = 'face_coefficient/train/'

labels_path = 'mosei_labels_min10_train.csv'

# Read label file
labels_df = pd.read_csv(labels_path, usecols=[0,2,7,8,9,10,11,12])  # Read (segment_id,id,happy,sad,anger,surprise,disgust,fear)

# Initialize a DataFrame to store the statistics results
stats_df = pd.DataFrame()

# Loop through each row in labels, reading the corresponding CSV file
for _, row in labels_df.iterrows():

    # mfcc
    mfcc_file_path = os.path.join(mfcc_data_path, f"{row['id']}_{row['segment_id']}.csv")
    mfcc_data_df = pd.read_csv(mfcc_file_path, skiprows=1, usecols=list(range(0, 20)))

    # Calculate mean and variance
    mfcc_columns = mfcc_data_df.iloc[:, :]

    mfcc_mean_columns = mfcc_columns.mean()
    mfcc_var_columns = mfcc_columns.var()
    


    # face coefficient
    face_file_path = os.path.join(face_data_path, f"{row['id']}_{row['segment_id']}.csv")
    face_data_df = pd.read_csv(face_file_path, skiprows=1, usecols=list(range(0, 52)))

    # Calculate mean and variance
    face_columns = face_data_df.iloc[:, :]
    face_mean_columns = face_columns.mean()
    face_var_columns = face_columns.var()
    
    # Label encoding for emotions
    emotion_labels = {}
    for emotion in ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']:
        emotion_value = row[emotion]
        if emotion_value == 0: 
            emotion_labels[emotion] = 0.0
        elif 0 < emotion_value <= 2:
            emotion_labels[emotion] = 1.0
        else: 
            emotion_labels[emotion] = 2.0

    # Construct a row for the results DataFrame, including mean and variance for all fields
    stats_row = {
        'segment_id': row['segment_id'],
        'id': row['id'],
        'happy': emotion_labels['happy'],
        'sad': emotion_labels['sad'],
        'anger': emotion_labels['anger'],
        'surprise': emotion_labels['surprise'],
        'disgust': emotion_labels['disgust'],
        'fear': emotion_labels['fear']
    }

    # Generate column names for means and variances and add to the results row
    stats_row.update({f'mfcc{idx+1}_mean': val for idx, val in enumerate(mfcc_mean_columns)})
    stats_row.update({f'mfcc{idx+1}_var': val for idx, val in enumerate(mfcc_var_columns)})
    stats_row.update({f'face{idx+1}_mean': val for idx, val in enumerate(face_mean_columns)})
    stats_row.update({f'face{idx+1}_var': val for idx, val in enumerate(face_var_columns)})
    
    # Use concat instead of append
    stats_df = pd.concat([stats_df, pd.DataFrame([stats_row])], ignore_index=True)

# Save the results to a new CSV file
stats_df.to_csv('all_feature/train.csv', index=False)
print("Process completed!")
