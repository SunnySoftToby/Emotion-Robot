from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# 模型架構
class EmotionModelCNN(nn.Module):
    def __init__(self, input_dim_mfcc, input_dim_face, sequence_length_mfcc, sequence_length_face, num_classes=3):
        super(EmotionModelCNN, self).__init__()

        # MFCC 輸入的卷積層
        self.conv_mfcc = nn.Sequential(
            nn.Conv1d(in_channels=input_dim_mfcc, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # 臉部特徵輸入的卷積層
        self.conv_face = nn.Sequential(
            nn.Conv1d(in_channels=input_dim_face, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # 最終分類的全連接層
        self.fc = nn.Sequential(
            nn.Linear(14528, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  
        )

    def forward(self, mfcc, face):
        # 使用 CNN 提取 MFCC 特徵
        mfcc_out = self.conv_mfcc(mfcc.permute(0, 2, 1))  # permute to [batch_size, channels, sequence_length]
        mfcc_out = mfcc_out.view(mfcc_out.size(0), -1)  # 展平

        # 使用 CNN 提取臉部特徵
        face_out = self.conv_face(face.permute(0, 2, 1))  # permute to [batch_size, channels, sequence_length]
        face_out = face_out.view(face_out.size(0), -1)  # 展平

        # 合併來自兩個流的特徵
        combined = torch.cat((mfcc_out, face_out), dim=1)

        # 最終分類層，這裡不再需要 softmax 操作
        logits = self.fc(combined)

        return logits

# 載入模型
model = EmotionModelCNN(input_dim_mfcc=20, input_dim_face=52, sequence_length_mfcc=624, sequence_length_face=300, num_classes=3)
model.load_state_dict(torch.load('emotion_model.pth', map_location=torch.device('cpu')))
model.eval()

# 修剪特徵
def trim_features(features, length):
    if len(features) < length:
        padding = length - len(features)
        features = torch.nn.functional.pad(features, (0, 0, 0, padding), 'constant', 0)
    elif len(features) > length:
        features = features[:length]
    return features

# 預測情緒
def predict_emotion(mfcc_features, face_features):
    mfcc_tensor = trim_features(torch.tensor(mfcc_features, dtype=torch.float32), 624).unsqueeze(0)
    face_tensor = trim_features(torch.tensor(face_features, dtype=torch.float32), 300).unsqueeze(0)
    with torch.no_grad():
        outputs = model(mfcc_tensor, face_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


@app.route('/predict',methods=['POST'])
def predict():
    # Check if the request contains JSON data
    if not request.json:
        return jsonify({'error': 'No JSON data received'}), 400
    
    # Get JSON data
    data = request.get_json()
    
    # Check if 'mfcc_features' and 'face_features' are in the JSON data
    if 'mfcc_features' not in data or 'face_features' not in data:
        return jsonify({'error': 'Missing required data (mfcc_features or face_features)'}), 400
    
    # Retrieve 'mfcc_features' and 'face_features' from JSON
    mfcc_features = data['mfcc_features']
    face_features = data['face_features']


    # Predict emotion using the model
    predicted_label = predict_emotion(mfcc_features, face_features)
    
    # Map predicted label index to emotion category
    labels = ['Negative', 'Neutral', 'Positive']
    predicted_emotion = labels[predicted_label]

    response = jsonify({'prediction': predicted_emotion})

    return response
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
