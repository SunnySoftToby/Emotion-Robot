import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import random

# 設置隨機種子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# 定義資料集類別
class EmotionDataset(Dataset):
    def __init__(self, comsei_data_csv, msa_data_csv, mfcc_dir, face_dir, mfcc_sequence_length=625, face_sequence_length=300, train_mode=False):
        self.comsei_data_csv = pd.read_csv(comsei_data_csv)[['id', 'segment_id']]
        self.msa_data_csv = pd.read_csv(msa_data_csv)[['id', 'segment_id', 'annotation']]

        # 合併兩個 DataFrame
        self.data_info = pd.merge(self.comsei_data_csv, self.msa_data_csv, on=['id', 'segment_id'])

        self.mfcc_dir = mfcc_dir
        self.face_dir = face_dir
        self.mfcc_sequence_length = mfcc_sequence_length
        self.face_sequence_length = face_sequence_length

        # 過濾不符合長度要求的數據
        self.data_info = self.filter_data(self.data_info)
        self.train_mode = train_mode

    def filter_data(self, data_info):
        valid_indices = []
        for idx in range(len(data_info)):
            row = data_info.iloc[idx]
            video_id = f"{row['id']}_{row['segment_id']}"
            mfcc_features = pd.read_csv(f"{self.mfcc_dir}/{video_id}.csv").values
            face_features = pd.read_csv(f"{self.face_dir}/{video_id}.csv").values
            # if len(mfcc_features) >= self.mfcc_sequence_length and len(face_features) >= self.face_sequence_length:
            valid_indices.append(idx)
        return data_info.iloc[valid_indices]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        video_id = f"{row['id']}_{row['segment_id']}"
        
        label_str = row['annotation']
        # 將字符串標籤轉換為整數（有序分類）
        if label_str == 'Positive':
            label = 2
        elif label_str == 'Neutral':
            label = 1
        elif label_str == 'Negative':
            label = 0
        else:
            raise ValueError(f"Unknown label: {label_str}")
        
        mfcc_features = pd.read_csv(f"{self.mfcc_dir}/{video_id}.csv").values
        face_features = pd.read_csv(f"{self.face_dir}/{video_id}.csv").values

        mfcc_features = self.trim_features(torch.tensor(mfcc_features, dtype=torch.float32), self.mfcc_sequence_length)
        face_features = self.trim_features(torch.tensor(face_features, dtype=torch.float32), self.face_sequence_length)
        
        if np.isnan(mfcc_features).any():
            print("Warning: NaN values found in MFCC features.")
        
        if np.isnan(face_features).any():
            print("Warning: NaN values found in Face features.")

        return mfcc_features, face_features, torch.tensor(label, dtype=torch.long)

    def read_csv_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Remove trailing comma if present
            lines = [line.strip().rstrip(',') for line in lines]
            data = [list(map(float, line.split(','))) for line in lines]
        return data

    def trim_features(self, features, length):
        if len(features) < length:
            padding = length - len(features)
            features = torch.nn.functional.pad(features, (0, 0, 0, padding), 'constant', 0)
        elif len(features) > length:
            if self.train_mode:
                start_idx = random.randint(0, len(features) - length)
                features = features[start_idx:start_idx + length]
            else:
                features = features[:length]
        return features

# 建立資料集
input_dim_mfcc = 20
input_dim_face = 52
sequence_length_mfcc = 624
sequence_length_face = 300
num_classes = 3
train_dataset = EmotionDataset('modified_dataset/all_feature/train.csv', 'modified_dataset/all_feature/merged_label.csv', 'modified_dataset/mfcc/test', 'modified_dataset/face_coefficient/test',mfcc_sequence_length=sequence_length_mfcc,face_sequence_length=sequence_length_face,train_mode=True)
valid_dataset = EmotionDataset('modified_dataset/all_feature/valid.csv', 'modified_dataset/all_feature/merged_label.csv', 'modified_dataset/mfcc/test', 'modified_dataset/face_coefficient/test',mfcc_sequence_length=sequence_length_mfcc,face_sequence_length=sequence_length_face)
# valid_dataset= EmotionDataset('modified_dataset/all_feature/train.csv', 'modified_dataset/all_feature/merged_label.csv', 'modified_dataset/mfcc/test', 'modified_dataset/face_coefficient/test',mfcc_sequence_length=sequence_length_mfcc,face_sequence_length=sequence_length_face)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

print(f"Remaining data in training dataset: {len(train_dataset)}")
print(f"Remaining data in validation dataset: {len(valid_dataset)}")
print('載入完成!')

# 定義模型類別
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

# 定義訓練
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義模型、損失函數和優化器


model = EmotionModelCNN(input_dim_mfcc, input_dim_face, sequence_length_mfcc, sequence_length_face, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 設置訓練參數
#num_epochs = 100
num_epochs = 30
train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []

# 訓練循環
for epoch in range(num_epochs):
    # 訓練模式
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for mfcc_data, face_data, labels in train_loader:
        mfcc_data, face_data, labels = mfcc_data.to(device), face_data.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向傳播，計算損失
        outputs = model(mfcc_data, face_data)
        loss = criterion(outputs, labels)

        # 反向傳播及優化
        loss.backward()
        optimizer.step()

        # 統計訓練損失
        running_loss += loss.item()

        # 計算訓練準確率
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # 計算平均訓練損失與準確率
    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train * 100
    train_losses.append(train_loss)
    train_accuracy.append(train_acc)

    # 驗證模式
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for mfcc_data, face_data, labels in valid_loader:
            mfcc_data, face_data, labels = mfcc_data.to(device), face_data.to(device), labels.to(device)

            # 前向傳播，計算損失
            outputs = model(mfcc_data, face_data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 計算驗證準確率
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # 計算平均驗證損失與準確率
    val_loss = val_loss / len(valid_loader)
    val_acc = correct_val / total_val * 100
    val_losses.append(val_loss)
    val_accuracy.append(val_acc)

    # 打印每個 epoch 的訓練和驗證結果
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# 儲存model
torch.save(model.state_dict(), 'emotion_model.pth')


# 繪製訓練過程中的損失圖和準確度圖
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 100)
plt.legend()
plt.show()

# 在驗證集上獲得所有預測值和真實標籤
model.eval()
val_preds = []
val_targets = []

with torch.no_grad():
    for mfcc_data, face_data, labels in valid_loader:
        mfcc_data, face_data, labels = mfcc_data.to(device), face_data.to(device), labels.to(device)

        # 前向傳播，獲得模型預測
        outputs = model(mfcc_data, face_data)
        _, predicted = torch.max(outputs.data, 1)

        # 保存預測和真實標籤
        val_preds.extend(predicted.cpu().numpy())
        val_targets.extend(labels.cpu().numpy())


from sklearn.metrics import confusion_matrix

# 計算混淆矩陣
conf_matrix = confusion_matrix(val_targets, val_preds)

# 繪製混淆矩陣
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, ['Negative', 'Neutral', 'Positive'])
plt.yticks(tick_marks, ['Negative', 'Neutral', 'Positive'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2. else "black")

plt.tight_layout()
plt.show()
