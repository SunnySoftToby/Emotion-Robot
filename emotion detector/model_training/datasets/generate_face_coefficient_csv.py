import pandas as pd
import os
from tools.get_face_coefficient import append_face_coefficient_row
# 讀取CSV
label_file = 'mosei_labels_min10_valid.csv'
labels = pd.read_csv(label_file, usecols=[0,1,2])  # 讀取第1至3欄 (segment_id,file,id)

# 讀取每row {id}_{file_without_mp4}資料夾裡的圖片 
# 輸出 {id}_{segment_id}
for index, row in labels.iterrows():
    segment_id = row['segment_id']
    file = row['file']
    id = row['id']

    # 去除掉.mp4
    file_name = os.path.splitext(file)[0]
    
    
    frame_count = 1
    
    parent_folder = 'Raw\output' # images
    target_folder = os.path.join(parent_folder, f"{id}_{file_name}")
    

    
    frames_path = []
    jpg_files = [f for f in os.listdir(target_folder) if f.endswith('.jpg')]
    
    # 按文件名排序
    jpg_files.sort()

    # 取300 frame => 30fps*10s
    selected_files = jpg_files[:300]


    input_image_paths = [os.path.join(target_folder, jpg_file) for jpg_file in jpg_files]
    append_face_coefficient_row(input_image_paths,f"face_coefficient\\valid\\{id}_{segment_id}.csv")

    # # 查找文件，直到文件名不再符合模式
    # while True:

    #     # 設定欲查找檔案模式
    #     new_file_name = f"{id}_{file_name}_{frame_count:05d}.jpg"
        
    #     # 查找路徑
    #     new_file_path = os.path.join(target_folder, new_file_name)
                
    #     # 如果沒有符合的圖片 break => 進行下一輪{id}_{file_without_mp4}查找
    #     if not os.path.exists(new_file_path):
    #         break
    #     else : 
    #         # 顯示查找到的圖片檔案
    #         frames_path.append(new_file_path)

        # frame_count += 1
    # append_face_coefficient_row(["--qXJuDtHPw_5_00002.jpg","--qXJuDtHPw_5_00003.jpg","--qXJuDtHPw_5_00001.jpg"],"--qXJuDtHPw_5.csv")

