import cv2
import numpy as np
import zarr

dataset_path = "pusht_cchi_v7_replay.zarr.zip"
dataset_root = zarr.open(dataset_path, 'r')

# 形状为(25650, 96, 96, 3)的NumPy数组
train_image_data = dataset_root['data']['img'][:]
# 转为uint8
train_image_data = train_image_data.astype(np.uint8)

# 设置输出视频的路径和帧率
output_video_path = "output_video.avi"
frame_rate = 30  # 每秒30帧
# 获取图像的宽度和高度
height, width = train_image_data.shape[1:3]
# 创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
# 写入每一帧到视频
for frame in train_image_data:
    out.write(frame)
# 释放 VideoWriter
out.release()
print("Video has been created at", output_video_path)
