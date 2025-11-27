import glob
import cv2
import os
import numpy as np

directory = "Record/demo/vid_path_figs"  # 替换为目标目录路径
directory2 = "Record/demo/vid_ground_truth_figs"  # 替换为目标目录路径
directory3 = "Record/demo/vis_path"  # 替换为目标目录路径

# png_files = glob.glob(f"{directory}/*.png")

# print("找到的 PNG 文件：")
# print(png_files)

def generate_vid(path):
# 获取所有PNG文件并按名称排序（确保顺序正确）
    all_files = glob.glob(os.path.join(path, "*"))  # 获取目录下所有文件
    img_files = sorted([f for f in all_files if f.lower().endswith(".png")])
    img_files = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    print(f"找到的 PNG 文件：{img_files}")
    # 读取第一张图片以获取尺寸信息
    img = cv2.imread(img_files[0])
    height, width, channels = img.shape

    # 设置视频参数
    output_filename = "output.mp4"
    fps = 30  # 帧率（Frames Per Second）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码器（根据系统调整，如 'XVID' 对应 AVI）
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # 计算每张图片需要写入的帧数（0.5秒 * 帧率）
    frame_count = int(fps * 0.8)

    # 遍历所有图片并写入视频
    for i, file in enumerate(img_files):
        img = cv2.imread(file)
        if i <= 41: img = cv2.putText(img, 'FBE Part', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)
        else: img = cv2.putText(img, 'Density Guide Part', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)
        # cv2.imshow('img', cv2.resize(img, (500, 500)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        for _ in range(frame_count):
            video.write(img)

    # 释放资源
    video.release()
    print(f"视频已生成：{output_filename}")

def generate_vid2(vid_path_figs, vid_ground_truth_figs, vis_path):
    
    all_files = glob.glob(os.path.join(vid_path_figs, "*"))  # 获取目录下所有文件
    img_files = sorted([f for f in all_files if f.lower().endswith(".png")])
    path_img_files = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    all_files = glob.glob(os.path.join(vid_ground_truth_figs, "*"))  # 获取目录下所有文件
    img_files = sorted([f for f in all_files if f.lower().endswith(".png")])
    ground_img_files = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    output_filename = "vis_output.mp4"
    output_filename1 = "path_output.mp4"
    output_filename2 = "ground_output.mp4"

    fps = 100 # 帧率（Frames Per Second）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码器（根据系统调整，如 'XVID' 对应 AVI）
    img = cv2.imread('Record/demo/vis_path/0/0.png')
    height, width, channels = img.shape
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    img = cv2.imread(path_img_files[0])
    height, width, channels = img.shape
    video1 = cv2.VideoWriter(output_filename1, fourcc, fps, (width, height))
    img = cv2.imread(ground_img_files[0])
    height, width, channels = img.shape
    video2 = cv2.VideoWriter(output_filename2, fourcc, fps, (width, height))


    for i in range(166):
        all_files = glob.glob(os.path.join(vis_path +'/' + str(i), "*"))  # 获取目录下所有文件
        img_files = sorted([f for f in all_files if f.lower().endswith(".png")])
        vis_img_files = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
        if i<= 40: frame_count = int(1)
        else: frame_count = int(3)
        img1 = cv2.imread(path_img_files[i])
        img2 = cv2.imread(ground_img_files[i])
        for j in range(len(vis_img_files)):
            img = cv2.imread(vis_img_files[j])
            # total_img = cv2.putText(total_img, 'Density Guide Part', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10, cv2.LINE_AA)
            # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # cv2.imshow('img', total_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            for _ in range(frame_count):
                video.write(img)
                video1.write(img1)
                video2.write(img2)
    video.release()
    video1.release()
    video2.release()
    print(f"视频已生成：{output_filename}")

if __name__ == "__main__":
    generate_vid2(directory, directory2, directory3)  # 替换为目标目录路径