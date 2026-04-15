import os
from tqdm import tqdm
from PIL import Image
import numpy as np

def get_img_path(img_path):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(img_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    img_paths = [os.path.join(img_path, frame_name) for frame_name in frame_names]

    return img_paths

def load_video(video_path, if_RGB = True):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    img_paths = [os.path.join(video_path, frame_name) for frame_name in frame_names]

    image_list = []
    for _, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        img_pil = Image.open(img_path)
        if if_RGB:
            img_np = np.array(img_pil.convert("RGB"))
        else:
            img_np = np.array(img_pil)[:, :, ::-1]
        image_list.append(img_np)
    return image_list

def load_numpy(path):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(path)
        if os.path.splitext(p)[-1] in [".npy"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    img_paths = [os.path.join(path, frame_name) for frame_name in frame_names]

    image_list = []
    for _, img_path in enumerate(tqdm(img_paths, desc="frame loading (.npy)")):
        img = np.load(img_path)
        image_list.append(img)
    return image_list

def load_partial_video(video_path, end_idx, if_RGB = True):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    img_paths = [os.path.join(video_path, frame_name) for frame_name in frame_names]

    image_list = []
    if end_idx is not None:
        for _, img_path in enumerate(tqdm(img_paths[:end_idx], desc="frame loading (JPEG)")):
            img_pil = Image.open(img_path)
            if if_RGB:
                img_np = np.array(img_pil.convert("RGB"))
            else:
                img_np = np.array(img_pil)[:, :, ::-1]
            image_list.append(img_np)
    else:
        for _, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
            img_pil = Image.open(img_path)
            if if_RGB:
                img_np = np.array(img_pil.convert("RGB"))
            else:
                img_np = np.array(img_pil)[:, :, ::-1]
            image_list.append(img_np)
    return image_list

def load_partial_numpy(path, end_idx):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(path)
        if os.path.splitext(p)[-1] in [".npy"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    img_paths = [os.path.join(path, frame_name) for frame_name in frame_names]

    image_list = []
    if end_idx is not None:
        for _, img_path in enumerate(tqdm(img_paths[:end_idx], desc="frame loading (.npy)")):
            img = np.load(img_path)
            image_list.append(img)
    else:
        for _, img_path in enumerate(tqdm(img_paths, desc="frame loading (.npy)")):
            img = np.load(img_path)
            image_list.append(img)
    return image_list