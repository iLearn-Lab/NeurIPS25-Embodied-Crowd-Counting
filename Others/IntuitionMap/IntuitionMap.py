
import numpy as np
import cv2
import os

class IntuitionMap:
    def __init__(self, config):
        self.config = config
        
    def get_panoramic_image(self, bgr_imgs):
        panorama_segment = []
        last_img = []
        for i, img in enumerate(bgr_imgs):
            width = img.shape[1]
            sub_img1, sub_img2 = img[:, :width // 2], img[:, width // 2:]
            if i == 0:
                sub_img1 = cv2.putText(sub_img1,"Direction %d" % (i), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
                last_img = cv2.putText(sub_img2,"Direction %d" % 7, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
                panorama_segment.append(sub_img1)                
            else:
                sub_img2 = cv2.putText(sub_img2,"Direction %d" % (i*2-1), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
                sub_img1 = cv2.putText(sub_img1,"Direction %d" % (i*2), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
                panorama_segment.append(sub_img2)
                panorama_segment.append(sub_img1)
        panorama_segment.append(last_img)
        panoramic_image = np.concatenate(panorama_segment, axis=1)
        return panoramic_image, panorama_segment
    
    def save_panoramic_image(self, panoramic_image, panorama_segments, timestamp):
        """
        Save the panoramic image with a timestamp-based folder, and also save the individual panorama segments.
        """
        # Create a directory with the timestamp as the folder name
        folder_path = os.path.join(self.save_directory, f"panorama_{timestamp}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the panoramic image in the folder
        panoramic_image_path = os.path.join(folder_path, "panoramic_image.png")
        cv2.imwrite(panoramic_image_path, panoramic_image)

        # Save the panorama segments (individual images) in the folder
        for idx, segment in enumerate(panorama_segments):
            segment_path = os.path.join(folder_path, f"segment_{idx}.png")
            cv2.imwrite(segment_path, segment)

        print(f"Panoramic image and segments saved in {folder_path}")

def get_record_img(path, idx):
    ret = []
    sub_paths = ['camera_front', 'camera_left', 'camera_back', 'camera_right']
    for sub_path in sub_paths:
        img_path = f'{path}\{sub_path}\{idx}.png'
        ret.append(cv2.imread(img_path))
    return np.array(ret)

def main():
    temp = IntuitionMap(None)
    ret = get_record_img("Record\\2025-02-26-16_34_02\ValueMap", 0)
    con, _ = temp.get_panoramic_image(ret)
    cv2.imwrite(".\\captured_images\\figure.png", con)