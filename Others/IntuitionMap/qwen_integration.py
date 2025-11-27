from openai import OpenAI
import os
import base64
import re
import cv2
import numpy as np
from Agent.Prompts2 import DECIDE_DIRECTION, TOP_DOWN_2, TOP_DOWN_3, DECIDE_DIRECTION_2
from utils.saver import image_saver
from PIL import Image
import io

class Qwen:

    def __init__(self, config):
        
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
            api_key = config['Qwen']['API_KEY'],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )    

        self.panorama_saver = image_saver(config['now'],config['Record_root'], f'LLMJudge/panorama')

        if config['LLM_name'] == 'Qwen':
            self.model = "qwen-vl-max-latest"
        elif config['LLM_name'] == 'Qwen7B':
            self.model = "qwen-vl-plus"     

    def image_to_data_url(self, img_rgb):

        imgByteArr = io.BytesIO()
        Image.fromarray(img_rgb).save(imgByteArr, format='PNG')
        img_data = imgByteArr.getvalue()

        base64_encoded_data = base64.b64encode(img_data).decode('utf-8')
        mime_type = 'image/png'
        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"
        
    def get_resopnse_2(self,request,img_bgr:list, print_result = False):

        if not isinstance(img_bgr, list):
            img_bgr = [img_bgr]

        content = [
            {
                "type": "text",
                "text": request
            }
        ]
        for img in img_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_url = self.image_to_data_url(img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": img_url
                }
            )

        message = [
            {
                "role": "user",
                "content": content
            }
        ]    

        # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/model-studio/getting-started/model        
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages = message
        )
        content = completion.choices[0].message.content
       
        if print_result:
            print('\n--------------------------------------')
            print(content)
            print('--------------------------------------\n')
        return content   
        
    def LLMChooseDown2(self, images, llm_score_thresh, print_result = True):
        print('\n--------------------------------------')        
        print('LLM choosing down...')        
        content = self.get_resopnse_2(TOP_DOWN_3, images, print_result)
        match = re.search(r'Answer:[\s\S]*?(\d+\.\d+)', content)
        if match:
            answer = match.group().split(' ')[-1]
            answer = float(answer)
            if answer > llm_score_thresh:
                return True
            else:
                return False
        else:
            print('=====> No integer found in string')
            return -1
        
    def get_visual_prompt(self, imgs, if_mark, downsample_rate, img_idx = None):

        # Add mark on each image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        line_type = 1
        color = (255, 255, 255)    
        marked_imgs = []    
        for i, img in enumerate(imgs):
            marked_img = img.copy()
            marked_img = cv2.resize(marked_img, (0, 0), fx=downsample_rate, fy=downsample_rate)
            if if_mark[i]:
                width = marked_img.shape[1]
                if img_idx is None:
                    text = "%d" % (i)
                else:
                    text = "%d" % (img_idx[i])
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)
                text_x = width//2-text_width//2; text_y = 0
                marked_img = cv2.rectangle(marked_img, 
                                (text_x-5, text_y), 
                                (text_x + text_width + 5, text_y + text_height + 5), 
                                (0, 0, 0), -1)                    
                marked_img = cv2.putText(marked_img, text, (text_x, text_y + text_height), font, font_scale, color, 4, cv2.LINE_AA)
            marked_imgs.append(marked_img)            

        # Add black borders between sub-images
        bordered_segments = []
        for img in marked_imgs[:-1]:  # Exclude the last image from adding a border
            bordered_segments.append(img)
            bordered_segments.append(np.zeros((img.shape[0], 10, 3), dtype=np.uint8))  # 10-pixel wide black border
        bordered_segments.append(marked_imgs[-1])  # Add the last image without a border
        panoramic_image = np.concatenate(bordered_segments, axis=1)

        self.panorama_saver.save(panoramic_image)
        return panoramic_image, marked_imgs    