import openai
import base64
import os
import cv2
import numpy as np
from utils.saver import image_saver, image_saver_plt
from Point_cloud.Map_element import UNKNOWN, OBSTACLE, EXPLORED
from Agent.Prompts import TOP_DOWN_2, TOP_DOWN_3, DECIDE_DIRECTION_2
import re
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from PIL import Image
import io
from langchain_openai import ChatOpenAI

class GPT:
    
    def __init__(self, config):
        openai.api_key = config['GPT']['OPENAI_API_KEY']
        self.config = config
        self.panorama_saver = image_saver(config['now'],config['Record_root'], f'LLMJudge/panorama')
        self.top_down_panorama_saver = image_saver(config['now'],config['Record_root'], f'LLMJudge/top_to_down')
        self.client = openai

        os.environ["OPENAI_API_KEY"] = self.config['GPT']['OPENAI_API_KEY']
        self.llm = ChatOpenAI(
            temperature=config['GPT']['temperature'],
            model_name=config['GPT']['model_name']
        )        

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

        message = [
            {
                "type": "text",
                "text": request
            }
        ]
        for img in img_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_url = self.image_to_data_url(img)
            message.append(
                {
                    "type": "image_url",
                    "image_url": img_url
                }
            )
        prompt = ChatPromptTemplate.from_messages(
            messages = [
                HumanMessagePromptTemplate.from_template(
                    message
                ),
            ]
        )
        chain = prompt | self.llm
        content = chain.invoke({}).content

        if print_result:
            print('\n--------------------------------------')
            print(content)
            print('--------------------------------------\n')
        return content
        
    def encode_image(self, image_array):
        _, buffer = cv2.imencode('.jpg', image_array)
        return base64.b64encode(buffer).decode("utf-8")
    
    def get_response(self, prompt, images, print_result = False):
        if not isinstance(images, list):
            images = [images]
        base64_images = [self.encode_image(image) for image in images]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images]
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=self.config['LLMJudge']['temperature']  # Set the temperature here
        )
        content = response.choices[0].message.content
        if print_result: print(content)
        return content
    
    def LLMChooseDown(self, images, print_result = True):
        print('\n--------------------------------------')        
        print('LLM choosing down...')        
        content = self.get_resopnse_2(TOP_DOWN_2, images, print_result)
        match = re.search(r'Answer: (?s).*', content)
        if match:
            answer = match.group().split(' ')[-1]
            if answer == 'Yes':
                return True
            elif answer == 'No':
                return False
        else:
            print('=====> No integer found in string')
            return -1  
        
    def LLMChooseDown2(self, images, llm_score_thresh, print_result = True):
        print('\n--------------------------------------')        
        print('LLM choosing down...')        
        content = self.get_resopnse_2(TOP_DOWN_3, images, print_result)
        match = re.search(r'(?<=Answer:)[\s\S]*?(\d+\.\d+)', content)
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
    
    def LLMChooseDirection(self, images, print_result = True):
        print('\n--------------------------------------')        
        print('LLM choosing direction...')
        content = self.get_resopnse_2(DECIDE_DIRECTION_2, images, print_result)
        try:
            ret = int(content.split("'Direction':")[1].split("}")[0].strip()[-2])
        except:
            return None
        return ret
    
    def get_panoramic_image(self, bgr_imgs, useless_directions=[]):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        line_type = 4
        color = (255, 255, 255)
        panorama_segment = []
        last_img = []
        for i, img in enumerate(bgr_imgs):
            width = img.shape[1]
            sub_img1, sub_img2 = img[:, :width // 2], img[:, width // 2:]
            text_x = 30; text_y = 50
            if i == 0:
                if 0 not in useless_directions:
                    sub_img1 = np.ascontiguousarray(sub_img1)
                    text = "Direction %d" % (i)
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)
                    cv2.rectangle(sub_img1, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)                    
                    sub_img1 = cv2.putText(sub_img1, text, (text_x, text_y), font, font_scale, color, 4, cv2.LINE_AA)
                    panorama_segment.append(sub_img1)
                    
                if 7 not in useless_directions: 
                    sub_img2 = np.ascontiguousarray(sub_img2)
                    text = "Direction %d" % 7
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)                    
                    cv2.rectangle(sub_img2, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)                    
                    last_img = cv2.putText(sub_img2, text, (text_x, text_y), font, font_scale, color, 4, cv2.LINE_AA)
            else:
                if (i * 2 - 1) not in useless_directions:
                    sub_img2 = np.ascontiguousarray(sub_img2)
                    text = "Direction %d" % (i * 2 - 1)
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)
                    cv2.rectangle(sub_img2, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
                    sub_img2 = cv2.putText(sub_img2, text, (text_x, text_y), font, font_scale, color, 4, cv2.LINE_AA)
                    panorama_segment.append(sub_img2)
                if (i * 2) not in useless_directions:
                    sub_img1 = np.ascontiguousarray(sub_img1)
                    text = "Direction %d" % (i * 2)
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)  
                    cv2.rectangle(sub_img1, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)                                      
                    sub_img1 = cv2.putText(sub_img1, text, (text_x, text_y), font, font_scale, color, 4, cv2.LINE_AA)
                    panorama_segment.append(sub_img1)
        if 7 not in useless_directions: panorama_segment.append(last_img)
        
        # Add black borders between sub-images
        bordered_segments = []
        for segment in panorama_segment[:-1]:  # Exclude the last image from adding a border
            bordered_segments.append(segment)
            bordered_segments.append(np.zeros((segment.shape[0], 10, 3), dtype=np.uint8))  # 10-pixel wide black border

        bordered_segments.append(panorama_segment[-1])  # Add the last image without a border
        panoramic_image = np.concatenate(bordered_segments, axis=1)
        self.panorama_saver.save(panoramic_image)
        return panoramic_image, panorama_segment
    
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
    
    def get_panoramic_image_down(self, bgr_imgs, useless_directions=[]):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        line_type = 4
        color = (255, 255, 255)        
        panorama_segment = []
        text_x = 80; text_y = 100
        for i, img in enumerate(bgr_imgs):
            if i > 3: break
            if (i * 2 - 1) % 8 not in useless_directions or (i * 2) % 8 not in useless_directions:
                sub_img2 = np.ascontiguousarray(img).copy()
                text = "Direction %d" % i
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)
                cv2.rectangle(sub_img2, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)                
                sub_img2 = cv2.putText(sub_img2, "Direction %d" % i, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 6, cv2.LINE_AA)
                panorama_segment.append(sub_img2)
        
        down_img = np.ascontiguousarray(bgr_imgs[-1]).copy()
        text = "Direction 4"
        text_x = 80; text_y = 100
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)
        cv2.rectangle(down_img, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)        
        panorama_segment.append(cv2.putText(down_img, "Direction 4", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 6, cv2.LINE_AA))
        
        # Add black borders between sub-images
        bordered_segments = []
        for segment in panorama_segment[:-1]:  # Exclude the last image from adding a border
            bordered_segments.append(segment)
            bordered_segments.append(np.zeros((segment.shape[0], 10, 3), dtype=np.uint8))  # 10-pixel wide black border

        bordered_segments.append(panorama_segment[-1])  # Add the last image without a border
        panoramic_image = np.concatenate(bordered_segments, axis=1)
        self.top_down_panorama_saver.save(panoramic_image)
        return panoramic_image, panorama_segment