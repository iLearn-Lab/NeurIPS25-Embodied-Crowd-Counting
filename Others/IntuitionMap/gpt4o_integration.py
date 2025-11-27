import openai
import base64
import os
import cv2
import numpy as np
from utils.saver import image_saver, image_saver_plt
from Point_cloud.Map_element import UNKNOWN, OBSTACLE, EXPLORED
from Agent.Prompts2 import DECIDE_DIRECTION, TOP_DOWN_2, TOP_DOWN_3, DECIDE_DIRECTION_2
import re
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from PIL import Image
import io
from langchain_openai import ChatOpenAI

class LLMJudge:
    
    def __init__(self, config):
        openai.api_key = config['LLMJudge']['OPENAI_API_KEY']
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
    
    def get_useless_directions(self, frontiers, location, yaw, global_2D_map):
        # Construct a 2D rotation matrix based on the yaw angle
        yaw_rad = np.deg2rad(-yaw)
        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)]
        ])
        clocation = np.array([location[1], location[0]], dtype = np.int32)
        total_directions = [0, 1, 2, 3, 4, 5, 6, 7]
        need_directions = []
        '''计算location和每一个frontier的中心点逆时针方向的夹角'''
        for frontier in frontiers:
            '''能否看到这个frontier'''
            directions = np.zeros(frontier['centers'].shape[0]) - 1
            for i, center in enumerate(frontier['centers']):
                if self.check_visible(center, location, global_2D_map[OBSTACLE], global_2D_map[EXPLORED]) == False: 
                    frontier['directions'] = directions
                    continue
                center = np.array([center[1], center[0]], dtype = np.int32)
                unit_v1 = (center-clocation) / np.linalg.norm(center-clocation)
                unit_v2 = rotation_matrix.dot(np.array([0, 1]))
                dot_product = np.dot(unit_v1, unit_v2)
                angle = np.arccos(dot_product)
                angle = angle / np.pi * 180
                '''判断v1在v2的左边还是右边'''
                if np.cross(unit_v1, unit_v2) > 0: angle = 360 - angle
                '''判断angle在哪个方向'''
                angle //= 45
                directions[i] = int(angle)
                if angle not in need_directions: need_directions.append(int(angle))
            frontier['directions'] = directions
            
        return list(set(total_directions) - set(need_directions))
    
    def check_visible(self, center, location, obstacle_location, explore_location):
        location = np.array(location, dtype = np.int32)
        obstacle_location = obstacle_location.tolist()
        explore_location = explore_location.tolist()
        
        
        for i in range(-4,5):
            for j in range(-4,5):
                tmp = center + np.array([i, j])
                tmp = tmp.tolist()
                if tmp not in explore_location: continue
                line = self.get_line(location, center + np.array([i, j]))
                '''检查line上的点是否有障碍物'''
                flg = True
                for point in line:
                    point = point.tolist()
                    if point in obstacle_location: flg = False
                if flg: return True
        return False
    
    def get_line(self, start, current):
        if current[0] - start[0] == 0:
            k = 0
        else:
            k = (current[1] - start[1]) / (current[0] - start[0])
        i_residual = current[0] - start[0]
        if i_residual > 0:
            i_line1 = np.arange(0,i_residual+1,1)
        else:
            i_line1 = np.arange(i_residual,1,1)
        j_line1 = (k * i_line1).astype(np.int32)

        i_line1 = i_line1 + start[0]
        j_line1 = j_line1 + start[1]
        
        return np.concatenate((i_line1.reshape(-1,1), j_line1.reshape(-1,1)), axis = 1).astype(np.int32)
        
# Example usage of GPT4ImageIntegration class
def example_usage():
    images_directory = r'Record\\2025-03-02-17_53_30\\LLMJudge\\panorama\\0.png'  # Folder where your images are saved
    img = cv2.imread(images_directory)
    gpt_integration = LLMJudge()

    result = gpt_integration.LLMResult(img)
    print(result)

if __name__ == '__main__':
    example_usage()