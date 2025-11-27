import numpy as np
import cv2
import base64
import io
import os

from langchain_openai import ChatOpenAI
from utils.saver import image_saver
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image

class GPT:
    
    def __init__(self,config):

        self.config = config
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
    
    def text_request(self,request):

        message = [
            {
                "type": "text",
                "text": request
            },
        ]
        prompt = ChatPromptTemplate.from_messages(
            messages = [
                HumanMessagePromptTemplate.from_template(
                    message
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({}).content        

    def image_request(self,request,img_bgr:list):

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
        return chain.invoke({}).content