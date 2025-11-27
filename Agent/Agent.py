from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.prompts import PromptTemplate

import os
import numpy as np
import airsim
import math

from Agent.Prompts import (GPT35_PROMPT,
                     IF_TARGET_TOOL_NAME,
                     IF_TARGET_TOOL_DESCRIPTION,
                     TARGET_FIND_TOOL_NAME,
                     TARGET_FIND_TOOL_DESCRIPTION,
                     APPROACH_TARGET_TOOL_NAME,
                     APPROACH_TARGET_TOOL_DESCRIPTION,
                     POSITION_CHANGE_TOOL_NAME,
                     POSITION_CHANGE_TOOL_DESCRIPTION)

os.environ["OPENAI_API_KEY"] = ""

from Drone.Control import drone_controller
from Perception.GroundingDINO import GroundingDINO_detector

class Drone_Agent:

    def __init__(self, config):

        self.config = config

        self.drone_controller = drone_controller(config['drone_config'])
        self.detectation = GroundingDINO_detector(config['GroundingDINO_config'],config['now'])
        
        self.create_agent(config['llm_config'])

    def create_agent(self, config):
            
        llm = ChatOpenAI(
            temperature=config['temperature'],
            model_name=config['model_name']
        )

        if_target = self.create_if_target_tool()
        target_find = self.create_target_find_tool()
        approach_target = self.create_approach_target_tool()
        position_change = self.create_position_change_tool()
        tools = [if_target,
                 target_find,
                 approach_target,
                 position_change]

        prompt = PromptTemplate(
            template=GPT35_PROMPT,
            input_variables=["instruction"],
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in tools]
                ),
            },
        )

        self.agent = ZeroShotAgent(
            llm_chain=LLMChain(llm=llm, prompt=prompt),
            allowed_tools=[tool.name for tool in tools],
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
                agent=self.agent, 
                tools=tools, 
                verbose=True, 
                handle_parsing_errors = True,
                return_intermediate_steps=True,
                max_iterations=100,
        )

    def action(self):

        # take off
        self.drone_controller.takeoff()

        # Question
        output = self.agent_executor(self.config['Prompt_config']['insturction'])

    def create_if_target_tool(self) -> Tool:    
        """Create a tool to find if a target is exist."""

        def _make_action(*args, **kwargs) -> str:

            # get the llm input 
            prompt = args[0].strip(" ").strip('"').strip("'")

            # first take a picture
            img_bgr = self.drone_controller.get_single_bgr_image() 
            
            # find if the target is in the picture
            boxes = self.perception.inference_single_image(img_bgr,prompt)['boxes']

            if boxes.shape[0] == 0:
                return "Can not find " + prompt + "."            
            else:
                return "The target is in the current view."
            
        return Tool(
            name=IF_TARGET_TOOL_NAME,
            func=_make_action,
            description=IF_TARGET_TOOL_DESCRIPTION,
        )

    def create_target_find_tool(self) -> Tool:    
        """Create a tool to adjust the camera and find a target."""

        def _make_action(*args, **kwargs) -> str:

            # get the llm input 
            prompt = args[0].strip(" ").strip('"').strip("'")

            yaw_rate = self.config['drone_config']['yaw_rate']
            rotated = 0
            while(1):
                self.drone_controller.client.rotateByYawRateAsync(yaw_rate = yaw_rate, duration = 45 / yaw_rate).join()
                rotated += 45
                _,box = self.watch_and_inference(prompt,if_argmax=False)
                if box.shape[0] != 0:
                    self.drone_controller.client.hoverAsync().join()
                    return "The " + prompt + " is in the current view."
                if rotated == 360:
                    self.drone_controller.client.hoverAsync().join()
                    result = "Can not find " + prompt + ". On this position you can not find this target, try to change your position."
                    return result 
                          
        return Tool(
            name=TARGET_FIND_TOOL_NAME,
            func=_make_action,
            description=TARGET_FIND_TOOL_DESCRIPTION,
        )
    
    def create_approach_target_tool(self) -> Tool:
        """Create a tool to get to the target."""

        def _make_action(*args, **kwargs) -> str:

            # get the llm input 
            prompt = args[0].strip(" ").strip('"').strip("'")

            self.approach_target(prompt)

            return "You are now at " + prompt + "."
        
        return Tool(
            name=APPROACH_TARGET_TOOL_NAME,
            func=_make_action,
            description=APPROACH_TARGET_TOOL_DESCRIPTION,
        )
    
    def create_position_change_tool(self) -> Tool:
        """Create a tool to change position."""

        def _make_action(*args, **kwargs) -> str:

            # get the llm input 
            prompt = args[0].strip(" ").strip('"').strip("'")

            self.change_view(prompt)

            return "You have changed to a new position."
        
        return Tool(
            name=POSITION_CHANGE_TOOL_NAME,
            func=_make_action,
            description=POSITION_CHANGE_TOOL_DESCRIPTION,
        )
        
    def watch_and_inference(self,prompt,camera = 0, if_argmax = True, if_save = True):

        img_bgr = self.drone_controller.get_single_bgr_image(camera)           
        detection = self.perception.inference_single_image(img_bgr,prompt,if_save=if_save) 
        logits = detection['boxes'].numpy()
        boxes = detection['boxes'].numpy()


        if if_argmax:
            idx = np.where(logits == np.max(logits))[0][0]
            boxes = boxes[idx]
            return logits[idx], boxes
        else:
            return logits, boxes
        
    def horizen_rotate_adjust(self, box, camera_name = 0):
        '''Yaw to adjust the drone so that the target horizen is in the center of the view. Threshold is to control the time to adjust. coordinates[0]:w coordinates[1]:h'''

        # get fov
        fov = self.drone_controller.client.simGetCameraInfo(camera_name=camera_name).fov
        # horizen yaw
        yaw = (0.5-box[0]) * fov

        if yaw < 0:
            yaw_rate = self.config['drone_config']['yaw_rate']
        else:
            yaw_rate = -1 * self.config['drone_config']['yaw_rate']

        # horizen rotate
        self.drone_controller.client.rotateByYawRateAsync(yaw_rate = yaw_rate, duration = abs(yaw/self.config['drone_config']['yaw_rate'])).join()
    
    def vertical_adjust(self, box, prompt, threshold, camera_name = 0):
        '''Adjust the drone so that the target vertical is no lower than a threshold. coordinates[0]:w coordinates[1]:h'''

        threshold_up = threshold
        threshold_down = 1 - threshold

        # vertical move
        # the target is above the camera
        if box[1]<0.5:
            vz = -1 * self.config['drone_config']['z_speed']
        elif box[1]>0.5:
            vz = self.config['drone_config']['z_speed']
        
        # self.drone_controller.speed_change(0,self.config['drone_config']['z_speed'],2,'vz')
        # self.drone_controller.client.moveByVelocityAsync(0, 0, vz=vz, duration=math.inf)
        self.drone_controller.start_moveByVelocityBodyFrameAsync(direction='z',duration=math.inf,speed=vz,if_join=False)
        while(1):         
            _,box = self.watch_and_inference(prompt,camera_name)   
            if box[1] > threshold_up or box[1] < threshold_down:
                self.drone_controller.stop(direction='z')
                return
            
    def horizen_move_adjust(self, box, prompt, threshold, camera_name = 0):

        threshold_left = threshold
        threshold_right = 1 - threshold

        vy = self.config['drone_config']['y_speed']

        self.drone_controller.start_moveByVelocityBodyFrameAsync(direction='y',duration=math.inf,speed=vy,if_join=False)
        while(1):         
            _,box = self.watch_and_inference(prompt,camera_name)   
            if box[0] < threshold_left or box[0] > threshold_right:
                self.drone_controller.stop(direction='y')
                return
                
    def track(self,prompt):

        drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
        while(1):
            
            # first take a picture
            img_bgr = self.drone_controller.get_single_bgr_image()
            
            # find if the target is in the picture
            detection = self.perception.inference_single_image(img_bgr,prompt)
            boxes = detection['boxes'].numpy()[0]

            yaw = self.yaw_to_target_horizen(boxes, camera_name = 0)
            yaw_mode = airsim.YawMode(True,  -1 * yaw)

            self.drone_controller.client.moveByVelocityZAsync(0, 5, self.config['drone_config']['takeoff_height'], duration=1, drivetrain = drivetrain, yaw_mode= yaw_mode)
    
    def approach_target(self,prompt,
                        threshold_left = 0.3,
                        threshold_up = 0.3,
                        box_size_width_threshold = 0.1,
                        box_size_height_threshold = 0.1,
                        ):
        
        def _check_if_adjust(box, threshold_left,threshold_up):
        
            threshold_right = 1 - threshold_left
            threshold_down = 1 - threshold_up

            horizen_adjust = False
            vertical_adjust  = False
            if box[0] <= threshold_left or box[0] >= threshold_right:
                horizen_adjust  = True

            if box[1] <= threshold_up or box[1] >= threshold_down:
                vertical_adjust  = True

            return horizen_adjust, vertical_adjust
        
        while(1):
                       
            _,box = self.watch_and_inference(prompt)
            
            horize_adjust, vertical_adjust = _check_if_adjust(box,threshold_left,threshold_up)

            if horize_adjust or vertical_adjust:

                self.drone_controller.stop(direction='x')

                if horize_adjust:
                    _,box = self.watch_and_inference(prompt)
                    self.horizen_rotate_adjust(box)
                if vertical_adjust:
                    _,box = self.watch_and_inference(prompt)
                    self.vertical_adjust(box,prompt,threshold=threshold_up)

            if box[2] > box_size_height_threshold and box[3]> box_size_width_threshold:
                self.drone_controller.stop(direction='x')
                return
                    
            self.drone_controller.start_moveByVelocityBodyFrameAsync(direction='x',duration=math.inf,if_join=False)

    def change_view(self,prompt,threshold=0.1):

        _,box = self.watch_and_inference(prompt)
        self.horizen_rotate_adjust(box)

        _,box = self.watch_and_inference(prompt)
        self.horizen_move_adjust(box,prompt,threshold)

    def take_photos(self):
        '''Make single step action to adjust the camera.'''

        prompt = self.config['Prompt_config']['target_prompt']

        # take off
        self.drone_controller.takeoff()

        while(1):
            self.approach_target(prompt)
            self.change_view(prompt)



















        # # get the depth img
        # img_depth = self.drone_controller.get_single_depth_image()

        # # get the target range
        # coordinates = boxes.numpy()[idx]
        # coordinate_w = int(coordinates[0] * img_depth.shape[1])
        # coordinate_h = int(coordinates[1] * img_depth.shape[0])

        # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(img_rgb)
        # draw = ImageDraw.Draw(image)

        # image = Image.fromarray(img_depth)
        # image = image.convert('RGB')
        # draw = ImageDraw.Draw(image)
        # draw.ellipse((coordinate_w-10, coordinate_h-10, coordinate_w+10, coordinate_h+10), fill="red")
        # image.save('test.jpg')

        # depth = img_depth[coordinate_h][coordinate_w]

        # self.drone_controller.client.moveToPositionAsync(0, -int(depth), -15, 2).join()

        # img_bottom_bgr = self.drone_controller.get_single_bgr_image(camera=3)
        # cv2.imwrite('bottom.png', img_bottom_bgr)

        # return "The target is "

                # self.drone_controller.client.rotateToYawAsync(45, timeout_sec = 3e+38, margin = 1, vehicle_name = '')

        # self.drone_controller.client.moveByRollPitchYawrateZAsync(roll=0, pitch=1, yaw_rate=0, z=-15, duration=5, vehicle_name = '')

        # idx = np.where(logits == np.max(logits))[0][0]

        # kinematic_state_groundtruth = self.drone_controller.client.simGetGroundTruthKinematics(vehicle_name='')
           
        
        # (_, _, yaw_base) = airsim.to_eularian_angles(kinematic_state_groundtruth.orientation)
        # yaw_base = yaw_base * 180 / math.pi

        # self.turn_to_target(boxes.numpy()[idx],prompt)