GPT4_PROMPT = \
"""You are now a drone pilot, and you will control a drone to finish instructions. 

You will receive an instruction at the begining. There is a camera on the drone. You can control the camera to take observation about the scene. According to this observation, you can take your next action to finish the instruction. You can also move the drone to a new location, when you find that you can not find your target on your current location. Your initial location is (0,0,0), which corresponds to your (x,y,z) coordinate.

Follow the given format and use provided tools.
{tool_descriptions}

----
Starting below, you should follow this format:

Instruction: an instruction you need to follow
Plan: you need to output a general plan containing steps you need to do for this task
Thought: you should always think about what to do next and why
Action: the tool you use, must be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have finished the instruction, I can stop.
Final Answer: Finished!
----

Begin!

Instruction: {instruction}
Thought: """

CHECK_OBSTACLE_TOOL_NAME = "check_obstacle"
CHECK_OBSTACLE_TOOL__DESCRIPTION = \
"""Can be used to check if there are obstacles in front of you. The tool will tell you the distance between you and the obstacles. The distance is the maximum unit that you can move forward. You should always use this tool before using move_to_location tool. The tool will also take a picture about the current scene and save it for your next action.   
For example:
Action: check_obstacle
Action Input: "None".
"""

REQUEST_TOOL_NAME = "request"
# REQUEST_TOOL_DESCRIPTION = \
# """Can be used to request an answer to a specfic question from an AI about the last picture you took. You can ask the AI to assist you to finish some tasks. You need to input your question to this tool. The tool will send the picture and your question to the AI.
# For example:
# Action: request
# Action Input: "What is in this scene?".
# """
REQUEST_TOOL_DESCRIPTION = \
"""Can only be used to request object attribute or description according to the last picture you took. You can not to use it to find whether an object is in the scene.
For example:
Action: request
Action Input: "What is the color of the apple in this scene?".
Action: request
Action Input: "Describe the scene in details".
"""

MOVE_TO_LOCATION_TOOL_NAME = "move_to_location"
MOVE_TO_LOCATION_TOOL_DESCRIPTION = \
"""Can be used to move to a location. You need to input the location tuple to this tool. The tuple indicates the relative (x,y,z) coordinate you want to move to. The y and z should always be 0 unit, which means that you can only move forward. Use check_obstacle at fisrt and you need to decide the unit you move forward. The safe distance to the obstacles is 5 units. Think how far you should move step by step.
For example:
Action: move_to_location
Action Input: "(20,0,0)".
For example:
Action: move_to_location
Action Input: "(50,0,0)".
"""

ROTATE_TOOL_NAME = "rotate"
ROTATE_TOOL_DESCRIPTION = \
"""Can be used to rotate the drone, when you find obstacles within 10. The input should be from -180 to 180, with 0 to 180 you rotate clockwise and -180 to 0 you rotate anticlockwise. You may find a way after rotation.
For example:
Action: rotate
Action Input: "90".
For example:
Action: rotate
Action Input: "-90".
"""

TARGET_FIND_TOOL_NAME = "target_find"
TARGET_FIND_TOOL_DESCRIPTION = \
"""Can be used to check if a target is in the current scene. The tool will first rotate the drone for 45° clockwise, and then check if the target is in the camera sight after the rotation. The input to this tool should be the description of the target. 
For example:
Action: target_find
Action Input: "car".
"""