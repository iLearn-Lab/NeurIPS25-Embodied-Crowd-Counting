GPT35_PROMPT = """As an intelligent agent, you will control a drone to approach a target. 

You will receive a target description.

There is a camera on the drone. Control the drone to make sure that you have find the target in the camera sight first, and then approach the target.
If you have approached the target, stop and output information using the format shown below.
If not, continue by using the tools provided.
Show your reasoning in the Thought section.

Follow the given format and use provided tools.
{tool_descriptions}

----
Starting below, you should follow this format:

Instruction: a description of the target you need to find
Thought: you should always think about what to do next and why
Action: the tool you use, must be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have approached the target, I can stop.
Final Answer: Finished!
----

Begin!

Instruction: {instruction}
Thought: """

IF_TARGET_TOOL_NAME = "if_target"
IF_TARGET_TOOL_DESCRIPTION = f'Can be used to check if a target is in the current camera sight when you are on a new position. For example:\nAction: if_target\nAction Input: "car".'

TARGET_FIND_TOOL_NAME = "target_find"
TARGET_FIND_TOOL_DESCRIPTION = f'Can be used to find the target if the target is not in the current camera sight. Only can be used after using if_target tool. The tool will first rotate the drone for 45°, and then check if the target is in the camera sight after the rotation.\nThe input to this tool should be the description of the target. For example:\nAction: target_find\nAction Input: "car".'

APPROACH_TARGET_TOOL_NAME = "approach_target"
APPROACH_TARGET_TOOL_DESCRIPTION = f'After finding the target in the current view, you need to approach the target using this tool. The input is the target you need to approach. For example:\nAction: approach_target\nAction Input: "car"'

POSITION_CHANGE_TOOL_NAME = "position_change"
POSITION_CHANGE_TOOL_DESCRIPTION = f'Use this tool to change to a new position. The input should be a target that you have already approached or found. For example:\nAction: position_change\nAction Input: "car"'