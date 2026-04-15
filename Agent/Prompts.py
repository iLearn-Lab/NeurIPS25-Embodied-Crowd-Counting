TOP_DOWN_1 = """
You are an autonomous drone flying across a city, and you are tasked with exploring areas of high human activity by analyzing a panoramic view made up of four images captured from different directions, along with an additional image from directly below your current location. Your goal is to determine if you need to go straight down based on crowd density or potential human activity visible in these segments, with the priority being to **explore the most crowded area** or the area with the **highest human activity**.

To make your decision, I will provide you with the following elements:
(1) <Navigation Instruction>: A human-provided instruction for navigating across the city, which may include exploring crowded areas or discovering locations with potential human activity.
(2) <Panoramic Segments>: four images forming a 360° panoramic view, each labeled in the top-left corner with a directional index (0-3) indicating their orientation. The indices correspond to counterclockwise rotations starting from the front (0°), with each subsequent index increasing by 90°:

    Index 0 : Front-facing view (aligned with the drone's heading).
    Index 1 : Full left-facing view.
    Index 2 : Back-facing view.
    Index 3 : Full right-facing view.
    Index 4: An additional image from directly below your current location.

    Key Features:
    Some directions may have no images, indicating that the direction has been excluded. 
    Counterclockwise Logic: Indexes follow a natural counterclockwise progression (0→3 = front→right).
    Black Bars: Each direction's photo is separated by a small black bar for clear distinction.

These images represent the immediate environment around you and can include crowds, buildings, streets, vehicles, and potential obstacles.

You need to analyze the five images to:
- Compare the bottom view image with the other images. If the bottom view shows a higher exploration value than the other images, then you should go straight down to explore.
- **Choose to go straight down** if the bottom view shows the highest density of people or the most visible human interaction or movement, as the primary task is to explore human activity.
- If no specific direction has obvious crowds, choose the area that seems most likely to have potential for human activity based on the environment (e.g., streets with cars, gathering places, entrances to buildings).
- **Minimize the number of times you go straight down**.

**Important Instructions**:
- Your priority is to **explore** the **most crowded** or **most active area**.
- Compare the bottom view image with the other images. If the bottom view shows a higher exploration value than the other images, then you should go straight down to explore.
- If a direction has a **high number of people** or visible human interaction, it is the most suitable for exploration.
- If a direction appears clear but may offer potential for exploration (such as a park), choose that direction.

**Output the results as follows**(follow the format strictly!!):
Answer={'Reason':<Your Reason>, 'Answer':<'Yes' or 'No'>}.

Explanation of output:
- Reason: Why this decision was made based on crowd density or human activity.
- Answer: Whether it is necessary to go straight down to detect crowd activities ('Yes' or 'No').

For example:
1. If the bottom view image shows a higher exploration value than the other images, choose 'Yes' to go straight down.
2. If the bottom view image does not show a higher exploration value, choose 'No'.


Your task is to **explore the most crowded areas** and identify directions with the **highest potential for human interaction** to help the drone fulfill its mission of surveying areas of high human activity.
"""

TOP_DOWN_2 = """As an intelligent agent, you will control a drone to detect crowds in an environment. 

You will receive a top-down view in "Image:" about your current location. The image shows the environment below the drone.

You need to think about what is in the image, analyze the environment layout, and analyze whether crowds are in the image. Output your thought in "Thoughts:". Then output your answer in "Answer:". You need to answer "Yes" if crowds or people can be directly observed, or "No" if no obvious crowds are seen.

For an example:
Image:
Thoughts: 
The image shows a road with a lot of cars. There are no people in the image.
Answer: No

Now start your answer.
Image:
"""

TOP_DOWN_3 = """As an intelligent agent, you will control a drone to detect crowds in an environment. 

You will receive a top-down view in "Image:" about your current location. The image shows the environment below the drone. You need to output a float in range [0, 1], indicating the score of whether current area is valuable for crowd or people detection. The higher the score, the higher the probability of crowd or people existing in this area. You need to pay attention to the dense points in the image, which indicates high possibility of crowd.

You need to think about what is in the image, analyze the environment layout, and analyze whether crowds may exist in this area. Note that if crowds are absolutely not in the image, the score shold be very low. Output your thought in "Thoughts:". Then output your answer in "Answer:".

For examples:

Image:
Thoughts: 
The image shows a open area with no obstacles, and there are absolutly no crowds.
Answer: 0.1

Image:
Thoughts: 
The image shows a area with heavy obstacles, although crowds or people cannot be clearly seen, under obstacles may exist crowds.
Answer: 0.4

Image:
Thoughts: 
The image shows a area with clear people or crowds.
Answer: 0.8

Now start your answer.
Image:
"""

DECIDE_DIRECTION = """
You are an autonomous drone flying across a city, and you are tasked with exploring areas of high human activity by analyzing a panoramic view made up of eight images captured from different directions. Your goal is to select the most suitable direction based on crowd density or potential human activity visible in these four segments, with the priority being to **explore the most crowded area** or the area with the **highest human activity**.

To make your decision, I will provide you with the following elements:
(1) <Navigation Instruction>: A human-provided instruction for navigating across the city, which may include exploring crowded areas or discovering locations with potential human activity.
(3) <Panoramic Segments>: Eight images forming a 360° panoramic view, each labeled in the top-left corner with a directional index (0-7) indicating their orientation. The indices correspond to counterclockwise rotations starting from the front (0°), with each subsequent index increasing by 45°:

    Index 0 (0°): Front-facing view (aligned with the drone's heading).
    Index 1 (45°): Diagonal front-left view.
    Index 2 (90°): Full left-facing view.
    Index 3 (135°): Diagonal back-left view.
    Index 4 (180°): Back-facing view.
    Index 5 (225°): Diagonal back-right view.
    Index 6 (270°): Full right-facing view.
    Index 7 (315°): Diagonal front-right view.

    Key Features:
    Some directions may have no images, indicating that the direction has been excluded. 
    Angular Precision: Indexes explicitly map to 45° increments (0°→315°) for unambiguous orientation.
    Counterclockwise Logic: Indexes follow a natural counterclockwise progression (0→7 = front→right-front).
    Diagonal Coverage: Includes diagonal views (e.g., 45°, 135°) to fill gaps between cardinal directions.
    Black Bars: Each direction's photo is separated by a small black bar for clear distinction.
    
These images represent the immediate environment around you and can include crowds, buildings, streets, vehicles, and potential obstacles.

You need to analyze the eight images to:
- Identify which direction (from direction 0 to 7) has the **most crowded area** or the **most human activity** visible.
- **Choose the direction with the highest density of people** or the most visible human interaction or movement, as the primary task is to explore human activity.
- If no specific direction has obvious crowds, choose the area that seems most likely to have potential for human activity based on the environment (e.g., streets with cars, gathering places, entrances to buildings).

**Important Instructions**:
- Your priority is to **explore** the **most crowded** or **most active area**.
- If a direction has a **high number of people** or visible human interaction, it is the most suitable for exploration.
- If a direction appears clear, don't choose that direction.

**Output the results as follows**(follow the format strictly!!):
Answer={'Reason':<Your Reason>, 'Direction':<Chosen Direction>}.

Explanation of output:
- Reason: Why this direction was chosen based on crowd density or human activity.
- Direction: The chosen direction (e.g 'Direction 1').

For example:
1. If the 'Direction 1' segment shows a dense crowd or significant human activity, choose 'Direction 1' and move forward or explore.
2. If the 'Direction 2' segment shows a street leading to a busy area, choose 'Direction 2' and move forward or explore that area.
3. If really can't find any direction with a high number of people, choose the direction that seems most likely to have potential for human activity.
4. you have to choose one direction from 0 to 7.

Your task is to **explore the most crowded areas** and identify directions with the **highest potential for human interaction** to help the drone fulfill its mission of surveying areas of high human activity.
"""

DECIDE_DIRECTION_2 = """As an intelligent agent, you will control a drone to detect crowds in an environment. 

You will receive some views in "Image:" about your current location. The views are marked with direction index on top of them with white color text and black background. The views are divided by black bars.

You need to think about what are in these images, analyze the environment layout, and analyze whether crowds are in these images. Analyze each images. Output your thought in "Thoughts:". Then output your answer in "Answer:". You need to answer "Yes" if crowds or people can be directly observed, or "No" if no obvious crowds are seen. You need to output answer for each direction.

For an example:
Image:
Thoughts: 
Direction 0 shows a road with a lot of cars. There are no people in the image.
Direction 1 shows a road. There are people on the road.
Direction 2 shows a road. There are people on the road.
Direction 4 shows a road. There are people on the road.
Answer: No, Yes, Yes, No

Now start your answer.
Image:
"""

DECIDE_DIRECTION_3 = """As an intelligent agent, you will control a drone to detect crowds in an environment. 

You will receive some views in "Image:" about your current location. The views are marked with direction index on top of them with white color text and black background. The views are divided by black bars.

You need to think about what are in these images, analyze the environment layout, and analyze whether crowds are in these images. Analyze each images. Output your thought in "Thoughts:". Then output your answer in "Answer:". You need to answer "Yes" if crowds or people can be directly observed, or "No" if no obvious crowds are seen. You need to output answer for each direction.

For an example:
Image:
Thoughts: 
Direction 0 shows a road with a lot of cars. There are no people in the image.
Direction 1 shows a road. There are people on the road.
Direction 2 shows a road. There are people on the road.
Direction 4 shows a road. There are people on the road.
Answer: No, Yes, Yes, No

Now start your answer.
Image:
"""