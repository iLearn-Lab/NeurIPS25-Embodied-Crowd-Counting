from Vision_models.GroundingDINO.modules.util.inference import load_model, load_image, predict, annotate
import Vision_models.GroundingDINO.modules.datasets.transforms as T
import cv2
from PIL import Image
import numpy as np


# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     traceback.print_stack()
#     log = file if hasattr(file,'write') else sys.stderr
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
# warnings.showwarning = warn_with_traceback
# warnings.simplefilter('always', DeprecationWarning)

def get_video():
    
    model = load_model(
        ".\Vision_models\GroundingDINO\config\GroundingDINO_SwinT_OGC.py", 
        ".\Vision_models\GroundingDINO\weights\groundingdino_swint_ogc.pth"
    )

    video_path = "path_to_video.mp4"
    cap = cv2.VideoCapture(video_path)

    IMAGE_PATH = "weights/demo.jpg"
    TEXT_PROMPT = "person.pigeon.tree."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)

        # 处理当前帧
        boxes, logits, phrases = predict(
            model=model,
            image=image_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite(f"annotated_image_{count}.jpg", annotated_frame)
        count += 1

    cap.release()
    cv2.destroyAllWindows()

    # image_source, image = load_image(IMAGE_PATH)
    #
    # boxes, logits, phrases = predict(
    #     model=model,
    #     image=image,
    #     caption=TEXT_PROMPT,
    #     box_threshold=BOX_TRESHOLD,
    #     text_threshold=TEXT_TRESHOLD
    # )
    #
    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # cv2.imwrite("annotated_image.jpg", annotated_frame)



























