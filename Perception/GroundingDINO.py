from Vision_models.GroundingDINO.modules.util.inference import load_model, predict, annotate, predict_batch
import Vision_models.GroundingDINO.modules.datasets.transforms as T
from Vision_models.GroundingDINO.modules.util.utils import get_phrases_from_posmap
import cv2
import numpy as np
import torch
from PIL import Image
from utils.saver import image_saver
import math
from tqdm import tqdm

class GroundingDINO_detector:

    def __init__(self,config):

        self.config = config

        self.model = load_model(".\Vision_models\GroundingDINO\config\GroundingDINO_SwinT_OGC.py", 
        ".\Vision_models\GroundingDINO\weights\groundingdino_swint_ogc.pth"
        )

    def preprocess_image(self,image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    def inference_single_image(self, img_bgr, prompt, low=False):

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_transformed = self.preprocess_image(img_bgr)
        if low == False: box_threshold=self.config['box_threshold']
        else: box_threshold=self.config['low_box_threshold']

        boxes, logits, phrases = predict(
            model=self.model,
            image=img_transformed,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=self.config['text_threshold'],
            device=self.config['device'],
        )

        annotated_frame = annotate(image_source=img_rgb, boxes=boxes, logits=logits, phrases=phrases) 

        return {
            'boxes' : boxes, 
            'logits' : logits, 
            'phrases' : phrases,
            'annotated_frame' : annotated_frame
        }
    
    def inference_batch_images(self, img_bgrs : list, batch_size : int, prompt):

        total_batches = len(img_bgrs) // batch_size + (1 if len(img_bgrs) % batch_size != 0 else 0)
        result_boxes = []
        result_logits = []
        result_phrases = []
        with tqdm(total=total_batches, desc="Target detection: ") as pbar:  
            for i in range(0, len(img_bgrs), batch_size):  
                img_batch = img_bgrs[i:i + batch_size]
                images = torch.stack([self.preprocess_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in img_batch])
                boxes, logits, phrases = predict_batch(
                    model=self.model,
                    images=images,
                    caption=prompt,
                    box_threshold=self.config['box_threshold'],
                    text_threshold=self.config['text_threshold'],
                    device=self.config['device'],
                )
                result_boxes += boxes
                result_logits += logits
                result_phrases += phrases                
                pbar.update(1)

        annotated_frames = []
        for i in range(len(result_boxes)):
            img_rgb = cv2.cvtColor(img_bgrs[i], cv2.COLOR_BGR2RGB)
            boxes = result_boxes[i]
            logits = result_logits[i]
            phrases = result_phrases[i]
            annotated_frame = annotate(image_source=img_rgb, boxes=boxes, logits=logits, phrases=phrases) 
            annotated_frames.append(annotated_frame)

        return {
            'boxes' : result_boxes, 
            'logits' : result_logits, 
            'phrases' : result_phrases,
        },  annotated_frames      
   
    def draw_single_box_on_image(self, img_bgr,box,logit,phrase):

        box = box.unsqueeze(dim=0)
        logit = logit.unsqueeze(dim=0)
        phrase = [phrase]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        annotated_frame = annotate(image_source=img_rgb, boxes=box, logits=logit, phrases=phrase)
        return annotated_frame

    def target_location_phaser(self, detection):
        """Describe the target location in nature language."""

        coordinates = detection['boxes'].numpy()[0]
        center_w = coordinates[0]
        center_h = coordinates[1]

        if center_w < 0.5:
            loc_w = 'left'
        elif center_w > 0.5:
            loc_w = 'right'

        if center_h < 0.5:
            loc_h = 'above'
        elif center_h > 0.5:
            loc_h = 'bottom'

        description = "The location of the target is on " + \
                    loc_h + ' ' + loc_w + " ."

        return description
    
    def phrase_GD_box(self,box,image):

        result = dict()
        result['center_h'] = int(box[1] * image.shape[0])
        result['center_w'] = int(box[0] * image.shape[1])
        result['length_h'] = int(box[3]* image.shape[0])
        result['length_w'] = int(box[2] * image.shape[1])
        result['start_h'] = result['center_h'] - math.floor(result['length_h']/2) + 1
        result['end_h'] = result['start_h'] + result['length_h'] - 1
        result['start_w'] = result['center_w'] - math.floor(result['length_w']/2) + 1
        result['end_w'] = result['start_w'] + result['length_w'] -1
        result['area'] = box[3] * box[2]
        return result
    
    def phrase_GD_boxes(self,boxes,image):    

        results = []
        for item in boxes:
            result = self.phrase_GD_box(item,image)
            results.append(result)
        return results
    
    def get_box_areas(self, boxes : torch.tensor, img_bgr : np.ndarray):

        items = []
        for i in range(boxes.shape[0]):
            box = boxes[i,:]
            box = self.phrase_GD_box(box,img_bgr)
            cut = img_bgr[box['start_h']:box['end_h'],
                          box['start_w']:box['end_w'],
                          :]
            items.append(cut)
        return items
    
    def get_target_loc(self, box, current_loc, X, Y, Z, camera):
        
        x = X[camera][box['center_h']][box['center_w']].reshape(1,-1)
        y = Y[camera][box['center_h']][box['center_w']].reshape(1,-1)
        z = Z[camera][box['center_h']][box['center_w']].reshape(1,-1)
        point_cloud = np.concatenate((x,y,z),axis = 1)
        current_loc_np = np.array(current_loc).reshape(1,-1)
        horiz_distance = np.linalg.norm(point_cloud[:,0:2] - current_loc_np[:,0:2], axis=1)
        target_loc = tuple(np.squeeze(point_cloud))
        return horiz_distance, target_loc
    
def test_GD(config):
    
    from Perception.GroundingDINO import GroundingDINO_detector
    import cv2
    GD = GroundingDINO_detector(config['GroundingDINO'])
    from utils.saver import image_saver
    IS = image_saver(config['now'],config['Record_root'],'GroundingDINO')
    img_path = "屏幕截图 2024-10-28 115645.png"
    img_bgr = cv2.imread(img_path)
    items = GD.inference_single_image(img_bgr,prompt='person')
    IS.save(items['annotated_frame'])
    # boxes = items['boxes']
    # cut_saver = image_saver(config['now'],config['Record_root'],'cut')
    # resized_image = cv2.resize(img_bgr, (800,800), interpolation=cv2.INTER_AREA)
    # cuts = GD.get_box_areas(boxes,resized_image)
    # cut_saver.save_list(cuts)