from Vision_models.GeneralizedLoss.models.vgg import vgg19
from Vision_models.GeneralizedLoss.datasets.crowd import Crowd
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class gereralizedloss:

    def __init__(self):

        self.model = vgg19()
        self.device = torch.device('cuda')
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('Vision_models/GeneralizedLoss/model.pth', self.device))  

    def inference_images(self, img_list : list):

        datasets = Crowd('Vision_models/GeneralizedLoss/data/test', 
                         512, 8, is_gray=False, method='val', im_list= img_list)
        dataloader = torch.utils.data.DataLoader(datasets, 8, shuffle=False,
                                             num_workers=1, pin_memory=False)
        outputs = []      
        for inputs in tqdm(dataloader,desc="Calculating density map: "):
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                output = self.model(inputs)
                output = output.cpu().numpy().squeeze()
                if len(output.shape) == 2:
                    outputs.append(output)
                else:
                    for item in output:
                        outputs.append(item)
        return outputs

    def draw_density_maps(self, density_maps):

        imgs = []
        for item in density_maps:
            plt.figure()
            plt.imshow(item, cmap='viridis', interpolation='nearest')  
            plt.colorbar()  # 显示颜色条 
            imgs.append(plt.gca())
        return imgs       

    def show_cam_on_image(self, img, density_map):
        img = cv2.resize(img, (266,266), interpolation=cv2.INTER_LINEAR)
        density_map = cv2.resize(density_map, (266,266), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.uint8(255 * density_map), cv2.COLORMAP_JET)
        # heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam