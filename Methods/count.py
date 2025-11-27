import warnings
import os
import numpy as np  
np.random.seed(42)
import random  
random.seed(42)
from Count.Count import count_GD, calculate_point_cloud, target_detection_GD, draw_gd_and_result, load_results, select_point_cloud_height, draw_targets, match_target_with_gd, target_detection_GL, count_GL, target_detection_PseCo, target_detection_MP, count_MP

warnings.filterwarnings("ignore")

def run_count(config):

    import shutil
    path = config['Record_root'] + config['Count']['record_path'] + 'count_config/'
    if not os.path.exists(path):  
        os.makedirs(path)
    shutil.copy2('Methods/countConfig.yml', path + 'Config.yml')       

    target_detection_GD(config)
    targets = count_GD(config)
    method_name = 'GD'

    # target_detection_MP(config)
    # count_MP(config)
    # method_name = 'MPcount'

    # target_detection_GL(config)
    # count_GL(config)
    # method_name = 'GL'

    targets, labels, change, ground_truth = load_results(config, method_name)
    from utils.saver import image_saver_plt
    saver = image_saver_plt(config['Count']['record_path'],config['Record_root'],'visualization/' + method_name)
    img = draw_gd_and_result(ground_truth, targets, config)
    saver.save(img)