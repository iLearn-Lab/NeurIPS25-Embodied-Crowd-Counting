import Methods as m
import yaml
from utils.logger import CompleteLogger
import warnings
import numpy as np  
np.random.seed(42)
import random  
random.seed(42)
warnings.filterwarnings("ignore")
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='choose the method')
    parser.add_argument('--method', type=str,  
                        default = 'count',
                        # default='OurMethod',
                        help='choose the method: FBE, FBEWithDG, OurMethod')
    args = parser.parse_args()
    method = args.method

    config = open('./Methods/' + method + 'Config.yml', 'rb')
    config = yaml.safe_load(config)

    logger = CompleteLogger("Log")
    config['now'] = logger.now
    
    match method:
        case 'FBE':
            from Methods.FBE import run_fbe
            run_fbe(config)
        case 'FBEWithDG':
            from Methods.FBEWithDG import run_fbe_with_density_guide
            run_fbe_with_density_guide(config)
        case 'OurMethod':
            from Methods.OurMethod import run_our_method
            run_our_method(config)
        case 'count':
            from Methods.count import run_count
            run_count(config)       
        case _:
            raise ValueError("Invalid method selected. Please choose from FBE, FBEWithDG, or OurMethod.")