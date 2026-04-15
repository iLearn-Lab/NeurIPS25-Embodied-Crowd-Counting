import Methods as m
import yaml
from utils.logger import CompleteLogger
import warnings
import numpy as np  
np.random.seed(42)
import random  
random.seed(42)
warnings.filterwarnings("ignore")
from pathlib import Path
import shutil
import subprocess
import time


def load_yaml_file(path: Path):
    with path.open('r', encoding='utf-8') as file:
        return yaml.safe_load(file) or {}


def get_dataset_config(config):
    dataset_config = config.get('Dataset')
    if not isinstance(dataset_config, dict):
        raise KeyError("Config.yml must define a Dataset mapping.")

    dataset_path = dataset_config.get('path', dataset_config.get('Path'))
    if dataset_path:
        dataset_root = Path(dataset_path)
    else:
        dataset_name = dataset_config.get('name', dataset_config.get('Name'))
        dataset_id = dataset_config.get('id', dataset_config.get('Id'))
        if dataset_name is None or dataset_id is None:
            raise KeyError("Dataset config must include 'path' or both 'name' and 'id'.")
        dataset_root = Path('Dataset') / str(dataset_name).upper() / str(dataset_id)

    return dataset_config, dataset_root


def get_method_name(config):
    method_config = config.get('Method')
    if isinstance(method_config, dict):
        method_name = method_config.get('name', method_config.get('Name'))
    else:
        method_name = method_config

    if not method_name:
        raise KeyError("Config.yml must define Method.name or Method.")

    return str(method_name)


def build_simulator_config(dataset_root: Path):
    project_root = dataset_root / 'WindowsNoEditor' / 'ECC' / 'Project'
    return {
        'route_path': str(project_root / 'Route' / 'route.csv'),
        'ground_truth_path': str(project_root / 'Location' / 'man' / 'location.csv'),
        'navigation_path': str(project_root / 'Navigation' / 'navigation.csv'),
    }


def sync_airsim_settings():
    source_path = Path('Configs') / 'settings.json'
    target_dir = Path.home() / 'Documents' / 'AirSim'
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_dir / 'settings.json')


def resolve_dataset_exe(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / 'WindowsNoEditor' / 'ECC.exe',
        dataset_root / 'WindowsNoEditor' / 'ECC' / 'Binaries' / 'Win64' / 'ECC-Win64-Shipping.exe',
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find executable under {dataset_root}. Expected ECC.exe or ECC-Win64-Shipping.exe."
    )


def launch_dataset_exe(exe_path: Path):
    return subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))


def close_process_tree(process):
    if process is None:
        return

    if process.poll() is None:
        subprocess.run(
            ['taskkill', '/PID', str(process.pid), '/T', '/F'],
            check=False,
            capture_output=True,
            text=True,
        )


def run_method(method: str, config: dict):
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
        case _:
            raise ValueError("Invalid method selected.")
        
    count_config_path = Path('Configs') / f'CountConfig.yml'
    count_config = load_yaml_file(count_config_path)
    count_config['now'] = config['now']
    from Methods.count import run_count
    run_count(count_config)

if __name__ == '__main__':

    root_config_path = Path('Config.yml')
    root_config = load_yaml_file(root_config_path)
    dataset_config, dataset_root = get_dataset_config(root_config)
    method = get_method_name(root_config)

    sync_airsim_settings()

    exe_path = resolve_dataset_exe(dataset_root)
    simulator_process = launch_dataset_exe(exe_path)

    try:
        startup_wait_seconds = dataset_config.get('startup_wait_seconds', 10)
        if startup_wait_seconds > 0:
            time.sleep(startup_wait_seconds)

        method_config_path = Path('Configs') / f'{method}Config.yml'
        config = load_yaml_file(method_config_path)

        logger = CompleteLogger('Log')
        config['now'] = logger.now
        config['Dataset'] = dataset_config
        config['Simulator'] = build_simulator_config(dataset_root)
        
        run_method(method, config)
    finally:
        close_process_tree(simulator_process)