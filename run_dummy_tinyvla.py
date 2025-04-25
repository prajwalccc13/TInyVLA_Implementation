import numpy as np
import torch
import pickle
from pathlib import Path

class DummyEnv:
    def __init__(self):
        self.step_count = 0

    def get_observation(self):
        obs = {
            'images': {
                'cam_1': np.zeros((270, 480, 3), dtype=np.uint8),
                'cam_2': np.zeros((270, 480, 3), dtype=np.uint8)
            }
        }
        return obs

    def step(self, action):
        self.step_count += 1
        print(f"[Dummy Step] Step #{self.step_count}, Action: {action}")
        return {}

    def reset(self, randomize=False):
        print("[DummyEnv] Reset called")
        self.step_count = 0

def get_obs(obs, stats):
    cam_1 = obs['images']['cam_1']
    cam_2 = obs['images']['cam_2']
    img = np.stack([cam_1, cam_2], axis=0)  # 2, H, W, C
    robot_state = np.zeros(10)  # fake robot state vector
    return img, robot_state

def create_dummy_stats(path):
    dummy_stats = {
        'action_mean': np.zeros(10),
        'action_std': np.ones(10),
        'action_min': np.zeros(10),
        'action_max': np.ones(10)
    }
    with open(path, 'wb') as f:
        pickle.dump(dummy_stats, f)
    print(f"[INFO] Dummy stats saved at: {path}")

if __name__ == "__main__":
    stats_path = Path("dataset_stats.pkl")
    if not stats_path.exists():
        create_dummy_stats(stats_path)

    # Example usage
    env = DummyEnv()
    env.reset()
    obs = env.get_observation()
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    img, state = get_obs(obs, stats)
    print(f"[Dummy Data] Image shape: {img.shape}, State shape: {state.shape}")
