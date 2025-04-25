import numpy as np
import torch
import time
import pickle
from pathlib import Path

# Constants
MAX_TIMESTEPS = 5
ACTION_DIM = 10
FPS = 10  # simulate control frequency
DT = 1 / FPS

# Dummy environment class
class DummyEnv:
    def __init__(self):
        self.step_count = 0

    def get_observation(self):
        obs = {
            'images': {
                'cam_1': np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8),
                'cam_2': np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
            }
        }
        return obs

    def step(self, action):
        self.step_count += 1
        print(f"[Step {self.step_count}] Action taken: {action}")
        return {}

    def reset(self, randomize=False):
        print("[ENV RESET]")
        self.step_count = 0

# Fake get_obs function
def get_obs(obs, stats):
    cam_1 = obs['images']['cam_1']
    cam_2 = obs['images']['cam_2']
    img = np.stack([cam_1, cam_2], axis=0)
    robot_state = np.random.randn(10)
    return img, robot_state

# Post-processing function (mocked)
def post_process_action(raw_action, stats):
    return raw_action * stats['action_std'] + stats['action_mean']

# Dummy policy (no actual model)
class DummyPolicy:
    def __init__(self):
        pass

    def process_batch_to_llava(self, image, state, lang):
        return None  # Not used in dummy mode

    def predict(self, image, state, lang):
        return np.random.randn(ACTION_DIM)  # dummy action

# Main eval function
def eval_bc(policy, env, stats, raw_lang="put the mug on the tray"):
    env.reset()
    time0 = time.time()
    delay_sum = 0

    for t in range(MAX_TIMESTEPS):
        obs = env.get_observation()
        image, robot_state = get_obs(obs, stats)

        # Convert inputs (mocked)
        robot_state_tensor = torch.from_numpy(robot_state).float()
        image_tensor = torch.from_numpy(image / 255.0).float()

        # Predict action
        raw_action = policy.predict(image_tensor, robot_state_tensor, raw_lang)
        action = post_process_action(raw_action, stats)

        # Print and step
        env.step(action)

        # Maintain FPS timing
        duration = time.time() - time0
        sleep_time = max(0, DT - duration)
        time.sleep(sleep_time)
        if duration > DT:
            delay_sum += (duration - DT)
            print(f"[WARNING] Step took {duration:.3f}s (expected {DT:.3f}s)")

        time0 = time.time()

# Entry point
if __name__ == "__main__":
    stats_path = Path("dataset_stats.pkl")
    if not stats_path.exists():
        dummy_stats = {
            'action_mean': np.zeros(10),
            'action_std': np.ones(10),
            'action_min': np.zeros(10),
            'action_max': np.ones(10)
        }
        with open(stats_path, 'wb') as f:
            pickle.dump(dummy_stats, f)

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    env = DummyEnv()
    policy = DummyPolicy()
    eval_bc(policy, env, stats)
