import os
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from robomimic.scripts.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
import torch
from torchvision import transforms
import cv2
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import numpy as np
import time
from aloha_scripts.constants import FPS

from data_utils.datasets import set_seed
from llava_pythia.model import *
from einops import rearrange
import torch_utils as TorchUtils
import matplotlib.pyplot as plt
import sys
def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    return curr_image


def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp


def get_obs(deplot_env_obs, stats):
    # obs['front'], ['wrist_1'], ['state']
    cur_traj_data = dict()
    # (480, 270, 4)
    cur_right_rgb = deplot_env_obs['image']['21729895_left']
    cur_left_rgb = deplot_env_obs['image']['29392465_left']
    # [..., ::-1]
    # cur_front_rgb = cv2.cvtColor(cur_front_rgb, cv2.COLOR_BGRA2BGR)[..., ::-1]
    # cur_wrist_rgb = cv2.cvtColor(cur_wrist_rgb, cv2.COLOR_BGRA2BGR)[..., ::-1]

    cur_right_rgb = cv2.cvtColor(cur_right_rgb, cv2.COLOR_BGRA2BGR)
    cur_left_rgb = cv2.cvtColor(cur_left_rgb, cv2.COLOR_BGRA2BGR)

    # cur_front_rgb = cv2.cvtColor(cur_front_rgb, cv2.COLOR_BGRA2RGB)
    # cur_wrist_rgb = cv2.cvtColor(cur_wrist_rgb, cv2.COLOR_BGRA2RGB)
    cv2.imshow('cur_rgb', cv2.hconcat([cur_left_rgb, cur_right_rgb]))
    cv2.waitKey(1)
    cur_right_depth = np.zeros_like(cur_right_rgb) - 1.0
    cur_right_depth = cur_right_depth[..., :1]
    cur_left_depth = np.zeros_like(cur_left_rgb) - 1.0
    cur_left_depth = cur_left_depth[..., :1]

    cur_cartesian_position = np.array(deplot_env_obs['robot_state']['cartesian_position'])
    # cur_cartesian_position = pre_process(cur_cartesian_position, 'tcp_pose', stats)

    cur_gripper_position = np.expand_dims(np.array(deplot_env_obs['robot_state']['gripper_position']), axis=0)
    # cur_gripper_position = pre_process(cur_gripper_position, 'gripper_pose', stats)

    cur_state_np_raw = np.concatenate((cur_cartesian_position, cur_gripper_position))

    cur_state_np = pre_process(cur_state_np_raw, 'qpos', stats)

    # [128, 128, 3] np array
    right_rgb_img = cur_right_rgb  # deplot_env_obs['front']
    right_depth_img = cur_right_depth
    left_rgb_img = cur_left_rgb  # deplot_env_obs['wrist_1']
    left_depth_img = cur_left_depth
    cur_state = cur_state_np  # deplot_env_obs['state']
    cur_state = np.expand_dims(cur_state, axis=0)

    # [2, 1, 128, 128, 3]
    # [2, 480, 480, 3]
    traj_rgb_np = np.array([left_rgb_img, right_rgb_img])
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))
    # print(f'1. traj_rgb_np size: {traj_rgb_np.shape}')
    # l, n, c, h, w = traj_rgb_np.shape
    # traj_rgb_np = np.reshape(traj_rgb_np, (l, n*c, h, w))

    traj_depth_np = np.array([right_depth_img, left_depth_img])
    traj_depth_np = np.expand_dims(traj_depth_np, axis=1)
    traj_depth_np = np.transpose(traj_depth_np, (1, 0, 4, 2, 3))
    # print(f'1. traj_depth_np size: {traj_depth_np.shape}')
    # l, n, c, h, w = traj_depth_np.shape
    # traj_depth_np = np.reshape(traj_depth_np, (l, n*c, h, w))

    print("#"*50)
    print(traj_rgb_np.shape)
    traj_rgb_np = np.array([[cv2.cvtColor(np.transpose(img, (1,2,0)), cv2.COLOR_BGR2RGB) for img in traj_rgb_np[0]]])

    if im_size == 320: # resize to 320
        traj_rgb_np = np.array([[cv2.resize(img, (320, 180)) for img in traj_rgb_np[0]]])

    traj_rgb_np = np.transpose(traj_rgb_np, (0,1,4,2,3))
    return cur_state_np_raw, cur_state, traj_rgb_np, traj_depth_np


def time_ms():
    return time.time_ns() // 1_000_000


def convert_actions(pred_action):
    # pred_action = torch.from_numpy(actions)
    # pred_action = actions.squeeze(0)
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    # print(f'cur_xyz size: {cur_xyz.shape}')
    # print(f'cur_euler size: {cur_euler.shape}')
    # print(f'cur_gripper size: {cur_gripper.shape}')
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    # print(f'4. pred_action size: {pred_action.shape}')
    print(f'4. after convert pred_action: {pred_action}')

    return pred_action

class llava_pythia_act_policy:
    def __init__(self, policy_config, data_args=None):
        super(llava_pythia_act_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        # self.conv = conv_templates[policy_config['conv_mode']].copy()
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]

        # if not os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        #     print('transfer weights to non_lora_trainables.bin......')
        #     convert_zero_checkpoint_to_fp32_state_dict(model_path, os.path.join(model_path, 'non_lora_trainables.bin'))

        self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base,
                                                                                                    model_name, False,
                                                                                                    False)
        self.config = LlavaPythiaConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()

        if len(curr_image.shape) == 5: # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        # for k,v in sample.items():
        #     print(k, v.shape)
        image, image_r = torch.chunk(curr_image, 2, dim=0)

        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt', do_normalize=True, do_rescale=False,
                                              do_center_crop=False)['pixel_values']

        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)

        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(image_r, return_tensors='pt', do_normalize=True, do_rescale=False,
                                              do_center_crop=False)['pixel_values']
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        # print('raw_lang')
        inp = raw_lang
        assert image is not None, 'image must be provided.'
        # first message
        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(self.conv.roles[0], inp)
        image = None

        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)
        # print(input_ids.dtype, attn_mask.dtype, image_tensor.dtype, image_tensor_r.dtype, states.dtype)

        data_dict = dict(input_ids=input_ids,
                         attention_mask=attn_mask,
                         images=image_tensor,
                         images_r=image_tensor_r,
                         states=states)

        # print(f"@@@@@@@@@@@@@@@{image_tensor.shape}")
        return data_dict

    def expand2square(self, pil_imgs, background_color):
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)
        expanded_imgs = np.full((batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32)

        if height == width:
            expanded_imgs = pil_imgs.permute(0,2,3,1).cpu().numpy()
        elif height > width:
            offset = (max_dim - width) // 2
            # expanded_imgs[:, :height, offset:offset + width] = pil_imgs
            expanded_imgs[:, :height, offset:offset + width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        else:
            offset = (max_dim - height) // 2
            # expanded_imgs[:, offset:offset + height, :width] = pil_imgs
            expanded_imgs[:, offset:offset + height, :width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        expanded_imgs = torch.tensor(expanded_imgs).to(dtype=pil_imgs.dtype, device=pil_imgs.device) # B H W C
        return expanded_imgs


def eval_bc(policy, deploy_env,policy_config, save_episode=True, num_rollouts=1, raw_lang=None):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)

    # rand_crop_resize = (policy.config.act.policy_class == 'Diffusion')
    if policy_config["action_head"] == 'act':
        rand_crop_resize = False
        real_robot = False
        temporal_agg = True
        model_config = policy.config.act['act']
    else:
        rand_crop_resize = True
        real_robot = False
        temporal_agg = True
        model_config = policy.config.transformer_diffusion['transformer_diffusion']

    action_dim = model_config['action_dim']
    state_dim = model_config['state_dim']

    # ckpt = torch.load(ckpt_path)
    # print(ckpt)
    # loading_status = policy.load_state_dict(torch.load(ckpt_path))
    # print(policy)
    # loading_status = policy.load_state_dict(ckpt.module)

    # policy = ckpt.module
    # policy.to("cuda:0")
    policy.policy.eval()

    import pickle
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"] == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif policy_config["action_head"] == 'transformer_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    env = deploy_env
    env_max_reward = 0

    query_frequency = model_config['chunk_size']
    if temporal_agg:
        query_frequency = 1
        num_queries = model_config['chunk_size']
    if real_robot:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    max_timesteps = int(model_config['episode_len'] * 1)  # may increase for real-world tasks

    # num_rollouts = 1 #50
    episode_returns = []
    highest_rewards = []

    for rollout_id in range(num_rollouts):
        # if real_robot:
        #     e()

        rollout_id += 0
        # ts = env.reset()
        # obs, _ = env.reset()
        # obs = env.get_observation()
        # print(f'obs: {obs}')
        env.reset(randomize=False)

        print(f"env has reset!")

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim],dtype=torch.float16).cuda()
            # print(f'all_time_actions size: {all_time_actions.size()}')

        # robot_state_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        robot_state_history = np.zeros((max_timesteps, state_dim))
        image_list = []  # for visualization
        depth_list = []
        robot_state_list = []
        target_action_list = []
        rewards = []

        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(max_timesteps):
                time1 = time.time()
                time2 = time.time()
                control_timestamps = {"step_start": time_ms()}
                ### process previous timestep to get qpos and image_list
                # obs = ts.observation
                obs = deploy_env.get_observation()
                # print(f'obs: {obs}')

                # have processed robot_state
                cur_state_np_raw, robot_state, traj_rgb_np, traj_depth_np = get_obs(obs, stats)
                image_list.append(traj_rgb_np)
                depth_list.append(traj_depth_np)
                robot_state_history[t] = cur_state_np_raw

                robot_state = torch.from_numpy(robot_state).float().cuda()

                if t % query_frequency == 0:
                    curr_image = torch.from_numpy(traj_rgb_np / 255.0).float().cuda()
                    if rand_crop_resize:
                        print('rand crop resize is used!')
                        original_size = curr_image.shape[-2:]
                        ratio = 0.95
                        curr_image = curr_image[...,
                                     int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)

                # control_timestamps["policy_start"] = time_ms()
                if t == 0:
                    # warm up
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        policy.policy(**batch, eval=True)
                    print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if model_config['policy_class'] == "ACT_VLM":
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        all_actions = policy.policy(**batch, eval=True)
                            # print(f'all_actions size: {all_actions.size()}')
                        if real_robot:
                            all_actions = torch.cat(
                                [all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)

                    if temporal_agg:
                        print(f"all_actions: {all_actions.size()}")
                        print(f"all_time_actions: {all_time_actions.size()}")
                        print(f"t: {t}, num_queries:{num_queries}")
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif model_config['policy_class'] == "transformer_diffusion":
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        all_actions = policy.policy(**batch, eval=True)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:
                            all_actions = torch.cat(
                                [all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                            
                    if temporal_agg:
                        print(f"all_actions: {all_actions.size()}")
                        print(f"all_time_actions: {all_time_actions.size()}")
                        print(f"t: {t}, num_queries:{num_queries}")
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif model_config['policy_class'] == "CNNMLP":
                    # raw_action = policy(qpos, curr_image)
                    raw_action = policy(robot_state, curr_image)
                    all_actions = raw_action.unsqueeze(0)
                else:
                    raise NotImplementedError

                time4 = time.time()
                print(f"raw action size: {raw_action.size()}")
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                print(f"after post_process action size: {action.shape}")
                # target_qpos = action

                action = convert_actions(action)
                time5 = time.time()
                ### step the environment
                # ts = env.step(action)
                print(f'step {t}, pred action: {action}')
                # obs, rew, done, truncated, info = env.step(action)
                action_info = deploy_env.step(action)

                ### for visualization
                robot_state_list.append(robot_state)
                target_action_list.append(action)
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                # print(sleep_time)
                time.sleep(sleep_time)
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(
                        f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()

        if real_robot:
            # move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            # save qpos_history_raw
            # log_id = get_auto_index(ckpt_dir)
            # np.save(os.path.join(ckpt_dir, f'robot_state_{log_id}.npy'), robot_state_history)
            plt.figure(figsize=(10, 20))
            # plot qpos_history_raw for each qpos dim using subplots
            for i in range(state_dim):
                plt.subplot(state_dim, 1, i + 1)
                plt.plot(robot_state_history[:, i])
                # remove x axis
                if i != state_dim - 1:
                    plt.xticks([])
            plt.tight_layout()
            # plt.savefig(os.path.join(ckpt_dir, f'robot_state_{log_id}.png'))
            plt.close()

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    return

if __name__ == '__main__':
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'transformer_diffusion'
    model_size = '70M'
    policy_config = {
        "model_path": f"/media/eai/WJJ1T/droid/{action_head}_results/pythia_{model_size}/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-pp_tennis_new_lora_all/checkpoint-4000",
        # "model_path": "/media/eai/WJJ1T/droid/results/pythia_410M/vanilla_pythia_37k_pretrain_lora/checkpoint-10000",
        "model_base": f"/media/eai/WJJ1T/droid/results/pythia_{model_size}/model_base/pythia_{model_size}/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune",
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": action_head,
    }
    global im_size
    im_size = 320 # default 480
    # raw_lang = 'pick up the bread and put it into the plate.'
    raw_lang = 'put the tennis ball on the right side into the tennis bucket'
    # raw_lang = "pick up the bread"
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    policy = llava_pythia_act_policy(policy_config)
    sys.path.insert(0, "/home/eai/Dev-Code/droid")
    from droid.robot_env import RobotEnv

    from pynput import keyboard

    policy_timestep_filtering_kwargs = {'action_space': 'cartesian_position', 'gripper_action_space': 'position',
                                        'robot_state_keys': ['cartesian_position', 'gripper_position',
                                                             'joint_positions']}
    # resolution (w, h)
    # todo H W or W H?

    policy_camera_kwargs = {
        'hand_camera': {'image': True, 'concatenate_images': False, 'resolution': (480, 270), 'resize_func': 'cv2'},
        'varied_camera': {'image': True, 'concatenate_images': False, 'resolution': (480, 270), 'resize_func': 'cv2'}}

    deploy_env = RobotEnv(
        action_space=policy_timestep_filtering_kwargs["action_space"],
        gripper_action_space=policy_timestep_filtering_kwargs["gripper_action_space"],
        camera_kwargs=policy_camera_kwargs
    )

    deploy_env._robot.establish_connection()
    deploy_env.camera_reader.set_trajectory_mode()

    eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=raw_lang)

    print()
    exit()

