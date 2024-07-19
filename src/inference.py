import argparse
import collections
import os
import time

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

from pushTEnvOriginal import PushTImageEnv
from pushTImageDataset import get_stats, normalize_data, unnormalize_data
from train_ddp import get_nets


def main(num_diffusion_iters, scheduler_type, model_path, max_inference_steps, seed, output_video_dir):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule=scheduler_type,
        clip_sample=True,
        prediction_type='epsilon'
    )
    # init scheduler
    noise_scheduler.set_timesteps(num_diffusion_iters)

    # model and parameters
    path_to_checkpoint = model_path
    nets, _ = get_nets()
    accelerator = Accelerator()
    ema_nets = accelerator.prepare(nets)
    accelerator.load_state(path_to_checkpoint)
    device = accelerator.device

    # fixed parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    action_dim = 2

    # the stats for normalization and unnormalization.
    stats = get_stats()
    # the max step of inference.
    max_steps = max_inference_steps
    modelEvaluate(ema_nets, noise_scheduler,
                  pred_horizon, obs_horizon, action_horizon, action_dim, stats,
                  max_steps, output_video_dir, seed, device)


def modelEvaluate(ema_nets, noise_scheduler,
                  pred_horizon, obs_horizon, action_horizon, action_dim, stats,
                  max_steps, output_video_dir, seed, device):
    score_list = list()
    step_list = list()
    timecost_list = list()  # 新增时间成本列表
    for seed_i in range(seed, seed + 20):
        video_name = os.path.join(output_video_dir, f"video{seed_i}.avi")
        score, step, timecost = actionGenerating(ema_nets, noise_scheduler,
                                                 pred_horizon, obs_horizon, action_horizon, action_dim, stats,
                                                 max_steps, video_name, seed_i, device)
        score_list.append(score)
        step_list.append(step)
        timecost_list.append(timecost)

    avg_score = sum(score_list) / len(score_list)
    avg_step = sum(step_list) / len(step_list)
    avg_timecost = sum(timecost_list) / len(timecost_list)
    stats_file_path = os.path.join(output_video_dir, "stats.txt")

    with open(stats_file_path, 'w') as stats_file:
        stats_file.write("Individual Scores:\n")
        for idx, score in enumerate(score_list):
            stats_file.write(f"Run seed{idx + seed}: {score}\n")
        stats_file.write(f"\nAverage Score: {avg_score}\n")

        stats_file.write("\nIndividual Steps:\n")
        for idx, step in enumerate(step_list):
            stats_file.write(f"Run seed{idx + seed}: {step} steps\n")
        stats_file.write(f"\nAverage Steps: {avg_step}\n")

        stats_file.write("\nIndividual Time Costs:\n")
        for idx, timecost in enumerate(timecost_list):
            stats_file.write(f"Run seed{idx + seed}: {timecost:.6f} seconds\n")
        stats_file.write(f"\nAverage Time Cost: {avg_timecost:.6f} seconds\n")

    print(f"Stats have been written to {stats_file_path}")


def actionGenerating(ema_nets, noise_scheduler, pred_horizon, obs_horizon, action_horizon, action_dim, stats, max_steps,
                     output_video_path, seed, device):
    env = PushTImageEnv()
    env.seed(seed)
    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0
    start_time = time.perf_counter()

    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]
            nimages = images

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = ema_nets['vision_encoder'](nimages)
                # (2,512)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end, :]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    end_time = time.perf_counter()

    print('Score: ', max(rewards))
    generate_video(imgs, output_video_path)
    print("Video has been created at", output_video_path)
    return max(rewards), step_idx - 1, end_time - start_time


def generate_video(images, output_video_path, frame_rate=30):
    """
    Generates a video from a sequence of images.
    :param images: List or array of images to be included in the video.
    :param output_video_path: Path where the video will be saved.
    :param frame_rate: Frame rate of the video in frames per second.
    """
    # Convert list of images to a numpy array
    images = np.array(images)
    height, width = images.shape[1:3]

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for frame in images:
        out.write(frame)

    # Release VideoWriter
    out.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for PushTImageEnv.")
    parser.add_argument('--num_diffusion_iters', type=int, default=100,
                        help='Number of diffusion iterations.')
    parser.add_argument('--scheduler_type', type=str, default='squaredcos_cap_v2',
                        choices=['linear', 'cosine', 'cosine_with_restarts', 'squaredcos_cap_v2'],
                        help='Type of the noise scheduler.')
    parser.add_argument('--model_path', type=str, default="/mnt/ssd/fyz/pushT/20240718-234633-499/",
                        help='Path to the model checkpoint.')
    parser.add_argument('--max_inference_steps', type=int, default=500,
                        help='Maximum number of inference steps.')
    parser.add_argument('--seed', type=int, default=200000,
                        help='Random seed for environment initialization.')
    parser.add_argument('--output_video_dir', type=str, default="../output_video/base_model/",
                        help='Output path for generating video.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_video_dir, exist_ok=True)
    main(args.num_diffusion_iters, args.scheduler_type, args.model_path, args.max_inference_steps, args.seed,
         args.output_video_dir)
