import argparse
import collections
import os
import time

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from diffusers import DPMSolverMultistepScheduler

from scheduler import get_scheduler
from tqdm.auto import tqdm

from pushTEnvOriginal import PushTImageEnv
from pushTImageDataset import get_stats, normalize_data, unnormalize_data
from train_ddp import get_nets


def main(num_diffusion_iters, num_inference_steps, scheduler_type, beta_type,
         model_path, max_inference_steps, seed, output_video_dir):
    # model and parameters
    path_to_checkpoint = model_path
    nets, _ = get_nets()
    accelerator = Accelerator()
    nets['vision_encoder'] = accelerator.prepare(nets['vision_encoder'])
    nets['noise_pred_net'] = accelerator.prepare(nets['noise_pred_net'])
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
    modelEvaluate(nets, num_diffusion_iters, num_inference_steps, scheduler_type, beta_type,
                  pred_horizon, obs_horizon, action_horizon, action_dim, stats,
                  max_steps, output_video_dir, seed, device)


def modelEvaluate(ema_nets, num_diffusion_iters, num_inference_steps, scheduler_type, beta_type,
                  pred_horizon, obs_horizon, action_horizon, action_dim, stats,
                  max_steps, output_video_dir, seed, device):
    score_list = list()
    step_list = list()
    timecost_list = list()
    for seed_i in range(seed, seed + 50):
        video_name = os.path.join(output_video_dir, f"video{seed_i}.avi")
        score, step, timecost = actionGenerating(ema_nets, num_diffusion_iters, num_inference_steps, scheduler_type,
                                                 beta_type,
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
        stats_file.write("Model Evaluation Statistics\n\n")

        stats_file.write("Individual Scores:\n")
        stats_file.write("Seed\tScore\n")
        for idx, score in enumerate(score_list):
            stats_file.write(f"{seed + idx}\t{score}\n")
        stats_file.write("\n")

        stats_file.write("Average Score:\t{avg_score}\n\n".format(avg_score=avg_score))

        stats_file.write("Individual Steps:\n")
        stats_file.write("Seed\tSteps\n")
        for idx, step in enumerate(step_list):
            stats_file.write(f"{seed + idx}\t{step}\n")
        stats_file.write("\n")

        stats_file.write("Average Steps:\t{avg_step}\n\n".format(avg_step=avg_step))

        stats_file.write("Individual Time Costs (in seconds):\n")
        stats_file.write("Seed\tTime Cost\n")
        for idx, timecost in enumerate(timecost_list):
            stats_file.write(f"{seed + idx}\t{timecost:.6f}\n")
        stats_file.write("\n")

        stats_file.write("Average Time Cost (in seconds):\t{avg_timecost:.6f}\n".format(avg_timecost=avg_timecost))

    print(f"Stats have been written to {stats_file_path}")


def actionGenerating(ema_nets, num_diffusion_iters, num_inference_steps, scheduler_type, beta_type,
                     pred_horizon, obs_horizon, action_horizon, action_dim, stats, max_steps,
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
            noise_scheduler = get_scheduler(num_diffusion_iters, num_inference_steps, scheduler_type, beta_type)
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

                if isinstance(noise_scheduler, DPMSolverMultistepScheduler):
                    timesteps = noise_scheduler.timesteps[1:]
                else:
                    timesteps = noise_scheduler.timesteps

                for k in timesteps:
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
    parser.add_argument('--num_training_steps', type=int, default=1000,
                        help='Number of diffusion iterations, same with training.')
    parser.add_argument('--num_inference_steps', type=int, default=25,
                        help='Number of steps of a diffusion iter.')
    parser.add_argument('--scheduler_type', type=str, default="ddpm",
                        help='Type of the noise scheduler.')
    parser.add_argument('--beta_type', type=str, default='squaredcos_cap_v2',
                        choices=['linear', 'scaled_linear', 'squaredcos_cap_v2'],
                        help='Type of the beta scheduler.')
    parser.add_argument('--model_path', type=str, default="/mnt/ssd/fyz/pushT/ecattention-epoch500/",
                        help='Path to the model checkpoint.')
    parser.add_argument('--max_inference_steps', type=int, default=500,
                        help='Maximum number of predicted action steps.')
    parser.add_argument('--seed', type=int, default=200000,
                        help='Random seed for environment initialization.')
    parser.add_argument('--output_video_dir', type=str, default="/mnt/ssd/fyz/pushT/output_video/ecattention_1000_ddpm25/",
                        help='Output path for generating video.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_video_dir, exist_ok=True)
    main(args.num_training_steps, args.num_inference_steps, args.scheduler_type, args.beta_type,
         args.model_path, args.max_inference_steps, args.seed, args.output_video_dir)
