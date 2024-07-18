import argparse
import os
import time

import numpy as np
import torch
from diffusers import EMAModel, get_scheduler, DDPMScheduler
from torch import nn
from tqdm.auto import tqdm

from pushTImageDataset import get_dataloader
from conditionalUnet import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from accelerate import Accelerator


def train_loop(nets, dataloader, optimizer, lr_scheduler, ema, noise_scheduler, num_epochs,
               save_directory):
    accelerator = Accelerator(gradient_accumulation_steps=2,
                              mixed_precision="fp16")
    device = accelerator.device

    nets, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        nets, optimizer, dataloader, lr_scheduler
    )
    vision_encoder = nets['vision_encoder']
    noise_pred_net = nets['noise_pred_net']
    ema.to(device)

    with tqdm(range(num_epochs), desc='Epoch', disable=not accelerator.is_local_main_process) as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(dataloader, desc='Batch', leave=False, disable=not accelerator.is_local_main_process) as tepoch:
                for nbatch in tepoch:
                    nimage = nbatch['image'][:, :].to(device)
                    nagent_pos = nbatch['agent_pos'][:, :].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    image_features = vision_encoder(nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage.shape[:2], -1)

                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)

                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                    noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # loss.backward()
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    ema.step(nets.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))
            if (epoch_idx + 1) % 20 == 0:
                exact_directory = os.path.join(save_directory, time.strftime("%Y%m%d-%H%M%S") + "-" + str(epoch_idx))
                ema.copy_to(nets.parameters())
                accelerator.wait_for_everyone()
                accelerator.save_state(exact_directory)
                print(f"Saved model to {exact_directory}, loss: {np.mean(epoch_loss)}")


def get_nets(obs_horizon=2):
    # Load your data and define your model here
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    vision_feature_dim = 512
    lowdim_obs_dim = 2
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    noise_pred_net = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon)
    nets = nn.ModuleDict({'vision_encoder': vision_encoder, 'noise_pred_net': noise_pred_net})
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    return nets, ema


def prepare_data(args):
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8

    nets, ema = get_nets(obs_horizon=obs_horizon)

    dataloader = get_dataloader(dataset_path=args.dataset_dir, pred_horizon=pred_horizon, obs_horizon=obs_horizon,
                                action_horizon=action_horizon,
                                batch_size=args.batch_size, small_dataset=True)

    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=args.lr, weight_decay=1e-6)

    lr_scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=500,
                                 num_training_steps=len(dataloader) * args.num_epochs)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_iters, beta_schedule='squaredcos_cap_v2',
                                    clip_sample=True,
                                    prediction_type='epsilon')
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return nets, dataloader, optimizer, lr_scheduler, ema, noise_scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the model.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train the model.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the dataloader.")
    parser.add_argument("--diffusion_iters", type=int, default=100, help="Iteration of one diffusion step.")
    parser.add_argument("--save_dir", type=str, default="/mnt/ssd/fyz/pushT/", help="Directory of saving model.")
    parser.add_argument("--dataset_dir", type=str, default="../pusht_cchi_v7_replay.zarr.zip", help="Path of dataset.")
    args = parser.parse_args()

    nets, dataloader, optimizer, lr_scheduler, ema, noise_scheduler = prepare_data(args)
    train_loop(nets, dataloader, optimizer, lr_scheduler, ema, noise_scheduler, args.num_epochs,
               args.save_dir)
