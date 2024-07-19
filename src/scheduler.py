from diffusers import DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, DDPMScheduler


def get_scheduler(num_diffusion_iters, num_inference_steps, scheduler_type, beta_type):
    if scheduler_type == "ddim":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_type,
            clip_sample=True,
            prediction_type='epsilon'
        )
    elif scheduler_type == "pndm":
        noise_scheduler = PNDMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_type,
            skip_prk_steps=True
        )
    elif scheduler_type == "euler":
        noise_scheduler = EulerDiscreteScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_type
        )
    elif scheduler_type == "euler_ancestral":
        noise_scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_type
        )
    elif scheduler_type == "dpm_solver":
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_type,
            use_karras_sigmas=True
        )
    elif scheduler_type == "heun":
        noise_scheduler = HeunDiscreteScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_type
        )
    elif scheduler_type == "lms":
        noise_scheduler = LMSDiscreteScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_type
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_type,
            clip_sample=True,
            prediction_type='epsilon'
        )
    noise_scheduler.set_timesteps(num_inference_steps)
    print(f"Scheduler type: {scheduler_type}")
    return noise_scheduler