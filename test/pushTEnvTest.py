from pushTEnv import PushTEnv, PushTImageEnv
import time

# 创建PushTEnv实例
env = PushTEnv(
    legacy=False,
    block_cog=None,
    damping=None,
    render_action=True,
    render_size=96,
    reset_to_state=None
)

# env = PushTImageEnv()

# 重置环境
obs, info = env.reset()

# 打印观察空间和动作空间信息
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

max_steps = 100  # 设置最大步数
step = 0

# 进行循环直到环境终止或达到最大步数
while True:
    # 定义随机动作
    action = env.action_space.sample()
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    # 渲染环境
    env.render(mode="human")
    # 检查是否需要结束循环
    if terminated or truncated or step >= max_steps:
        print(f"Simulation ended at step {step}.")
        break
    step += 1
    time.sleep(0.1)


# 清理资源
env.close()
