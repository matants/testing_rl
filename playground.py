import gym
from itertools import count
# # print(gym.envs.registry.all())
#
env = gym.make('procgen:procgen-coinrun-v0')
# env = gym.wrappers.Monitor(env, directory="monitors", force=True)
for i_episode in range(20):
    observation = env.reset()
    env.render()
    for t in count():
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

