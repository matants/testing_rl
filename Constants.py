import torch
import torch.optim as optim
from ExperienceReplay import *
import gym
from NeuralNet import *

EXP_REPLAY_SIZE = 50000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY_TIME = 50
TARGET_UPDATE = 8
PRINT_PER = 10
# LEARNING_RATE = 1e-3
NUM_EPISODES = 301
STEPS_PER_TRAINS = 4
TRAIN_ITERATIONS = 1
IS_PROCGEN = False


if IS_PROCGEN:
    env = gym.make('procgen:procgen-coinrun-v0')
else:
    env = gym.make('CartPole-v1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_actions = env.action_space.n

if IS_PROCGEN:
    # Get neural net input size from gym observation space
    n_channels = env.observation_space.shape[2]
    # Get number of actions from gym action space
    height = env.observation_space.shape[0]
    width = env.observation_space.shape[1]

    policy_net = DQN_conv(n_channels, n_actions, height, width).to(device).float()
    target_net = DQN_conv(n_channels, n_actions, height, width).to(device).float()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
else:
    n_obs = env.observation_space.shape[0]
    policy_net = DQN(n_obs, n_actions).to(device).float()
    target_net = DQN(n_obs, n_actions).to(device).float()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(EXP_REPLAY_SIZE)
