import torch
import gym

EXP_REPLAY_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY_TIME = 50
TARGET_UPDATE = 12
PRINT_PER = 1
# LEARNING_RATE = 1e-3
NUM_EPISODES = 21
STEPS_PER_TRAINS = 16
TRAIN_ITERATIONS = 1

env = gym.make('procgen:procgen-coinrun-v0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Get neural net input size from gym observation space
n_channels = env.observation_space.shape[2]
# Get number of actions from gym action space
n_actions = env.action_space.n
height = env.observation_space.shape[0]
width = env.observation_space.shape[1]
