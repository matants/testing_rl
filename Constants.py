import torch
import torch.optim as optim
from ExperienceReplay import *
import gym
from NeuralNet import *

EXP_REPLAY_SIZE = 50000
BATCH_SIZE = 16
REPTILE_BATCH_SIZE = 16
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY_TIME = 100
TARGET_UPDATE = 1
PRINT_PER = 1
LEARNING_RATE = 1e-3  # MER paper uses 1e-4, but this seems better?
NUM_EPISODES = 2000
STEPS_PER_TRAINS = BATCH_SIZE
TRAIN_ITERATIONS = 1
IS_PROCGEN = False
BETA_MER = 1
GAMMA_MER = 0.3
IS_MER = False


if IS_PROCGEN:
    env_name = 'procgen:procgen-coinrun-v0'
else:
    env_name = 'CartPole-v1'

env = gym.make(env_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    n_actions = env.action_space.n
except AttributeError:
    n_actions = env.action_space.shape[0]

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

# optimizer = optim.SGD(policy_net.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
if IS_MER:
    memory = ReservoirReplayMemory(EXP_REPLAY_SIZE)
else:
    memory = ReplayMemory(EXP_REPLAY_SIZE)
loss_func = nn.MSELoss()
