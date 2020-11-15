import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ExperienceReplay import Transition, ReplayMemory
from NeuralNet import DQN

env = gym.make('CartPole-v0')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXP_REPLAY_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY_TIME = 50
TARGET_UPDATE = 1
PRINT_PER = 1
# LEARNING_RATE = 1e-3
NUM_EPISODES = 1001
STEPS_PER_TRAINS = 1
TRAIN_ITERATIONS = 1

# Get neural net input size from gym observation space
n_obs = env.observation_space.shape[0]
# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(n_obs, n_actions).to(device).float()
target_net = DQN(n_obs, n_actions).to(device).float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(EXP_REPLAY_SIZE)


def calc_eps_linear(episode, eps_start=EPS_START, eps_end=EPS_END, decay_time=EPS_DECAY_TIME):
    eps_ret = np.max([eps_end, eps_start * (1 - episode / decay_time)])
    # print("eps = {:5.2f}".format(eps_ret))
    return eps_ret


def select_action(state, is_training=True, eps_threshold=None):
    sample = random.random()
    if eps_threshold is None or sample > eps_threshold or not is_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy_net.train(mode=False)
            return policy_net(state.unsqueeze(0)).argmax().view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def print_average_over_episodes(n=100):
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if len(durations_t) < n:
        return
    means = durations_t.unfold(0, n, 1).mean(1).view(-1)
    print("Episode {}: Mean duration over last {} episodes: {}".format(len(durations_t), n, means[-1]))


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                         if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    policy_net.train(mode=True)
    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


steps_done = 0
for i_episode in range(NUM_EPISODES):
    # Initialize the environment and state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float)
    for t in count():
        # Select and perform an action
        action = select_action(state, eps_threshold=calc_eps_linear(i_episode))
        next_state, reward, done, _ = env.step(action.item())

        # Observe new state
        if done:
            next_state = None
            reward = -1
        else:
            next_state = torch.tensor(next_state, dtype=torch.float)

        # Store the transition in memory
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        steps_done += 1
        if steps_done % STEPS_PER_TRAINS == 0:
            for _ in range(TRAIN_ITERATIONS):
                optimize_model()
        if done:
            episode_durations.append(t + 1)
            if i_episode % PRINT_PER == 0:
                print_average_over_episodes(PRINT_PER)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
print("Training complete")
plot_durations()
plt.show()
# Showoff round
state = env.reset()
for t in count():
    env.render()
    state = torch.tensor(state, dtype=torch.float)
    action = select_action(state, is_training=False)
    state, reward, done, _ = env.step(action.item())
    sleep(0.01)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

print(steps_done)
env.close()
input('Run finished')
