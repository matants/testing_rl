import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from time import sleep, time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Constants import *
from ExperienceReplay import Transition, ReplayMemory
from select_action import calc_eps_linear, select_action
from display_results import plot_rewards, print_average_over_episodes

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ioff()



def optimize_model(recent_transitions):
    if IS_MER:
        optimize_model_mer_batches(recent_transitions)
    else:
        optimize_model_standard()


def optimize_model_standard():
    if len(memory) < BATCH_SIZE * REPTILE_BATCH_SIZE:
        return
    for k in range(REPTILE_BATCH_SIZE):
        transitions = memory.sample(BATCH_SIZE)
        train_batch(transitions)


def train_batch(transitions):
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

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(len(transitions), device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    policy_net.train(mode=True)
    loss = loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def optimize_model_mer_batches(recent_transitions):
    if len(memory) < BATCH_SIZE * REPTILE_BATCH_SIZE:
        return
    use_recent_transitions_index = random.randint(0, REPTILE_BATCH_SIZE - 1)
    for k in range(REPTILE_BATCH_SIZE):
        if k == use_recent_transitions_index:
            transitions = recent_transitions
        else:
            transitions = memory.sample(BATCH_SIZE)
        state_dict_before = policy_net.state_dict()
        train_batch(transitions)
    reptile_step(BETA_MER, state_dict_before)


def reptile_step(step_size, state_dict_before):
    current_state_dict = policy_net.state_dict()
    for key in current_state_dict:
        current_state_dict[key] = state_dict_before[key] + step_size * (
                current_state_dict[key] - state_dict_before[key])
    policy_net.load_state_dict(current_state_dict)


steps_done = 0
episode_rewards = []
episode_durations = []
recent_transitions = []
start_time = time()
for i_episode in range(NUM_EPISODES):
    # Initialize the environment and state
    episode_start_time = time()
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)
    if IS_PROCGEN:
        state = state.permute(2, 0, 1)
    tot_reward = 0
    for t in count():
        # Select and perform an action
        action = select_action(policy_net, state, eps_threshold=calc_eps_linear(i_episode))
        next_state, reward, done, _ = env.step(action.item())

        # Observe new state
        if done:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
            if IS_PROCGEN:
                next_state = next_state.permute(2, 0, 1)

        tot_reward += reward
        reward = torch.tensor([reward], device=device)
        # store transition in replay memory
        if IS_MER:
            recent_transitions.append(Transition(state, action, next_state, reward))
        else:
            memory.push(state, action, next_state, reward)

        # Perform one step of the optimization (on the target network)
        if steps_done % STEPS_PER_TRAINS == 0:
            state_dict_before = policy_net.state_dict()
            for _ in range(TRAIN_ITERATIONS):
                optimize_model(recent_transitions)
            if IS_MER:
                # reptile step
                reptile_step(GAMMA_MER, state_dict_before)
                for transition in recent_transitions:
                    memory.push(*transition)
                recent_transitions = []

        # Move to the next state
        steps_done += 1
        state = next_state
        if done:
            episode_rewards.append(tot_reward)
            episode_durations.append(time() - episode_start_time)
            if i_episode % PRINT_PER == 0:
                print_average_over_episodes(episode_rewards, start_time, PRINT_PER)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Training complete. Total time: {:.2f} seconds.".format(time() - start_time))
plot_rewards(episode_rewards)
plt.show()
# Showoff round
if IS_PROCGEN:
    env = gym.make(env_name, render_mode="human")
state = env.reset()
for t in count():
    env.render()
    state = torch.tensor(state, dtype=torch.float, device=device)
    if IS_PROCGEN:
        state = state.permute(2, 0, 1)
    action = select_action(policy_net, state, is_training=False)
    state, reward, done, _ = env.step(action.item())
    sleep(0.01)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

print(steps_done)
env.close()
print('Run finished')
