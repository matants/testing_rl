import matplotlib.pyplot as plt
import torch
from time import time


def plot_rewards(episode_rewards, is_ipython=False):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(range(len(episode_rewards)), rewards_t.numpy())
    # Take 100 episode averages and plot them too
    mean_over = 100
    if len(rewards_t) >= mean_over:
        means = rewards_t.unfold(0, mean_over, 1).mean(1).view(-1)
        # means = torch.cat((torch.zeros(mean_over - 1), means))
        plt.plot(range(mean_over - 1, len(episode_rewards)), means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        from IPython import display
        display.clear_output(wait=True)
        display.display(plt.gcf())


def print_average_over_episodes(episode_rewards, start_time, n=100, ):
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if len(rewards_t) < n:
        return
    means = rewards_t.unfold(0, n, 1).mean(1).view(-1)
    print(
        "Episode {}: Mean reward over last {} episodes: {}. Elapsed time: {:.2f} seconds.".format(len(rewards_t), n,
                                                                                                  means[-1],
                                                                                                  time() - start_time))
