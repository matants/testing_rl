import matplotlib.pyplot as plt
import torch


def plot_rewards(episode_rewards, is_ipython=False):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        from IPython import display
        display.clear_output(wait=True)
        display.display(plt.gcf())


def print_average_over_episodes(episode_rewards, n=100):
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if len(rewards_t) < n:
        return
    means = rewards_t.unfold(0, n, 1).mean(1).view(-1)
    print("Episode {}: Mean reward over last {} episodes: {}".format(len(rewards_t), n, means[-1]))
