from Constants import *
import numpy as np
import random
import torch


def calc_eps_linear(episode, eps_start=EPS_START, eps_end=EPS_END, decay_time=EPS_DECAY_TIME):
    eps_ret = np.max([eps_end, eps_start * (1 - episode / decay_time)])
    # print("eps = {:5.2f}".format(eps_ret))
    return eps_ret


def select_action(policy_net, state, is_training=True, eps_threshold=None):
    policy_net.eval()
    sample = random.random()
    if eps_threshold is None or sample > eps_threshold or not is_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.unsqueeze(0)).argmax().view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
