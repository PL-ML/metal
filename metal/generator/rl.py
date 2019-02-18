from __future__ import print_function

import os
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from metal.common.checker import eval_result
from metal.common.cmd_args import cmd_args
from metal.common.spec_tree import is_tree_complete
from metal.parser.sygus_parser import SyExp

class RLEnv(object):
    def __init__(self, specsample):
        """

        specsample:
            a SpecSample object containing the SpecTree of the original program

        """
        self.specsample = specsample
        self.pg = specsample.pg
        self.t = 0
        self.generated_tree = SyExp('Start', [])
        self.expand_ls = [self.generated_tree]

    def reset(self):
        self.t = 0
        self.generated_tree = SyExp('Start', [])

    def is_finished(self):
        return len(self.expand_ls) == 0

    def is_finished_generic(self):
        return is_tree_complete(self.specsample.spectree.grammar.nonTerminals, self.generated_tree)

    def get_cfg_mapping(self):
        return self.pg.cfg_mapping

    def get_spec_embedding(self):
        return self.pg.spec_embedding


def rollout(specsample, mem, decoder, rudder, avg_return, num_episode, use_random, eps):

    total_loss = 0.0
    rudder_loss = 0.0
    best_reward = -5.0
    best_tree = None
    acc_reward = 0.0

    # run num_episode times of episode and average out the loss for variance reduction
    for _ in range(num_episode):

        nll_list = []
        value_list = []
        reward_list = []

        env = RLEnv(specsample)
        decoder.reset()
        if cmd_args.use_rudder == 1:
            rudder.reset()

        while not env.is_finished():

            nll, vs = decoder(env, mem, use_random=use_random, eps=eps)
            reward = eval_result(env.specsample, env.generated_tree) if env.is_finished() else 0.0

            nll_list.append(nll)
            value_list.append(vs)
            reward_list.append(reward)

            env.t += 1

        true_return = np.sum(reward_list)
        if cmd_args.use_rudder == 1:
            rudder_loss += rudder.get_loss(reward_list)
            reward_list = rudder.integrated_gradient(avg_return, true_return)

        policy_loss, value_loss = actor_critic_loss(nll_list, value_list, reward_list)
        total_loss += policy_loss + value_loss

        if true_return > best_reward:
            best_reward = true_return
            best_tree = env.generated_tree
        acc_reward += true_return

    total_loss /= num_episode
    rudder_loss /= num_episode
    acc_reward /= num_episode

    return total_loss, rudder_loss, best_reward, best_tree, acc_reward


def actor_critic_loss(nll_list, value_list, reward_list):
    r = 0.0
    rewards = []
    for t in range(len(reward_list) - 1, -1, -1):
        r = r + reward_list[t]  # accumulated future reward
        rewards.insert(0, r / 10.0)

    policy_loss = 0.0
    targets = []
    for t in range(len(reward_list)):
        adv = rewards[t] - value_list[t].data[0, 0]
        policy_loss += nll_list[t] * adv
        targets.append(Variable(torch.Tensor([[rewards[t]]])))

    policy_loss /= len(reward_list)

    value_pred = torch.cat(value_list, dim=0)
    targets = torch.cat(targets, dim=0)
    value_loss = F.mse_loss(value_pred, targets)

    return policy_loss, value_loss
