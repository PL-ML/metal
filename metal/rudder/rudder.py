import os
import sys
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from metal.common.cmd_args import cmd_args
from metal.common.pytorch_util import weights_init, to_num, glorot_uniform
from metal.generator.tree_encoder import LogicEncoder
from metal.parser.sygus_parser import SyExp
import metal.common.constants as constants


class RewardRedistributionLSTM(nn.Module):
    def __init__(self, latent_dim):
        super(RewardRedistributionLSTM, self).__init__()

        self.latent_dim = latent_dim

        # takes [state, act_embedding, t] as input
        self.state_gru = nn.GRUCell(latent_dim*2+1, latent_dim)

        # predicts [acc_reward_in_10steps, value, acc_reward]
        self.pred_lin = nn.Linear(self.latent_dim, 4)

        self.state = None
        self.pred_ls = []
        self.lstm_input_ls = []
        self.reset()

        weights_init(self)

        # used for auxillary loss
        self.filter = Variable(torch.ones(1, 1, cmd_args.future_steps), requires_grad=False)

        # used to generated interpolated input
        self.interpolate_w = torch.Tensor(np.linspace(0, 1, cmd_args.ig_step, dtype=np.float32))

    def reset(self):
        self.state = Variable(torch.Tensor(1, self.latent_dim), requires_grad=True)
        glorot_uniform(self.state.data)
        self.pred_ls = []
        self.lstm_input_ls = []

    def forward(self, lstm_input):

        self.state = self.state_gru(lstm_input, self.state)

        pred = self.pred_lin(self.state)

        self.pred_ls.append(pred)
        self.lstm_input_ls.append(lstm_input.data.clone())

    def get_loss(self, reward_ls):
        reward_vec = torch.Tensor(reward_ls) # (T)-dim vec
        pred_mat = torch.cat(self.pred_ls, dim=0) # (T,4)-dim mat

        # aux target 1
        tmp_vec = Variable(torch.cat([reward_vec, torch.zeros(cmd_args.future_steps-1)], dim=0))
        short_term_acc_reward = F.conv1d(tmp_vec.view(1, 1, -1), self.filter).view(-1, 1) # (T,1)-dim vec

        # aux target 2
        acc_reward_vec = torch.cumsum(reward_vec, dim=0)
        true_return = torch.sum(reward_vec)
        state_value_vec = Variable((true_return - acc_reward_vec).view(-1, 1)) # (T,1)-dim vec

        # aux target 3
        acc_reward_vec = Variable(torch.cumsum(reward_vec, dim=0).view(-1, 1)) # (T,1)-dim vec

        aux_target = torch.cat([short_term_acc_reward, state_value_vec, acc_reward_vec], dim=-1) # (T,3)-dim mat
        aux_loss = F.mse_loss(pred_mat[:, :-1], aux_target)

        return_pred_loss = F.mse_loss(pred_mat[-1, -1], Variable(torch.Tensor([true_return])))

        loss = return_pred_loss + aux_loss

        return loss

    def integrated_gradient(self, avg_return, true_return):

        lstm_input_mat = torch.cat(self.lstm_input_ls, dim=0).unsqueeze(1) # (T, 1, latent_dim*2+1)-dim mat

        interpolated_input = torch.cat([lstm_input_mat * w for w in self.interpolate_w], dim=1)
        interpolated_input = Variable(interpolated_input, requires_grad=True) # (T, ig_step, latent_dim*2+1)-dim mat

        self.reset()
        lstm_state = Variable(torch.Tensor(cmd_args.ig_step, self.latent_dim), requires_grad=True)
        glorot_uniform(lstm_state.data)

        pred_mat = None
        for t in range(interpolated_input.shape[0]):

            lstm_state = self.state_gru(interpolated_input[t], lstm_state)

            if t+1 == interpolated_input.shape[0]:
                pred_mat = self.pred_lin(lstm_state)

        ig_preds = torch.sum(pred_mat[:, 2:], dim=-1) # (ig_step)

        intgrd_grads = None

        for i in range(cmd_args.ig_step):
            single_ig = torch.autograd.grad(ig_preds[i], interpolated_input, retain_graph=True)[0]
            single_ig = single_ig[:, i, :] # (T, latent_dim*2+1)-dim mat
            intgrd_grads = single_ig if intgrd_grads is None else (intgrd_grads + single_ig)

        # scale
        intgrd_grads /= cmd_args.ig_step
        intgrd_grads *= interpolated_input[:, -1, :]

        # get per step ig
        intgrd_grads = torch.sum(intgrd_grads, dim=-1) # (T)-dim vec

        # sets end to 0 for stability
        intgrd_grads[-cmd_args.future_steps:] = 0

        # scale by return

        true_return = Variable(torch.Tensor([true_return]))
        avg_return = Variable(torch.Tensor([avg_return]))

        pred_return = pred_mat[-1, -1]
        error = pred_return - torch.sum(intgrd_grads)
        intgrd_grads += error / true_return

        epsilon_sqr = torch.sqrt(torch.abs(avg_return).clamp(1e-5))
        epsilon_sqr = torch.sqrt(epsilon_sqr) / 5
        x = torch.sign(pred_return) * epsilon_sqr
        x = true_return / (pred_return + x)
        intgrd_grads *= torch.clamp(x, 1e-5, 1.5)

        return intgrd_grads



