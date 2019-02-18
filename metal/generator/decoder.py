from __future__ import print_function

import os
import sys
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

from metal.common.constants import TYPES_IN_SPEC
from metal.common.cmd_args import cmd_args
from metal.common.pytorch_util import weights_init, to_num
from metal.generator.tree_encoder import LogicEncoder
from metal.parser.sygus_parser import SyExp

import metal.common.constants as constants


class RecursiveDecoder(nn.Module):
    def __init__(self, latent_dim, rr_lstm):
        super(RecursiveDecoder, self).__init__()
        assert cmd_args.attention  # always use attention

        self.latent_dim = latent_dim
        self.rr_lstm = rr_lstm

        # rnn state tracker
        self.state_gru = nn.GRUCell(latent_dim, latent_dim)

        # TODO global embedding of operators
        self.op_embedding = nn.Embedding(len(constants.OP_NAME2IND), latent_dim)

        # define state value predictor
        self.value_pred_w1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.value_pred_w2 = nn.Linear(self.latent_dim, 1)
        self.value_net = lambda s: self.value_pred_w2(F.relu(self.value_pred_w1(s)))

        # params used to get the first attention
        self.first_att = nn.Linear(self.latent_dim, 1)
        # encoder (TreeLSTM) that generates a embedding for a entire tree 
        self.tree_encoder = LogicEncoder(self.latent_dim)
        # embedding that represents the current state of the generated program tree
        self.state = None

        weights_init(self)

    def gen_attention(self, mem, env):
        if env.t == 0:
            logits = self.first_att(mem)
        else:
            logits = torch.sum(mem * self.state, dim=1, keepdim=True)

        weights = F.softmax(logits, dim=0)
        attention = torch.sum(weights * mem, dim=0, keepdim=True)
        
        return attention

    def reset(self):
        self.state = None

    def forward(self, env, mem, use_random, eps=0.05):

        spectree = env.specsample.spectree

        if env.t == 0:
            attention = self.gen_attention(mem, env) # get attention
            self.state = self.tree_encoder(mem, attention, env) # generate overall state

        value = self.value_net(self.state) # get state value

        self.nll = 0.0

        syexp = env.expand_ls.pop(0)
        cfg_mapping = env.get_cfg_mapping()
        act_space = cfg_mapping[syexp.app][1:] # get all index except for the node itself
        avail_act_embedding = mem[act_space]
        if len(avail_act_embedding.shape) == 1:
            avail_act_embedding = avail_act_embedding.unsqueeze(0)

        # # stack the embeddings of currently available nodes
        # act_space = spectree.grammar.productions[syexp.app]  # e.g. [['and', 'depth1', 'depth1'], ['LN29']]
        # avail_act_embedding = []
        # for avail_act in act_space:
        #     nodename = avail_act[0]
        #     if nodename in constants.OP_NAME2IND:
        #         ind = Variable(torch.LongTensor([constants.OP_NAME2IND[nodename]]), requires_grad=False)
        #         avail_act_embedding.append(self.op_embedding(ind))
        #     else:
        #         avail_act_embedding.append(mem[spectree.nodename2ind[nodename], :].unsqueeze(0))
        # avail_act_embedding = torch.cat(avail_act_embedding, dim=0)

        # choose node to add
        act = self.choose_action(self.state, avail_act_embedding, use_random, eps)
        act_embedding = torch.index_select(avail_act_embedding, 0, act)

        # call RewardRedistributionLSTM for prediction
        if cmd_args.use_rudder:
            t = Variable(torch.Tensor([env.t]), requires_grad=False)
            rr_lstm_input = torch.cat([self.state.detach(), act_embedding, t], dim=-1)
            self.rr_lstm(rr_lstm_input)

        self.update_state(act_embedding)

        # expand the syexp
        prod_rule = spectree.grammar.productions[syexp.app]  # e.g. [['and', 'depth1', 'depth1'], ['LN29']]
        act_ind = act.data.cpu()[0]
        syexp.app = prod_rule[act_ind][0]  # e.g syexp.app='and'
        arg_name_ls = prod_rule[act_ind][1:]
        syexp.args = [SyExp(arg_name, []) for arg_name in arg_name_ls]  # e.g. syexp.args=[SyExp('depth1', []), ...]

        # push to stack for pre-order visit
        env.expand_ls = syexp.args[::-1] + env.expand_ls

        return self.nll, value

        # self.recursive_decode(env.generated_tree, env.specsample.spectree, mem, use_random, eps)

    def choose_action(self, state, cls_w, use_random, eps):
        """
        given current state and stack embeddings of possible nodes to add, perform softmax(dot(state, cls_w)), thus
        choosing the node to add to the tree

        state:
        cls_w:
            stack of node embeddings
        use_random:
        eps:

        return:
            index of the chosen node

        """
        if type(cls_w) is Variable or type(cls_w) is Parameter or type(cls_w) is torch.Tensor:
            logits = F.linear(state, cls_w, None)
        elif type(cls_w) is torch.nn.modules.linear.Linear:
            logits = cls_w(state)
        else:
            raise NotImplementedError()

        ll = F.log_softmax(logits, dim=1)
        if use_random:
            scores = torch.exp(ll) * (1 - eps) + eps / ll.shape[1]
            picked = torch.multinomial(scores, 1)
        else:
            _, picked = torch.max(ll, 1)

        picked = picked.view(-1)

        self.nll += F.nll_loss(ll, picked)
        return picked

    def update_state(self, input_embedding):
        self.state = self.state_gru(input_embedding, self.state)

    def recursive_decode(self, syexp, spectree, mem, use_random, eps):
        """
        Completely expand a SyExp object according to the production rule provided by the grammar of spectree

        syexp:
            a SyExp object to be expanded
        spectree:
            SpecTree object containing CFG object (grammar) and other info about the original program
        mem:
            external memory, i.e. embedding table of vars and ops
        use_random:
        eps:

        """

        # if SyExp is terminal node then return
        if syexp.app in spectree.grammar.terminals:
            return

        # stack the embeddings of currently available nodes
        act_space = spectree.grammar.productions[syexp.app] # e.g. [['and', 'depth1', 'depth1'], ['LN29']]
        avail_act_embedding = []
        for avail_act in act_space:
            nodename = avail_act[0]
            if nodename in constants.OP_NAME2IND:
                ind = Variable(torch.LongTensor([constants.OP_NAME2IND[nodename]]), requires_grad=False)
                avail_act_embedding.append(self.op_embedding(ind))
            else:
                avail_act_embedding.append(mem[spectree.nodename2ind[nodename], :].unsqueeze(0))
        avail_act_embedding = torch.cat(avail_act_embedding, dim=0)

        # choose node to add
        act = self.choose_action(self.state, avail_act_embedding, use_random, eps)
        self.update_state(torch.index_select(avail_act_embedding, 0, act))

        # expand the syexp
        act_ind = act.data.cpu()[0]
        syexp.app = act_space[act_ind][0] # e.g syexp.app='and'
        arg_name_ls = act_space[act_ind][1:]
        syexp.args = [SyExp(arg_name, []) for arg_name in arg_name_ls] # e.g. syexp.args=[SyExp('depth1', []), ...]

        # recursively expand the child nodes
        for child_syexp in syexp.args:
            self.recursive_decode(child_syexp, spectree, mem, use_random, eps)
