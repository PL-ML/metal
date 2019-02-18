from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
import torch.optim as optim
from itertools import chain
from tqdm import tqdm

from metal.common.cmd_args import cmd_args, tic, toc
from metal.common.checker import eval_result
from metal.common.dataset import Dataset
from metal.common.utils import stat_counter
from metal.generator.rl import RLEnv, rollout, actor_critic_loss
from metal.spec_encoder.embedding import LSTMEmbed, EmbedMeanField
from metal.rudder.rudder import RewardRedistributionLSTM
import metal.generator.decoder as decoder
import metal.common.constants as constants

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    tic()

    # use batchsize=1 for simplicity
    assert cmd_args.rl_batchsize == 1

    # is training meta-learner?
    is_meta_learner = False
    if cmd_args.single_sample is None:
        assert cmd_args.exit_on_find == 0
        assert cmd_args.tune_test == 0
        is_meta_learner = True

        tqdm.write('learning meta learner')
    else:
        # if using pre-train for single_sample
        if cmd_args.tune_test == 1:
            assert os.path.isfile(cmd_args.data_root + '/mem_encoder')
            assert os.path.isfile(cmd_args.data_root + '/decoder')

            tqdm.write('solve single sample with pre-trained model')

        else:
            tqdm.write('solve single sample from scratch')

    dataset = Dataset()
    numOf_node_type = constants.NUM_OF_TYPE

    # define rudder
    rudder = None
    if cmd_args.use_rudder == 1:
        rudder = RewardRedistributionLSTM(cmd_args.embedding_size)

    # define mem_encoder, decoder and opt
    # mem_encoder = LSTMEmbed(cmd_args.embedding_size, numOf_node_type)
    mem_encoder = EmbedMeanField(cmd_args.embedding_size, len(dataset.node_type_dict), max_lv=cmd_args.s2v_level)

    decoder_class = getattr(decoder, cmd_args.decoder_model)
    decoder = decoder_class(cmd_args.embedding_size, rudder)

    if cmd_args.tune_test == 1:
        mem_encoder.load_state_dict(torch.load(cmd_args.data_root + '/mem_encoder'))
        decoder.load_state_dict(torch.load(cmd_args.data_root + '/decoder'))
        if cmd_args.use_rudder == 1:
            rudder.load_state_dict(torch.load(cmd_args.data_root + '/rudder'))

    params = [mem_encoder.parameters(), decoder.parameters()]
    if cmd_args.use_rudder == 1:
        params.append(rudder.parameters())

    optimizer = optim.Adam(chain.from_iterable(params), lr=cmd_args.learning_rate)

    eps = cmd_args.eps
    for epoch in range(cmd_args.num_epochs):

        epoch_best_reward = -5.0
        epoch_best_root = None
        epoch_acc_reward = 0.0

        pbar = tqdm(range(100))
        for k in pbar:

            specsample_ls = dataset.sample_minibatch(cmd_args.rl_batchsize, replacement=True)
            mem_batch = mem_encoder(specsample_ls)

            batch_total_loss = 0.0
            batch_rudder_loss = 0.0
            embedding_offset = 0
            for b in range(cmd_args.rl_batchsize):
                specsample = specsample_ls[b]
                # mem = mem_batch[embedding_offset: embedding_offset + specsample.spectree.numOf_nodes, :]
                embedding_offset += specsample.spectree.numOf_nodes

                total_loss, rudder_loss, best_reward, best_tree, acc_reward = rollout(specsample, mem_batch, decoder,
                                                                                      rudder,
                                                                                      (epoch_acc_reward / (k + 1)),
                                                                                      num_episode=cmd_args.num_episode,
                                                                                      use_random=True, eps=eps)
                eps *= cmd_args.eps_decay

                epoch_acc_reward += acc_reward / cmd_args.rl_batchsize
                batch_total_loss += total_loss
                batch_rudder_loss += rudder_loss
                if best_reward > epoch_best_reward:
                    epoch_best_reward = best_reward
                    epoch_best_root = best_tree

            optimizer.zero_grad()
            loss = batch_total_loss / cmd_args.rl_batchsize
            if cmd_args.use_rudder == 1:
                batch_rudder_loss /= cmd_args.rl_batchsize
                loss += batch_rudder_loss
            loss.backward()
            optimizer.step()
            pbar.set_description('avg reward: %.4f' % (epoch_acc_reward / (k + 1)))


        if is_meta_learner:
            if epoch % 50 == 0:
                torch.save(mem_encoder.state_dict(), cmd_args.data_root + '/mem_encoder_' + str(epoch))
                torch.save(decoder.state_dict(), cmd_args.data_root + '/decoder_' + str(epoch))

            torch.save(mem_encoder.state_dict(), cmd_args.data_root + '/mem_encoder')
            torch.save(decoder.state_dict(), cmd_args.data_root + '/decoder')
            if cmd_args.use_rudder == 1:
                torch.save(rudder.state_dict(), cmd_args.data_root + '/rudder')

        # tinytest for every 100 epochs
        specsample_ls = dataset.sample_minibatch(1, replacement=True)
        mem = mem_encoder(specsample_ls)
        _, _, _, generated_tree, _ = rollout(specsample_ls[0], mem, decoder, rudder,
                                          epoch_acc_reward / 100.0, num_episode=1, use_random=True, eps=0.0)

        print('epoch: %d, average reward: %.4f, Random sample result_r: %.4f' % (
        epoch, epoch_acc_reward / 100.0, eval_result(specsample_ls[0], generated_tree)))
        # print('epoch: %d, average reward: %.4f, Random: %s, result_r: %.4f' % (
        # epoch, acc_reward / 100.0, generated_tree, eval_result(specsample_ls[0], generated_tree)))
        # print("best_reward:", best_reward, ", best_root:", best_root)
        # print("best_reward:", best_reward)
        print('specs solved:', len(stat_counter.reported))
