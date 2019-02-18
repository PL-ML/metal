import argparse
import os
from tqdm import tqdm

cmd_opt = argparse.ArgumentParser(description='Argparser')
cmd_opt.add_argument('-data_root', default=None, help='root of dataset')
cmd_opt.add_argument('-file_list', default=None, help='list of programs')
cmd_opt.add_argument('-init_model_dump', default=None, help='init model dump')
cmd_opt.add_argument('-save_dir', default=None, help='root for output')
cmd_opt.add_argument('-att_dir', default=None, help='root for att output')
cmd_opt.add_argument('-log_file', default=None, help='log file')
cmd_opt.add_argument('-aggressive_check', default=0, type=int, help='penalize verbose/unnecessary sub expression')
cmd_opt.add_argument('-ctx', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-inv_reward_type', default='any', help='any/ordered')
cmd_opt.add_argument('-phase', default='test', help='train/test')
cmd_opt.add_argument('-train_frac', default=0.9, type=float, help='fraction for training')
cmd_opt.add_argument('-tune_test', default=0, type=int, help='active search or not')
cmd_opt.add_argument('-init_samples', default=10000, type=int, help='initial number of samples')
cmd_opt.add_argument('-interpolate_samples', default=-1, type=int, help='interpolation samples')
cmd_opt.add_argument('-use_interpolation', default=0, type=int, help='whether use interpolation')

cmd_opt.add_argument('-seed', default=1, type=int, help='random seed')
cmd_opt.add_argument('-use_ce', default=1, type=int, help='whether use counter examples')
cmd_opt.add_argument('-rl_batchsize', default=1, type=int, help='batch size for rl training')
cmd_opt.add_argument('-single_sample', default=None, type=str, help='tune single program')
cmd_opt.add_argument('-replay_memsize', default=100, type=int, help='replay memsize')
cmd_opt.add_argument('-num_epochs', default=10000, type=int, help='num epochs')
cmd_opt.add_argument('-embedding_size', default=128, type=int, help='embedding size')
cmd_opt.add_argument('-s2v_level', default=20, type=int, help='# propagations of s2v')
cmd_opt.add_argument('-ce_batchsize', default=10000, type=int, help='batchsize for counter example check')
cmd_opt.add_argument('-eps', default=0.85, type=float, help='exploration constant')
cmd_opt.add_argument('-eps_decay', default=0.9999, type=float, help='exp decay of the exploration constant')
cmd_opt.add_argument('-num_episode', default=10, type=int, help='how many episode to accumulate before training')

cmd_opt.add_argument('-use_rudder', default=0, type=int, help='whether use rudder')
cmd_opt.add_argument('-ig_step', default=100, type=int, help='num of integrated gradient steps')
cmd_opt.add_argument('-future_steps', default=5, type=int, help='num to look ahead in rudder aux/to clip IG to zero')


cmd_opt.add_argument('-attention', default=1, type=int, help='attention for embedding')
cmd_opt.add_argument('-exit_on_find', default=0, type=int, help='exit when found')

cmd_opt.add_argument('-decoder_model', default='RecursiveDecoder', help='decoder model')
cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='random seed')

cmd_args, _ = cmd_opt.parse_known_args()

start_time = None
import time
def tic():
    global start_time
    start_time = time.time()

def toc():
    global start_time
    cur_time = time.time()
    return cur_time - start_time 
if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

tqdm.write(str(cmd_args))
