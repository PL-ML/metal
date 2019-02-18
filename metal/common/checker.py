from __future__ import print_function

import os
import sys
import numpy as np
from tqdm import tqdm

from metal.common.cmd_args import cmd_args
from metal.common.utils import CEHolder, CounterExample, stat_counter, py_eval_helper
from metal.common.constants import CE_KEYS

from metal.solver.sat import SatProxy
from metal.solver.cegar2qbf import CegarQBF


code_ce_dict = {}


def get_ce(g, syn_exp):
    spec = g.spectree.spec
    proxy = SatProxy(spec, syn_exp)
    status, ce_model = proxy.find_counter_example()
    if status:
        # found a counter example
        # decide the key, T or F
        kind = 'T' if py_eval_helper(ce_model, spec) else 'F'

        # print("spec:", spec)
        # print("syn_exp:", syn_exp)

        # a = py_eval_helper(ce_model, spec)
        # b = py_eval_helper(ce_model, syn_exp)
        # print("ce_model:", ce_model)
        # print("ce_model on spec:", a)
        # print("ce_model on syn:", b)

        # assert a != b

        ce = CounterExample(syn_exp, kind, ce_model) 
        return (-1, kind, ce)
    else:
        # no more counter examples, done!
        return (1, None, None)


def reward_w_interpolation(sample_index, holder, lambda_holder_eval, lambda_new_ce):
    # check if it passes
    status, key, ce = lambda_new_ce()
    if status > 0:
        return 1.0

    # interpolate ce and add neary ones into the buffer
    holder.interpolate_ce(ce) 
    
    #harmonic mean
    scores = []
    for key in CE_KEYS:                
        score = lambda_holder_eval(key)

    t = sum(scores) # t \in [0, 2.0]
    if t > 0:
        hm_t = 4.0 * scores[0] * scores[1] / t
    else:
        hm_t = 0.0

    return -2.0 + hm_t


def reward_1(sample_index, holder, lambda_holder_eval, lambda_new_ce):
    # print("\n\nsample_index:", sample_index)
    # holder.show_stats()
    # ct = 0
    # s = 0
    scores = []
    for key in CE_KEYS:                
        score = lambda_holder_eval(key)
        # print("key:", key,  "score: ", score, "ce_per_key:", holder.ce_per_key)
        # if key in holder.ce_per_key:
        #     ct += len(holder.ce_per_key[key].ce_list)
        #     s += 0.99
        scores.append(score)      
    t = sum(scores) # t \in [0, 2.0]
    if t > 0:
        hm_t = 4.0 * scores[0] * scores[1] / t
    else:
        hm_t = 0.0
    # print("ct=",ct, "t=", t, "s=",s)

    return -2.0 + hm_t


def eval_result_simple(g, generated_tree):
    stat_counter.add(g.sample_index, 'eval_result')
    if not g.sample_index in code_ce_dict:
        code_ce_dict[g.sample_index] = CEHolder(g)     
    holder = code_ce_dict[g.sample_index]

    passed_ct, all_ct = holder.eval_both(generated_tree)

    return 1.0 * passed_ct / all_ct


def eval_result(g, generated_tree):
    """

    g:
        SpecSample object
    generated_tree:
        SyExp object representing the tree we generate so far

    """
    # print("\n\neval_result:")
    # print("generated tree:", generated_tree)

    stat_counter.add(g.sample_index, 'eval_result')
    if not g.sample_index in code_ce_dict:
        code_ce_dict[g.sample_index] = CEHolder(g)     
    holder = code_ce_dict[g.sample_index]

    #  C(x,y,z) <==> F(x,y,z)
    # forall x,y,z:  C => F && F => C

    # expr-1:(p && q)   =>  expr-2: p 
    # expr-2  !=> expr-1

    #  x=T,y=F,z=T 
    #   a AND (not a)

    res = None
    lambda_holder_eval = lambda key: holder.eval(key, generated_tree)
    #lambda_new_ce = lambda: get_ce( g, generated_tree)
    lambda_new_ce = lambda: holder.get_failed_ce(generated_tree)

    if cmd_args.use_interpolation:
        res = reward_w_interpolation(g.sample_index, holder, lambda_holder_eval, lambda_new_ce)
    else:
        res = reward_1(g.sample_index, holder, lambda_holder_eval, lambda_new_ce)

    if res > -0.000001:
        tqdm.write("Found a solution: " + generated_tree.to_py())
        # stat_counter.report_once(g.sample_index)
        if cmd_args.exit_on_find:
            sys.exit()

    return res


def report_tested_stats(g, roots):
    if not g.sample_index in code_ce_dict:
        code_ce_dict[g.sample_index] = CEHolder(g)        
    holder = code_ce_dict[g.sample_index]

    stats = [ holder.eval_count(rt) for rt in roots  ]
    arr = np.array( stats )
    print("mean: ", np.mean(arr), " std: ", np.std(arr), "min: ", np.min(arr), "max: ", np.max(arr), "median: ", np.median(arr))

def show_ce_stats():
    for i in code_ce_dict:
        print("ce stats for prog index:", i)
        holder = code_ce_dict[i]    
        ct = {}
        for key in CE_KEYS:                
            if key in holder.ce_per_key:
                ct[key] = len(holder.ce_per_key[key].ce_list)

        print("counter examples info: ", ct)


def report_ce_stats(g, best_expr = None):
    if not g.sample_index in code_ce_dict:
        code_ce_dict[g.sample_index] = CEHolder(g)        
    holder = code_ce_dict[g.sample_index]

    ct = {}
    for key in CE_KEYS:                
        if key in holder.ce_per_key:
            if best_expr is not None:
                holder.eval_detail(key, best_expr)
            ct[key] = len(holder.ce_per_key[key].ce_list)

    print("counter examples info: ", ct)
