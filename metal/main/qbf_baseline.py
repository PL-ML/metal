#!/usr/bin/env python3

import os
import sys
from tqdm import tqdm

from metal.common.cmd_args import cmd_args, tic, toc
from metal.common.dataset import Dataset
from metal.common.spec_tree import SygusInstance
from metal.parser.sygus_parser import SyExp
from metal.solver.cegar2qbf import CegarQBF


class QBFSearch(object):
    def __init__(self, spec, grammar):
        self.spec = spec
        self.G = grammar
        self.partial_tree = SyExp(grammar.start, [])
        self.done = False
    
    def any_hope(self):
        checker = CegarQBF(self.spec, self.partial_tree)
        status = checker.any_hope()
        return status

    def countNT(self, nonT, sexp):
        # just count the number of cominbations of nonT

        if nonT == sexp.app:
            return self.G.count_dict[nonT]

        rules = self.G.productions[nonT]
        pick_rule = None            
        for r  in rules:
            if sexp.app == r[0]:
                pick_rule = r
                break
        if pick_rule is None:
            print("countNT cannot find picked rule!!  nonT:", nonT, " sexp:", sexp)
 
        res = 1
        for i in range(1, len(pick_rule)):
            res *= self.countNT( pick_rule[i], sexp.args[i-1] )
        return  res


    def search_in_rule(self, pick_rule, index):
        # decide the part of the body to fill
        if len(pick_rule) == 1:
            return self.search(pick_rule[0], 0)

        bases = [1]
        cts = []
        n = len(pick_rule)
        for i in range(1, n):
            bases.append( self.G.count_dict[ pick_rule[i] ])
        
        for i in range(n):
            tmp = 1
            for j in range(i+1, n):
                tmp *= bases[j]
            cts.append(tmp)

        indices = [] # the index for each part of the body
        c = index
        for i in range(n):
            indices.append( c // cts[i] )
            c = c % cts[i]

        assert indices[0] == 0


        for i in range(1, n):
            nonT = pick_rule[i]
            k = indices[i]
            status = self.search(nonT, k)
            if not status:
                return False
        return True


    def search(self, nonT, from_k):
        # evaluate the K-th program of (non-)terminal nonT
        if nonT in self.G.terminals:
            return self.any_hope()

        status, node = self.partial_tree.dfs_find( nonT )
        if not status:
            print("nonT:", nonT)
            print("pt:", self.partial_tree)
        assert status 

        k = from_k
        all_ct = self.G.count_dict[nonT]
        assert k < all_ct
        if not self.any_hope():
            # no need to expand
            return False

        # decide which rule to pick
        res = 0
        rules = self.G.productions[nonT]
        pick_rule = None            
        index = k
        for r  in rules:
            tmp = 1
            for i in range(1, len(r)):
                tmp *= self.G.count_dict[r[i]]
            if index >= tmp:
                index -= tmp
            else:
                pick_rule = r
                break

        # print("search: nonT=", nonT, "picke_rule:",  pick_rule)
        assert pick_rule


        # print("\n\nbefore replacing: ", self.partial_tree)
        node.app = pick_rule[0]
        node.args = []
        if len(pick_rule) > 1:
            for i in range(1,len(pick_rule)):
                node.args.append( SyExp(pick_rule[i], []) )
        # print("replaced ", nonT, " with ", pick_rule)
        # print("partial tree:", self.partial_tree)
        # print("\n\n")

        return self.search_in_rule(pick_rule, index)

    def synthesize(self):
        # print("count_dict: ", self.G.count_dict)
        k = 0
        suc = False
        all_ct = self.G.count_dict[ self.G.start ]
        # print("all_ct:", all_ct)
        delta_stats = {}
        while k < all_ct:
            self.partial_tree = SyExp(self.G.start, [])
            status = self.search(self.G.start, k)
            if status:
                suc = True
                break
            delta = self.countNT(self.G.start, self.partial_tree)

            if delta not in delta_stats:
                delta_stats[delta] = 0

            delta_stats[delta] += 1
            
            # print("k:", k)
            # print("partial_tree: ", self.partial_tree)
            # print("delta:", delta)

            k += delta

        if suc:
            print("delta stats:", delta_stats)
            print("k=",k, "solution is found: ", self.partial_tree.to_py())
        else:
            print("No solution exists!")
    

def run_on_bench():
    fpath = sys.argv[1]
    # load data
    with open(fpath, 'r') as f:
        lines = f.read().splitlines()
        res = []
        for l in lines:
            if ";" in l or "(declare-var" in l:
                continue
            res.append(l)
        content = "\n".join(res)
    
    sygus_instance = SygusInstance(content)
    
    spec = sygus_instance.get_spec()
    grammar = sygus_instance.get_grammar()
    print("search_space: ", grammar.count_dict[grammar.start])
    qbfs = QBFSearch(spec, grammar)
    qbfs.synthesize()


import math
class BenchStats(object):
    def __init__(self, name, sy_tree):
        self.G = sy_tree.grammar

        self.name = name
        self.space = self.get_space()
        self.depth = self.get_depth()
        self.num_var = self.get_num_var()

    def get_space(self):
        x = self.G.count_dict[ self.G.start ]
        return math.ceil(math.log(x, 10))

    def get_num_var(self):
        return len(self.G.terminals)

    def get_depth(self):
        return len(self.G.productions)

    def get_rep(self):
        return (self.name, self.space, self.depth, self.num_var)


if __name__ == '__main__':
    tic()

    dataset = Dataset()
    benchmarks = dataset.spec_list
    bench_names = dataset.file_names
    nb = len(benchmarks)


    res = []
    pbar = tqdm(range(nb))
    for k in pbar:
        bench = benchmarks[k]
        name = bench_names[k]
        # print("process bench: ", name, "space: ", bench.grammar.count_dict['Start'])


        qbfs = QBFSearch(bench.spec, bench.grammar)
        qbfs.synthesize()
        print("count_stats:", bench.grammar.count_dict)
        pbar.set_description('processed : %.4f %%' % (100.0 * (k+1) / nb) )

        # res.append( BenchStats(name, bench) )

    # for x in res:
    #     print( x.get_rep() )
    # import pickle
    # pickle.dump(res, open("benchstats.pickle","wb"))