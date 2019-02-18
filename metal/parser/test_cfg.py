#!/usr/bin/env python3
from __future__ import print_function

from sygus_parser import SyExp, parse_sexp
from cfg import CFG

import os
import sys

from metal.common.grammar_graph_builder import GrammarGraph

#from tseitin_cnf import TseitinCNF

def R(fpath):
    with open(fpath, "r") as fin:
        return fin.read()

if __name__ == '__main__':

    import os
    from os.path import join as joinpath

    # dir_path = 'C:/work/research/prj/ce_guided_rl/benchmarks/CrCi'
    # for filename in os.listdir(dir_path):
    #     with open(joinpath(dir_path, filename), 'r') as f:
    #         res = parse_sexp(f.read())
    #         for x in res:
    #             if x.app != 'synth-fun':
    #             # if x.app != 'define-fun':
    #                 continue
    #             print(x.get_args()[3])
    #             prods = CFG(x.get_args()[3])
    #             print(prods)



    import sys
    if len(sys.argv) != 2:
        print("usage: ", sys.argv[0], "a.sl")
        exit()
    
    f = R(sys.argv[1])
    res = parse_sexp(f)
    
    for x in res:
        # parse and print the context free grammar
        if x.get_app() == "synth-fun":
            # continue
    
            prods = x.get_args()[3]
            print("CFG grammar:")
            cfg = CFG(prods)
            # print(CFG(prods) )

            # g = GrammarGraph(cfg)
            # print(g.num_nodes(), g.num_edges())
            # g.dump_dot()
            continue
    
        # print logic specification
        if x.get_app() == "define-fun":
            continue
            print("logic spec:")
            print( x.get_args()[3] )
    
            #t = TseitinCNF(x.args[3], 1)
            #t.show()

