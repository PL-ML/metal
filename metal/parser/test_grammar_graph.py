#!/usr/bin/env python3
from __future__ import print_function

from sygus_parser import SyExp, parse_sexp
from cfg import CFG

import os
import sys

from metal.common.grammar_graph_builder import GrammarGraph
from metal.common.spec_tree import SygusInstance


def R(fpath):
    with open(fpath, "r") as fin:
        return fin.read()

if __name__ == '__main__':

    import os
    import sys
    if len(sys.argv) != 2:
        print("usage: ", sys.argv[0], "a.sl")
        exit()
    
    f = R(sys.argv[1])

    si = SygusInstance(f)
    g = GrammarGraph( si )
    # print(g.num_nodes(), g.num_edges())
    g.dump_dot()
   
