from __future__ import print_function

import sys
import os

from metal.parser.sygus_parser import SyExp, parse_sexp
from metal.parser.cfg import CFG

from metal.common.constants import AND_TYPE, OR_TYPE, XOR_TYPE, NOT_TYPE, VAR_TYPE

def collect_types_in_preorder(sexp):
    typ = None
    if sexp.app == "and":
        typ = AND_TYPE
    elif sexp.app == "or":
        typ = OR_TYPE
    elif sexp.app == "xor":
        typ = XOR_TYPE
    elif sexp.app == "not":
        typ = NOT_TYPE
    else:
        typ = VAR_TYPE

    res = [ (sexp.app, typ) ]

    for x in sexp.args:
        res.extend( collect_types_in_preorder(x) )
    
    return res


def collect_vars(sexp):
    res = set()
    if sexp.app in ["and", "or", "xor", "not"]:
        for x in sexp.args:
            res = res | collect_vars(x)
    else:
        res.add(sexp.app)
    return res

class SygusInstance(object):
    def __init__(self, s):
        sexp_list = parse_sexp(s)
        self.spec = None
        self.grammar = None
        self.constraint = None
        for s in sexp_list:
            if s.get_app() == "define-fun":
                self.spec = s.get_args()[3]
            elif s.get_app() == "synth-fun":
                self.grammar = CFG(s.get_args()[3])
            elif s.get_app() == "constraint":
                self.constraint = s
            else:
                pass

        assert self.spec
        assert self.grammar
        # print("sygus instance is created")

    def get_spec(self):
        return self.spec

    def get_grammar(self):
        return self.grammar
    

class SpecTree:
    def __init__(self, sygus_instance):
        """
        this class casts the SyExp of a logic spec into list tokens(integers) in order to be fed to encoder

        sygus_instance:
            a SygusInstance object containing the logic spec and the grammar

        """

        self.spec = sygus_instance.get_spec()
        self.grammar = sygus_instance.get_grammar()
        self.vars = list( collect_vars(self.spec) )

        self.node_seq = collect_types_in_preorder(self.spec) # e.g. [('and', AND_TYPE), ('LN29', VAR_TYPE)]
        self.node_type_seq = [t for (n, t) in self.node_seq]
        self.numOf_nodes = len(self.node_type_seq) # int
        self.nodename2ind = dict([(self.node_seq[i][0], i) for i in range(self.numOf_nodes)]) # e.g. {'and':0, 'LN29':1}

        # dump all tinytest cases
        self.all_tests = self.dump_all_tests()
    
    def dump_all_tests(self):
        i, n = 0, len(self.vars)
        m = 2 ** n
        Ts = []
        Fs = []
        while i < m:
            # 1. build env
            env = {}
            for k in range(n):
                env[ self.vars[k] ] = i & (1<<k) > 0
            # 2. evaluate current config
            res = self.spec.eval_py(env)
            if res:
                Ts.append( env )
            else:
                Fs.append( env )

            i += 1
        # print("Ts: ", len(Ts), "Fs: ", len(Fs))
        return (Ts,Fs)
        





def is_tree_complete(ntset, spectree):
    """
    helper function to tinytest if a SpecTree contains non-terminals

    ntset:
        set of names of non-terminals
    spectree:
        SpecTree obj

    return:
        False if contains non-terminal, otherwise True

    """
    return (spectree.app not in ntset) and all([is_tree_complete(ntset, syexp) for syexp in spectree.args])
