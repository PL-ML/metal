#!/usr/bin/env python3

import sys
import os 

from metal.parser.sygus_parser import SyExp

class Namespace(object):
    def __init__(self, start_id):
        self.next_id = start_id
        self.exp_to_id = {}
        self.id_to_exp = {}
    
    def __str__(self):
        res = ""
        for x in self.id_to_exp:
            res += "%d -> %s\n" % (x, self.id_to_exp[x].app) 
        return res

    def get_id(self, exp):
        if exp not in self.exp_to_id:
            self.exp_to_id[exp] = self.next_id
            self.id_to_exp[self.next_id] = exp
            self.next_id += 1

        return self.exp_to_id[exp]
    
    def exists(self, exp):
        return exp in self.exp_to_id

    def exists_id(self, vid):
        return vid in self.id_to_exp

    def get_primitive_ids(self):
        res = set()
        for exp in self.exp_to_id:
            if len(exp.args) == 0:
                res.add( self.exp_to_id[exp] )
        return res

    def get_primitive_id_name(self):
        ids = set()
        names = {}
        for exp in self.exp_to_id:
            if len(exp.args) == 0:
                vid = self.exp_to_id[exp]
                vname = exp.app
                ids.add( vid )
                names[vid] = vname
        return ids,names

    def get_n2v(self):
        n2v = {}
        for exp in self.exp_to_id:
            if len(exp.args) == 0:
                vid = self.exp_to_id[exp]
                vname = exp.app
                n2v[vname] = vid
        return n2v

    def interpret(self, cl):
        if len(cl) == 0:
            return SyExp("empty",[])

        if len(cl) == 1:
            x =  cl[0]
            assert self.exists_id( abs(x) )
            if x > 0:
                return self.id_to_exp[ x ]
            else:
                exp = self.id_to_exp[ -x ]
                return SyExp("not", [exp])

        neg = []
        pos = []
        for x in cl:
            assert self.exists_id( abs(x) )
            if x > 0:
                pos.append(x)
            else:
                neg.append(-x)

        if pos == []:
            return SyExp("at lest one should be false",[ self.id_to_exp[x] for x in neg ] )

        if neg == []:
            return SyExp("at least one should be true" ,[ self.id_to_exp[x] for x in pos ])

        neg_exp = None
        pos_exp = None
        if len(neg) == 1:
            neg_exp = self.id_to_exp[ neg[0] ]
        else:
            neg_exp = SyExp("AND",[ self.id_to_exp[x] for x in neg ] )

        if len(pos) == 1:
            pos_exp = self.id_to_exp[ pos[0] ]
        else:
            pos_exp = SyExp("OR" ,[ self.id_to_exp[x] for x in pos ])
    
        return SyExp("=>", [neg_exp, pos_exp] )

'''

x <--> p /\ q

x --> p /\ q  =  not x \/ (p /\ q) = (not x \/ p)  /\ (not x \/ q)
p /\ q --> x

* (not x) \/ p
* (not x) \/ q
* not p \/ not q \/ x


x <--> p \/ q
x --> p \/ q  = not x \/ p \/ q
p \/ q --> x = (not p /\ not q) \/ x  =  (not p \/ x) /\ (not q \/ x)

* not x \/ q \/ q
* not p \/ x
* not q \/ x


x <--> p xor q

# run test_cnf.py we get:
(-x | p | q) & (-x | -q | -p) & (p | x | -q) & (q | x | -p)


x <--> not p

x --> not p =  not x \/ not p
not p --> x = p \/ x

* not x \/ not p
* p \/ x


x <--> y
* not x \/ y
* not y \/ x
'''


def tseitin_transformation(exp, env):
    if env.exists(exp):
        # exp has been processed earlier, no need to add duplicated constraints
        return []

    assert len(exp.args) <= 2
    op = exp.get_app()

    if op not in ["and", "or", "xor", "not", "eqv"]:
        # print(exp)
        assert len(exp.args) == 0
        return []
    
    args_cnfs = []
    for x in exp.args:
        #cnfs.extend( tseitin_transformation(x, env) )
        args_cnfs.append(  tseitin_transformation(x, env) )

    cnfs = []

    if op == "not":
        assert len(exp.args) == 1
        p = env.get_id( exp.args[0] )
        x = env.get_id( exp ) # make sure the entire express correspond to largest ID (compared to sub-expressions)
        cnfs.append( [ x,  p] )
        cnfs.append( [-x, -p] )
    else:
        assert len(exp.args) == 2
        p = env.get_id( exp.args[0] )
        q = env.get_id( exp.args[1] )
        x = env.get_id( exp ) # make sure the entire express correspond to largest ID (compared to sub-expressions)

        if op == "and":
            cnfs.append( [-x,  p] )
            cnfs.append( [-x,  q] )
            cnfs.append( [ x, -p, -q] )
        elif op == "or":
            cnfs.append( [ x, -p] )
            cnfs.append( [ x, -q] )
            cnfs.append( [-x,  p, q] )
        elif op == "xor":
            cnfs.append( [-x, -p, -q] )
            cnfs.append( [-x,  p,  q] )
            cnfs.append( [ x, -p,  q] )
            cnfs.append( [ x,  p, -q] )
        elif op == "eqv":
            # x is useless here
            # cnfs.append( [ x ] ) # marker used to split clauses
            # cnfs.append( [-p,  q] )
            # cnfs.append( [ p, -q] )

            cnfs.append( [ x,  p,  q])
            cnfs.append( [ x, -p, -q])
            cnfs.append( [-x,  p, -q])
            cnfs.append( [-x, -p,  q])
        else:
            assert False
    
    res = args_cnfs[0] + cnfs
    if len(args_cnfs) > 1:
        res += args_cnfs[1]
    return res



class TseitinCNF(object):
    def __init__(self, exp, start_id=1):
        assert start_id > 0

        self.exp = exp
        self.env = Namespace(start_id)
        self.cnfs = tseitin_transformation(exp, self.env)
 
    def show(self):
        print("=== TseitinCNF instance ===")
        print("exp:")
        print(self.exp)
        print("naming environment:")
        print(self.env)
        print("cnfs:")
        for x in self.cnfs:
            print(x)
        print("======")

    def get_name2var(self):
        return self.env.get_n2v()

    def get_primitive_vars(self):
        return self.env.get_primitive_id_name()

    def get_clauses(self):
        return self.cnfs

    # default assumption: overall id should be the largest variable ID
    def get_overall_id(self):
        assert self.env.exists(self.exp)
        return self.env.get_id( self.exp )

    # def get_first_child_id(self):
    #     assert self.exp.app == "eqv"
    #     spec = self.exp.args[0]
    #     assert self.env.exists( spec )
    #     return self.env.get_id( spec )

    def interpret(self, K):
        assert K < len(self.cnfs)
        cl = self.cnfs[K]

        print("interpret cl: ", cl)

        return self.env.interpret( cl )



if __name__ == '__main__':
    #  (or (not (xor (xor  LN231 LN240 ) LN218 ) ) LN336 )
    LN231 = SyExp("LN231", [])
    LN240 = SyExp("LN240", [])
    LN218 = SyExp("LN218", [])
    LN336 = SyExp("LN336", [])
    e1 = SyExp("not", [ SyExp("xor", [SyExp("xor", [LN231, LN240]), LN218]) ] )
    e2 = SyExp("or", [e1, LN336])

    t = TseitinCNF(e2, 1)
    t.show()

