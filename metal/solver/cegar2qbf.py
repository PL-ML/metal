
#!/usr/bin/env python3

import sys
import os 

from pysat.solvers import Minisat22
from metal.parser.sygus_parser import SyExp
from metal.solver.tseitin_cnf import TseitinCNF


def deduplicate(cnf):
    tmp = set()
    for dnf in cnf:
        tmp.add( tuple(dnf) )
    
    return [  list(dnf) for dnf in  tmp ]

def fill_in_simplify(cnf, partial_sol):
    cl = []
    for dnf in cnf:
        tmp = []
        for x in dnf:
            if -x in partial_sol:
                continue
            if x in partial_sol:
                tmp = []
                break
            tmp.append(x)

        if len(tmp) > 0:
            cl.append(tmp)
    return cl

def negate_clauses(clauses, start_id):
    # if the original clauses are Tseitin transformed, after reducing,
    # there should be at most two universal vars in each clause
    
    # optimization:
    # 1) for unit clause, no fresh variable is needed     # bug: [[1,2], [1]] for opt-1
    # 2) for clause with two vars, apply standard tseitin transformation: 
    '''
    x <--> p /\ q
    * (not x) \/ p
    * (not x) \/ q
    * not p \/ not q \/ x
    '''
    # not (a \/ b)  === fresh_var
    # p = -a,  q=-b,  frewsh_var=x

    res = []
    vs = []
    for cl in clauses:
        assert len(cl) >=1
        assert len(cl) <=2
        if len(cl) == 1:
            # res.append( [ -cl[0] ] )
            res.append( [-start_id, -cl[0]] )
            res.append( [start_id, cl[0]] )
        else:
            res.append( [-start_id, -cl[0]] )
            res.append( [-start_id, -cl[1]] )
            res.append( [start_id, cl[0], cl[1]] )

        vs.append(start_id)
        start_id += 1

    # if len(vs) > 0: a bug, len(vs) = len(clauses)

    res.append(vs)    
    return res, start_id


def qbf(v_forall, F):
    ''' Implementation of the CEGAR 2-QBF algorithm  forall x, exists y, F
    For efficiency reason, x and y would better be sets
    '''
    syn_man = []
    ver_man = F

    max_id = 1 + max( [ max( [abs(x) for x in cnf] ) for cnf in F ]  )

    universal_clauses = []
    for dnf in F:
        for x in dnf:
            if abs(x) in v_forall:
                universal_clauses.append(dnf)
                break
    
    # print("universal clauses:", universal_clauses)

    gas = 16
    i = 0
    while i < gas:
        sol_x = None
        sol_y = None
        
        # find solution for synMan
        if syn_man == []:
            sol_x = v_forall
        else:
            with Minisat22(bootstrap_with= syn_man) as m:
                if m.solve():
                    sol_x = m.get_model()
            if sol_x is None:
                return True

        sol_x = list( filter(lambda x: abs(x) in v_forall,  sol_x) )
        # print("sol_x", sol_x)

        # find solution for VerMan
        with Minisat22(bootstrap_with= ver_man) as m:
            if m.solve(assumptions=sol_x):
                sol_y = m.get_model()
        if sol_y is None:
            return False

        # print("sol_y", sol_y)

        sol_y = set( filter(lambda x: abs(x) not in v_forall,  sol_y) )

        # update synMan
        # 1. fill in values in sol_y and reduce universal_clauses
        cl = fill_in_simplify(universal_clauses, sol_y)

        # print("cl:", cl)

        # 2. tseitin transform  "not cl"                
        res, max_id = negate_clauses(cl, max_id)

        # print("res:", res)
        if res == []:
            # no more cases for x need to be considered, done
            return True

        for dnf in res:
            syn_man.append(dnf)

        # print("syn_man:", syn_man)
        syn_man = deduplicate(syn_man)
        # print("syn_man:", syn_man)
        i += 1

#    print("run out of gas!!")
    return True    

def deepcopy(sexp):
    res = SyExp(sexp.app, [])
    for x in sexp.args:
        res.args.append(  deepcopy(x) )
    return res

nonT = set( ["nonT", "depth1", "depth2", "depth3", "depth4", "depth5", "depth6", "depth7", "depth8", "depth9", "depth10", "depth11", 
"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"] )

def rename_nonT(sexp, sid=0):
    if sexp.args == []:
        if sexp.app in nonT:
            return SyExp(sexp.app + ("-%d" % sid),[]), sid+1
        else:
            return sexp, sid

    assert sexp.app not in nonT

    args = []
    for x in sexp.args:
        a, sid = rename_nonT(x, sid)
        args.append(a)
    
    sexp.args = args
    return sexp, sid


class CegarQBF(object):
    def __init__(self, spec, partial_tree):

        # extract forall vars (assume all of which are in spec)
        vnames = spec.get_vars()
        # print("vnames:", vnames)

        ptree = deepcopy(partial_tree)
        ptree, sid = rename_nonT(ptree)
        eqv = SyExp("eqv", [spec,  ptree])
        eqv = eqv.simplify_bdd()
        # print("eqv:", eqv.show2(0))

        # get cnf
        t = TseitinCNF(eqv)
        n2v = t.get_name2var()
        # print("names:", n2v)
        self.v_forall = [n2v[name] for name in vnames]

        cnf = t.cnfs
        cnf.append( [t.get_overall_id()] )
        self.cnf = cnf

        # print("cnf:", cnf)

    def any_hope(self):
        return qbf(self.v_forall, self.cnf)
