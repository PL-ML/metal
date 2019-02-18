#!/usr/bin/env python3

from pysat.solvers import Minisat22

from metal.solver.tseitin_cnf import TseitinCNF
from metal.parser.sygus_parser import SyExp

#
# this is bit tricky, as we are dealing with constraint with universal quantifier: forall x,y,z,  phi(x,y,z)
# while SAT engine is to solve constraint with existential quantifier:  exists x,y,z, phi(x,y,z)
# 
# first, find an satisfiable solution for:  NOT (entire formulation)
# then, 1) extracts truth value for primitive variables (that are not created during Tseitin transformation)
#       2) replace the primitive variable with their corresponding truth values
# finally, the updated formula should be unsatisfiable, now we can extract an UNSAT core
# 
# the good thing is that the UNSAT core is in the high-level (a nice coincidence of using Tseitin transformation!)
# 
# Perhaps, we can even iteratively repeat such a process, so that the UNSAT core will be higher level abstraction
#

def compute_unsat_core(clauses, start_index):
    # attach selector literal
    # print("clauses:", clauses)
    clauses_with_indices = []
    index = start_index
    refs = []
    for cl in clauses:
        clauses_with_indices.append( cl + [ -index ] )
        refs.append(index)
        index += 1

    # solve with minisat
    with Minisat22(bootstrap_with= clauses_with_indices) as m:
        status = m.solve(assumptions=refs)
        # print("status:", status)
        unsat_core = m.get_core()
        if unsat_core is None:
            model = m.get_model()
            # print("start_index:", start_index)
            # print("sat model size:", len(model) )
            # print("model:", model)
            return True, [ x for x in model if abs(x) < start_index  ] 
        else:
            return False, sorted([ x - start_index for x in unsat_core])

def compute_counter_example(clauses, overall_index, primitive_vars):
    # wrap with NOT, convert universal quantifier to existential quatifier
    x = overall_index + 1
    # constraints for:  x <==> not p
    clauses.append( [-x, -overall_index] )
    clauses.append( [ x,  overall_index] )
    clauses.append( [ x ] ) # assert the entire constraint 

    # invoke minisat wrapper
    status, res = compute_unsat_core(clauses, x+1)

    # pop the last three introduced above
    clauses.pop(-1)
    clauses.pop(-1)
    clauses.pop(-1)

    ce = None
    if status:
        # minisat returns SAT, an counterexample is found
        ce = set([v for v in res if abs(v) in primitive_vars])

    return status, ce

def compute_invalid_core(clauses, overall_index, primitive_vars):
    status, ce = compute_counter_example(clauses, overall_index, primitive_vars)

    #clauses.append( [ overall_index ] ) # assert the entire constraint 

    if status:
        
        # treat overall_id as primitive so that the invalid does not contain entire constraint
        # TODO: does this really make sense?
        #primitive_vars.add( overall_index ) 
        
        #assert overall_index not in primitive_vars


        ce.add( overall_index )

        # replace primitive vars with truth values of the found CE
        clauses_wo_prim = []
        for cl in clauses:
            if len( [x for x in cl if x in ce] ) > 0:
                # trivially true, put a place hoder
                clauses_wo_prim.append( [overall_index+1] )
                continue

            update_cl = []
            for x in cl:
                if -x in ce:
                    continue
                update_cl.append(x)
            ##  what if update_cl is empty?? 
            # this is impossible, because we find a SAT solution, 
            # each individual clause should be satisfiable
            assert update_cl != []

            clauses_wo_prim.append( update_cl )

        # print("primitive_vars:", primitive_vars)
        # derive an UNSAT core in the high level
        s2, r2 = compute_unsat_core( clauses_wo_prim, overall_index + 2 )
        assert s2 == False

        return r2
    else:
        # minisat cannot find CE, synthesis is DONE
        return None


class SatProxy(object):

    def __init__(self, spec, syn_exp):
        assert isinstance(spec, SyExp)
        assert isinstance(syn_exp, SyExp)
        
        self.spec = spec
        self.syn_exp = syn_exp
        #self.spec_tseitin_cnf = None
        #self.syn_exp_tseitin_cnf = None

        eqv = SyExp("eqv", [self.spec, self.syn_exp])
        self.overall_exp = eqv.simplify_bdd()
        self.overall_cnf = TseitinCNF(self.overall_exp, 1)

    def find_counter_example(self):
        clauses = self.overall_cnf.get_clauses()
        prim_vars, prim_names = self.overall_cnf.get_primitive_vars()
        overall_id = self.overall_cnf.get_overall_id()

        status, ce = compute_counter_example(clauses, overall_id, prim_vars)

        if not status:
            return status, ce

        model = {}
        for v in ce:
            vid = abs(v)
            assert vid in prim_names
            model[ prim_names[vid] ] = (v > 0)
        assert len(prim_names) == len(model)
        return status, model

    def solve(self):
        clauses = self.overall_cnf.get_clauses()
        prim_vars = self.overall_cnf.get_primitive_vars()
        overall_id = self.overall_cnf.get_overall_id()

        print("overall id:", overall_id)

        invalid_core = compute_invalid_core(clauses, overall_id, prim_vars)

        # find clauses generated from specification (first K clauses)
        split_id = overall_id
        K = 0
        found = False
        for cl in clauses:
            for x in cl:
                if abs(x) == split_id:
                    found = True
                    break
            if found:
                break
            K += 1
        assert K < len(clauses)

        if invalid_core is None:
            return None,None
        
        # print("invalid_core:", invalid_core)
        # for x in invalid_core:
        #     print(clauses[x])
        #
        # for x in invalid_core:
        #     print(x, " refers: ", self.overall_cnf.interpret(x) )

        invalid_core_in_spec = [ x for x in invalid_core if x < K]
        invalid_core_in_syn = [x for x in invalid_core if x >= K]
        return invalid_core_in_spec, invalid_core_in_syn 


if __name__ == '__main__':
    print("dummy")