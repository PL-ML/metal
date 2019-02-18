#!/usr/bin/env python3

from __future__ import print_function

from pysat.solvers import Solver
from sat import SatProxy

import sys
import os 

from metal.parser.sygus_parser import SyExp, parse_sexp

def get_def():
    import sys
    if len(sys.argv) != 2:
        print("usage: ", sys.argv[0], "a.sl")
        exit()

    with open( sys.argv[1] ) as fin:
        res = parse_sexp( fin.read() )
        for x in res:
            # print logic specification
            if x.get_app() == "define-fun":
                return x.args[3]

    return None

def test_pysat():
    s = Solver()
    s.add_clause([-1,2, -3])  # 3 is a selector for this clause
    s.add_clause([-1,-2, -4])  # 4 is a selector for this clause

    r = s.solve()
    print("r:", r)
    print( s.get_model() )

    s.add_clause([1, -5])  # 5 is a selector for this clause
    r = s.solve(assumptions=[3, 4, 5])  # you need to call the solver assuming these literals are true
    print("r:", r)

    print(s.get_core())  # this should report [3, 4, 5] as a core

def test_simplify():
    spec = get_def()
    syn_exp = SyExp("and", [ SyExp("LN231", []), SyExp("LN218", []) ] )
    overall = SyExp("eqv", [spec, syn_exp])
    print("overall:", overall)
    x = overall.simplify_bdd()
    print("after simplify:", x)
    

def test_satproxy_1():
    #spec = get_def()
    #t = TseitinCNF(spec, 1)
    #t.show()

    spec = SyExp("and", [ SyExp("LN231", []), SyExp("LN218", []) ] )
    #syn_exp = SyExp("and", [ SyExp("LN231", []), SyExp("LN218", []) ] )
    syn_exp = SyExp("xor", [ SyExp("LN231", []), SyExp("LN240", []) ] )

    sp = SatProxy(spec, syn_exp)
    a,b = sp.solve()
    print(a)
    print(b)

def test_satproxy_2():
    spec = SyExp("and", [ SyExp("x", []), SyExp("y", []) ] )
    syn_exp = SyExp("x", [])
    sp = SatProxy(spec, syn_exp)
    a,b = sp.solve()
    print(a)
    print(b)

def test_satproxy_3():
    spec = SyExp("or", [  SyExp("x", []),   SyExp("and", [ SyExp("y", []),SyExp("z", []) ])  ]   )

    syn_exp = SyExp("or", [  SyExp("or", [ SyExp("y", []),SyExp("z", []) ]),   SyExp("and", [ SyExp("y", []),SyExp("z", []) ])  ]   )

    proxy = SatProxy(spec, syn_exp)
    status, ce_model = proxy.find_counter_example()

    # a,b = proxy.solve()
    # print(a)
    # print(b)

    print("status:", status)
    print("ce_model:", ce_model)

if __name__ == '__main__':
    #test_pysat()
    #test_satproxy_2()
    test_satproxy_3()
