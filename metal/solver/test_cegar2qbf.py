#!/usr/bin/env python3

import os
import sys

from metal.parser.sygus_parser import SyExp
from cegar2qbf import qbf, CegarQBF
from tseitin_cnf import TseitinCNF


def test0():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("and", [ x,y])

    t = TseitinCNF(e1, 1)
    vids, names = t.get_primitive_vars()

    print(vids)
    print(names)

    cnf = t.cnfs
    cnf.append( [t.get_overall_id()] )

    status = qbf( [1], cnf)

    print(e1, cnf)
    print("qbf: ", status)
    assert status is False



def test1a():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("and", [ x,y])
    e2 = SyExp("eqv", [e1, x])
    t = TseitinCNF(e2, 1)
    vids, names = t.get_primitive_vars()

    print(vids)
    print(names)

    cnf = t.cnfs
    cnf.append( [t.get_overall_id()] )

    status = qbf( [1], cnf)

    print(e2, cnf)

    print("qbf: ", status)
    assert status is True

    # forall x,   x /\y === x
    # status = qbf( [2], t.cnfs)
    # print("qbf: ", status)

def test1b():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("and", [ x,y])
    e2 = SyExp("eqv", [e1, x])
    t = TseitinCNF(e2, 1)
    vids, names = t.get_primitive_vars()

    print(vids)
    print(names)

    cnf = t.cnfs
    cnf.append( [t.get_overall_id()] )

    print(e2, cnf)

    # forall x,   x /\y === x
    status = qbf([2], cnf)
    print("qbf: ", status)
    assert status is True

def test2a():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("and", [ x,y])
    e2 = SyExp("and", [ x,  SyExp("not",[y]) ])
    e3 = SyExp("eqv", [e1, e2])
    t = TseitinCNF(e3, 1)
    vids, names = t.get_primitive_vars()

    print(vids)
    print(names)

    cnf = t.cnfs
    cnf.append( [t.get_overall_id()] )

    status = qbf( [1], cnf)

    print(e3, cnf)

    print("qbf: ", status)
    assert status is False

def test2b():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("and", [ x,y])
    e2 = SyExp("or", [ x,  SyExp("not",[y]) ])
    e3 = SyExp("eqv", [e1, e2])
    t = TseitinCNF(e3, 1)
    vids, names = t.get_primitive_vars()

    print(vids)
    print(names)

    cnf = t.cnfs
    cnf.append( [t.get_overall_id()] )

    status = qbf( [1], cnf)
    print(e3, cnf)
    print("qbf: ", status)
    assert status is True

    status = qbf( [1,2], cnf)
    print("qbf: ", status)
    assert status is False


def test3():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("or", [ x,y])
    e3 = SyExp("eqv", [e1, SyExp("not",[x]) ])
    t = TseitinCNF(e3, 1)
    vids, names = t.get_primitive_vars()

    print(vids)
    print(names)

    cnf = t.cnfs
    cnf.append( [t.get_overall_id()] )

    status = qbf( [1], cnf)

    print(e3, cnf)

    print("qbf: ", status)
    assert status is False


def test4():
    x = SyExp("x",[])
    y = SyExp("d1",[])

    c = CegarQBF(x,y)
    print("status: ", c.any_hope())
    assert c

def test5():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("or", [ x,y])
    e2 = SyExp("d1",[])

    c = CegarQBF(e1,e2)
    status = c.any_hope()
    print("status: ", status)
    assert status

def test6():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("or", [ x,y])
    e2 = SyExp("and", [x, SyExp("d1",[])] )

    c = CegarQBF(e1,e2)
    status = c.any_hope()
    print("status: ", status)
    assert status is False

def test7():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("or", [ x,y])
    e2 = SyExp("and", [x,y] )

    c = CegarQBF(e1,e2)
    status = c.any_hope()
    print("status: ", status)
    assert status is False

    c = CegarQBF(e2,e2)
    status = c.any_hope()
    print("status: ", status)
    assert status is True

def test8():
    x = SyExp("x",[])
    y = SyExp("y",[])
    z = SyExp("z",[])
    d1 = SyExp("depth1", [])

    e1 = SyExp("or", [x , SyExp("and", [y,z])  ])
    e2 = SyExp("and", [SyExp("and", [y,y]), d1])

    c = CegarQBF(e1,e2)
    status = c.any_hope()
    print("status: ", status)
    assert status is False

def test9():
    x = SyExp("x",[])
    y = SyExp("y",[])
    z = SyExp("z",[])

    e1 = SyExp("or", [x , SyExp("and", [y,z])  ])
    # e2 = SyExp("and", [SyExp("and", [y,y]), SyExp("and", [y,y])])

    # e1 = SyExp("or", [x,y] )
    e2 = SyExp("and", [y,y])

    c = CegarQBF(e1,e2)
    status = c.any_hope()
    print("status: ", status)
    assert status is False

def test10():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e = SyExp("and", [x,y])
    c = CegarQBF(e,y)
    status = c.any_hope()
    print("status: ", status)
    assert status is False

def test11():
    x = SyExp("x",[])
    y = SyExp("y",[])
    z = SyExp("z",[])
    d1 = SyExp("depth1", [])

    e1 = SyExp("or", [x , SyExp("and", [y,z])  ])
    e2 = SyExp("not", [SyExp("not", [y])])

    c = CegarQBF(e1,e2)
    status = c.any_hope()
    print("status: ", status)
    assert status is False



def test_simplify_bdd():
    x = SyExp("x",[])
    y = SyExp("y",[])
    e1 = SyExp("or", [ x,y])
    e2 = SyExp("and", [SyExp("x", []) , SyExp("d1",[])] )

    e3 = SyExp("eqv", [e1, e2])

    print("before e3: ", e3.show2(0))    
    e4  = e3.simplify_bdd()
    print("after e3: ", e4.show2(0))    

if __name__ == '__main__':
    # test0()
    # test1a()
    # test1b()
    # test2b()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    # test8()
    # test9()
    # test10()
    test11()
    # test_simplify_bdd()
