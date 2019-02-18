#!/usr/bin/env python3

from __future__ import print_function

from cnf import Variable, Cnf


def tostring(cnf):
    """Convert Cnf object ot Dimacs cnf string
    
    cnf: Cnf object
    
    In the converted Cnf there will be only numbers for
    variable names. The conversion guarantees that the
    variables will be numbered alphabetically.
    """
    varname_dict = {}
    varobj_dict = {}

    varis = set()
    for d in cnf.dis:
        for v in d:
            varis.add(v.name)

    ret = "p cnf %d %d" % (len(varis), len(cnf.dis))

    varis = dict(list(zip(sorted(list(varis)),list(map(str,list(range(1,len(varis)+1)))))))

    for v in varis:
        vo = Variable(v)
        varname_dict[vo] = varis[v]
        varobj_dict[varis[v]] = vo

    for d in cnf.dis:
        ret += "\n"
        vnamelist = []
        for v in d:
            vnamelist.append(("-" if v.inverted else "") + varis[v.name])
        ret += " ".join(vnamelist) + " 0"

    return ret

def test_0():
    v1 = Variable('v1')
    v2 = Variable('v2')
    v3 = Variable('v3')

    exp = v1 & v2 | v3
    exp2 = v1 ^ v2

    exp3 = exp | exp2

    print(exp)
    print(exp2)
    print(exp3)

    print( tostring(exp) )


def show_xor():
    x = Variable("x")
    p = Variable("p")
    q = Variable("q")
    exp = (x >> (p ^ q)) & ( (p ^ q) >> x)
    print( "xor:", exp )

def show_eqv():
    x = Variable("x")
    p = Variable("p")
    q = Variable("q")

    e1 = (p >> q) & (q >> p)
    e2 = (x >> e1) & (e1 >> x)
    print("eqv:", e2 )

def test1():
    a = Variable("a")
    b = Variable("b")
    exp = (a & (-b)) | (b & (- a))
    print("exp:", exp)

def test2():
    a = Variable("a")
    b = Variable("b")
    exp = (a >> (-b)) & (-b >> a)
    print("exp:", exp)

def test3():
    a = Variable("a")
    b = Variable("b")
    exp = (a >> b) & (b >> a)
    print("exp:", exp)
    

if __name__ == '__main__':
    #test_0()
    # show_xor()
    # show_eqv()
    # test1()
    # test2()
    test3()
