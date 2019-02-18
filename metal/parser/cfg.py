#!/usr/bin/env python3

from metal.parser.sygus_parser import SyExp

class CFG(object):

    # sexp: ( (rules ..) (rules ..) )
    def __init__(self, sy_exp):
        self.productions = dict()
        self.nonTerminals = set()
        self.terminals = set()
        self.start = "Start"

        # print("syexp: ", sy_exp)

        for se in sy_exp.get_args():
            assert len(se.args) == 2

            nt = se.get_app()
            self.nonTerminals.add( nt )

            typ = se.args[0] # type information, useless for now
            # print("typ: ", typ)
            derivations = []
            if se.args[1].app != "":
                derivations.append( SyExp(se.args[1].app, []) )
            derivations.extend( se.args[1].get_args() )
            # print("app: ", se.args[1].app, len(se.args[1].app))
            res = []
            for prod in derivations:
                # print("prod:", prod)
                #print("prod.app = ", prod.app)
                #assert prod.app in ["and", "or", "xor", "not"]
                body = [prod.app] + [ x.app for x in prod.args]
                res.append( body )

                for x in body:
                    self.terminals.add(x)

            self.productions[nt] = res

        for x in self.nonTerminals:
            if x in self.terminals:
                self.terminals.remove(x)
    
        self.count_dict = {}
        self.compute_count(self.start)



    def __str__(self):
        s = []
        for nt in self.productions:
            bs = []
            for body in self.productions[nt]:
                bs.append( ' '.join(body) )
            s.append( nt + " -> " + " | ".join(bs))

        return "\n".join(s)

    def compute_count(self, t):
        if t in self.count_dict:
            return self.count_dict[t]

        if t in self.terminals:
            self.count_dict[t] = 1
            # print("compute_count, t:",t, "res:", 1)
            return 1

        res = 0
        rules = self.productions[t]
        for r  in rules:
            tmp = 1
            for i in range(1, len(r)):
                tmp *= self.compute_count( r[i] )
            res += tmp
        
        self.count_dict[t] = res

        # print("rules:", rules)
        # print("compute_count, t:",t, "res:", res)
        return res
        



'''
    # prods:  [ (T,[T]]) ]
        for (hd,body) in prods:
            if hd not in self.nonTerminals:
                self.nonTerminals.add(hd)
            if hd in self.terminals:
                self.terminals.remove(hd)
            for x in body:
                if x not in self.nonTerminals:
                    self.terminals.add(x)

            if hd not in self.productions:
                self.productions[hd] = []

            self.productions[hd].append( body )
'''
