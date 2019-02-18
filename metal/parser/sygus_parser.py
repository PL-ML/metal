#!/usr/bin/env python3


class SyExp(object):
    def __init__(self, app, args):
        if app == '=':
            app = '=='
        self.app = app
        self.args = args

    def __str__(self):
        return self.show(0)
        #return self.show2(0)

    def get_app(self):
        return self.app
    
    def get_args(self):
        return self.args
        
    def show(self, k):
        if len(self.args) == 0:
            return  "  " * k + self.app
        else:
            indent = "  " * k
            res = indent + "(" + self.app + "\n"
            args = "\n".join( [a.show(k+1) for a in self.args]  )
            res += args
            res += "\n" + indent + ")"
            return res

    def show2(self, k):
        if len(self.args) == 0:
            return  "  " * k + self.app + '_' + str(self.__repr__())
        else:
            indent = "  " * k
            res = indent + "(" + self.app + '_' + str(self.__repr__()) + "\n"
            args = "\n".join( [a.show2(k+1) for a in self.args]  )
            res += args
            res += "\n" + indent + ")"
            return res

    def collect_stats(self, stats):
        if self.app == '+' or self.app == '-' or self.app == '*' or len(self.args)  == 0:
            return

        if self.app not in stats:
            stats[self.app] = 0
        stats[self.app] += 1
        for a in self.args:
            a.collect_stats(stats)

    def to_str2(self):
        if self.app == 'and':
            return "(" + " && ".join( [a.to_str() for a in self.args] ) + ")"
        elif self.app == 'or':
            return "(" + " || ".join( [a.to_str() for a in self.args] ) + ")"
        elif self.app == 'not':
            assert len(self.args) == 1
            return "( !(%s) )" % self.args[0].to_str()
        elif len(self.args) == 0:
            return "(%s)" % self.app
        else:
            #print("debug: ", self.app, self.args)
            assert len(self.args) == 2
            return "(%s %s %s)" % ( self.args[0].to_str(), self.app, self.args[1].to_str() )

    def to_py(self):
        ''' Convert the expression to python format
        '''
        if self.app == 'and':
            assert len(self.args) == 2
            return "(" + " & ".join( [a.to_py() for a in self.args] ) + ")"
        elif self.app == 'xor':
            assert len(self.args) == 2
            return "(" + " ^ ".join( [a.to_py() for a in self.args] ) + ")"
        elif self.app == 'or':
            assert len(self.args) == 2
            return "(" + " | ".join( [a.to_py() for a in self.args] ) + ")"
        elif self.app == 'not':
            assert len(self.args) == 1
            return "(not (%s) )" % self.args[0].to_py()
        else:
            assert len(self.args) == 0
            return self.app

    def eval_py(self, env):
        if self.app == 'and':
            assert len(self.args) == 2
            return self.args[0].eval_py(env) & self.args[1].eval_py(env)
        elif self.app == 'xor':
            assert len(self.args) == 2
            return self.args[0].eval_py(env) ^ self.args[1].eval_py(env)
        elif self.app == 'or':
            assert len(self.args) == 2
            return self.args[0].eval_py(env) | self.args[1].eval_py(env)
        elif self.app == 'not':
            assert len(self.args) == 1
            return (not self.args[0].eval_py(env))
        else:
            assert len(self.args) == 0
            # if self.app not in env:
            #     print("dbg eval_py, env:", env)
            return env[self.app]

    def get_vars(self):
        res = set()
        if len(self.args) == 0:
            res.add(self.app)
        else:
            for x in self.args:
                res |= x.get_vars()
        return res



    def extract_numbers(self, st):
        if len(self.args) > 0:
            for a in self.args:
                a.extract_numbers(st)
        else:
            if self.app.isdigit() or ( self.app[0] == '-' and self.app[1:].isdigit() ):
                st.add( self.app )


    def equiv_to(self, other):
        ''' Test equivalence between two SyExp objects
        '''
        if self == other:
            return True

        if self.app == other.app:
            if len(self.args) == len(other.args):
                n = len(self.args)
                for i in range(n):
                    if not self.args[i].equiv_to( other.args[i]):
                        return False
                return True
        return False

    def simplify_bdd(self, pool=None):
        if pool is None:
            pool = set()

        new_args = []
        for x in self.args:
            new_args.append( x.simplify_bdd(pool) )
        self.args = new_args
        
        for p in pool:
            if self.equiv_to(p):
                return p
        
        pool.add(self)
        return self

    def dfs_find(self, name):
        if self.app == name:
            return True, self
        
        for x in self.args:
            s,r = x.dfs_find(name)
            if s: 
                return s,r
        return False, None
                



def match_p(s, i):
    L = len(s)
    ct = 0 
    while i < L:
        c = s[i]
        if c == '(':
            ct += 1
        elif c == ')':
            ct -= 1
            if ct == 0:
                return i
        i += 1

    return -1

def separate_p(s):
    return  s.replace('(',' ( ').replace(')',' ) ')

def parse_sexp(s):
    s = separate_p(s)
    #print("s:", s)
    vs = s.split()
    if len(vs) == 0:
        return []

    if vs[0] == '(':
        app = vs[1]  # app name follows the left parenthesis

        # stop when reaching the last statement, which is assumed to be check-synth
        if app.startswith("check-synth"):
            hd = SyExp("check-synth",[])
            return [hd]

        # locate the right parenthesis
        r = match_p(vs, 0)
        assert r > 0

        if app == "(":
            #app = "_TUPLE_"
            app = ""
            args = parse_sexp( " ".join(vs[1:r]) )
        else:
            args = parse_sexp( " ".join(vs[2:r]) )

        hd = SyExp(app, args)
        res = parse_sexp(" ".join(vs[r+1:]) )
        return [hd] + res

    else:

        hd = SyExp(vs[0],[])
        res = parse_sexp(" ".join(vs[1:]))
        return [hd] + res