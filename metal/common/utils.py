
from __future__ import print_function

# import sys
import random
import copy
from collections import Counter
from tqdm import tqdm

from metal.common.cmd_args import cmd_args, toc


class StatsCounter(object):
    def __init__(self):
        self.stats_dict = {}
        self.reported = set()

    def add(self, pid, name, delta=1):
        if not pid in self.stats_dict:
            self.stats_dict[pid] = Counter()
        c = self.stats_dict[pid]
        c[name] += delta

    def report(self, pid):
        if not pid in self.stats_dict:
            self.stats_dict[pid] = Counter()
        c = self.stats_dict[pid]
        dur = toc()
        tqdm.write('report time: %.2f pid: %s stats: %s' % (dur, str(pid), str(c)))

    def report_once(self, pid):
        if pid in self.reported:
            return
        self.reported.add(pid)
        self.report(pid)

    def report_global(self):
        t = Counter()
        for key in self.stats_dict:
            c = self.stats_dict[key]
            for k in c:
                t[k] += c[k]
        tqdm.write('global_stats: %s' % str(t))

stat_counter = StatsCounter()


def py_eval_helper(env, expr_root):
    '''Evaluate SynExp expr_root with given variable assignment.

    Args:
        env: variable name to value mapping
        expr_root: an expression of type SynExp
    Returns:
        evaluation result, True or False
    Raises:
        exceptions that might be raised by python exec() / eval()

    '''
    return expr_root.eval_py(env)

    py_exp = expr_root.to_py()

    # print("env:", env)
    # print("expr_root", expr_root)
    # print("py_exp:", py_exp)
    vs_in_exp = expr_root.get_vars()
    for key in env:
        vs_in_exp.discard(key)
        val = env[key]
        exec( key + '=' + ('True' if val else 'False')  )
    for v in vs_in_exp:
        # assign random initialization value (e.g. False) for 
        # variables that do not appear in env
        exec( v + "=False" )
    # print("py_exp:", py_exp)
    return eval( py_exp )

class CounterExample(object):
    def __init__(self, ce_expr, kind, ce_model):
        self.instance = ce_expr
        self.kind = kind
        self.config = ce_model

        self.ce_str = self.to_ce_str()
    
    def update_ce_str(self):
        self.ce_str = self.to_ce_str()

    def to_ce_str(self):
        ts = []
        fs = []
        for k in self.config:
            if self.config[k]:
                ts.append(k)
            else: 
                fs.append(k)
        ts.sort()
        fs.sort()
        return 'T{' + ','.join(ts) + "}F{" + ','.join(fs) + "}"


    def check(self, expr_root):
        if self.kind == "T":
            # we should get a SAT result for positive example
            if py_eval_helper(self.config, expr_root):
                return "good"
            else:
                return "bad"

        elif self.kind == "F":
            # we should get an UNSAT result for negative example
            if not py_eval_helper(self.config, expr_root):
                return "good"
            else:
                return "bad"                

        print("kind:", self.kind)
        raise Exception("check crashed")


class ReplayMem(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size

        self.ce_list = []

        self.count = 0
        self.current = 0
        self.hist_set = set()

    def add(self, ce):
        if ce.ce_str in self.hist_set:
            return
        self.hist_set.add(ce.ce_str)
        if len(self.ce_list) <= self.current:
            self.ce_list.append(ce)
        else:
            self.hist_set.remove(self.ce_list[self.current].ce_str)
            self.ce_list[self.current] = ce
        
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size
    
    def sample(self, num_samples):
        if num_samples >= self.count:
            return self.ce_list

        sampled_ce = []

        for i in range(num_samples):
            idx = random.randint(0, self.count - 1)
            sampled_ce.append(self.ce_list[idx])
        
        return sampled_ce

class CEHolder(object):
    def __init__(self, sample):
        self.sample = sample
        self.ce_per_key = {}
        #self.warmup()
        self.all_ces = []
        self.ce_map = {}
        self.warmup2()


    def interpolate_ce(self, ce):
        n = len(ce.config)
        p = 1. / n
        res = []
        isamples = n if cmd_args.interpolate_samples < 1 else cmd_args.interpolate_samples

        for _ in range(isamples):
            new_ce = copy.deepcopy(ce)
            ks = new_ce.config.keys()
            for k in ks:
                # randomly flip
                if random.random() < p:
                    old_val = new_ce.config[k]
                    new_ce.config[k] = not old_val

            new_ce.update_ce_str()
            label = self.ce_map[ new_ce.ce_str ]
            new_ce.kind = label

            self.add_ce(label, new_ce)
            res.append(new_ce)

        return res

    def get_failed_ce(self, expr_root):
        for ce in self.all_ces:            
            if ce.check(expr_root) == "good":
                continue
            return (-1, ce.kind, ce)
        return (1, None, None)

    def warmup2(self):
        Ts,Fs = self.sample.spectree.all_tests

        ss = [ ('T', x) for x in Ts] + [ ('F', x) for x in Fs]
        random.shuffle(ss)

        # save all ces
        for x in ss:
            ce = CounterExample(None, x[0], x[1])
            self.all_ces.append( ce )
            self.ce_map[ ce.to_ce_str() ] = x[0]

        # warm up samples
        n = len(ss)
        i = 0
        while i < n and i < cmd_args.init_samples:
            key, model = ss[i]
            ce = CounterExample(None, key, model)
            self.add_ce(key, ce) 
            i += 1

    def warmup(self):
        Ts,Fs = self.sample.spectree.all_tests
        key = 'T'
        for model in Ts:
            ce = CounterExample(None,key,model)
            self.add_ce(key, ce)

        key = 'F'
        for model in Fs:
            ce = CounterExample(None,key,model)
            self.add_ce(key, ce)

    def add_ce(self, key, ce):                
        if key not in self.ce_per_key:
            self.ce_per_key[key] = ReplayMem(cmd_args.replay_memsize)
        
        mem = self.ce_per_key[key]
        mem.add(ce)
        # print("mem.count:", mem.count)

    def eval(self, key, expr_root):
        if not cmd_args.use_ce:
            return 1.0
        if not key in self.ce_per_key:
            return 0.0
        mem = self.ce_per_key[key]
        samples = mem.sample(cmd_args.ce_batchsize)
        assert len(samples)

        stat_counter.add(self.sample.sample_index, 'ce-' + key, len(samples))
        s = 0.0
        for ce in samples:            
            if ce.check(expr_root) == "good":
                s += 1.0
        return s / len(samples)

    def eval_both(self, expr_root):
        passed_ct = 0
        all_ct = 0
        for key in self.ce_per_key:
            mem = self.ce_per_key[key]
            all_ct += mem.count

            for ce in mem.ce_list:
                if ce.check(expr_root) == "good":
                    passed_ct += 1
        
        return (passed_ct, all_ct)


    def eval_count(self, expr_root):
        ct = 0
        #py_exp = expr_root.to_py()
        for key in self.ce_per_key:
            mem = self.ce_per_key[key]
            samples = mem.sample(cmd_args.ce_batchsize)
            assert len(samples)
            for ce in samples:            
                if ce.check(expr_root) == "good":
                    ct += 1
        return ct

    def eval_detail(self, key, expr_root):
        if not key in self.ce_per_key:
            return

        mem = self.ce_per_key[key]
        samples = mem.sample(cmd_args.ce_batchsize)
        assert len(samples)
        #py_exp = expr_root.to_py()

        print("key:", key)
        for ce in samples:            
            print(">>  ", ce.check(expr_root), "  ", ce.config)

    def show_stats(self):
        for x in self.ce_per_key:
            print("key: ", x, ", size: ", self.ce_per_key[x].count )