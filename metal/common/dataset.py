from __future__ import print_function

import json
import sys
import os
from os.path import join as joinpath
import random
import numpy as np

from metal.common.cmd_args import cmd_args
from metal.common.spec_tree import SygusInstance, SpecTree
from metal.common.grammar_graph_builder import GrammarGraph

from metal.spec_encoder.s2v_lib import S2VLIB, S2VGraph


class SpecGrammarSample(S2VGraph):
    def __init__(self, sample_index, db, s, pg, node_type_dict):
        super(SpecGrammarSample, self).__init__(pg, node_type_dict)
        self.spectree = s
        self.sample_index = sample_index
        self.db = db        


class Dataset(object):
    def __init__(self):
        self.spec_list = []
        self.grammar_list = []
        self.sample_specs = []
        self.file_names = []
        
        self.setup(SpecGrammarSample)

    def load_spec_list(self, fname):
        with open(cmd_args.data_root + '/CrCi/' + fname, 'r') as f:
            lines = f.read().splitlines()
            res = []
            for l in lines:
                if ";" in l or "(declare-var" in l:
                    continue
                res.append(l)
            content = "\n".join(res)
            sygus_instance = SygusInstance(content)
            self.spec_list.append(SpecTree(sygus_instance))
            self.grammar_list.append( GrammarGraph(sygus_instance) )

    def setup(self, classname):

        if cmd_args.single_sample is None:

            with open(cmd_args.data_root + '/' + cmd_args.file_list, 'r') as f:
                for row in f:
                    self.file_names.append(row.strip())
                    self.load_spec_list(row.strip())
        else:
            self.file_names.append(cmd_args.single_sample)
            self.load_spec_list(cmd_args.single_sample)

        self.build_node_type_dict()

        for i in range(len(self.spec_list)):
            s = self.spec_list[i]
            pg = self.grammar_list[i]
            self.sample_specs.append( classname(i, self, s, pg, self.node_type_dict) )
 
        self.sample_idxes = list(range(len(self.sample_specs)))
        random.shuffle(self.sample_idxes)
        self.sample_pos = 0

    def build_node_type_dict(self):
        self.node_type_dict = {}
        
        for g in self.grammar_list:
            for node in g.node_list:
                if not node.node_type in self.node_type_dict:
                    v = len(self.node_type_dict)
                    self.node_type_dict[node.node_type] = v        

    def sample_minibatch(self, num_samples, replacement=False):        
        # if cmd_args.single_sample is not None:
            # return [self.sample_specs[cmd_args.single_sample]]

        s_list = []
        if replacement:
            for i in range(num_samples):
                idx = np.random.randint(len(self.sample_specs))
                s_list.append(self.sample_specs[idx])
        else:
            assert num_samples <= len(self.sample_idxes)
            if num_samples == len(self.sample_idxes):
                return self.sample_specs

            if self.sample_pos + num_samples > len(self.sample_idxes):
                random.shuffle(self.sample_idxes)
                self.sample_pos = 0

            for i in range(self.sample_pos, self.sample_pos + num_samples):
                s_list.append(self.sample_specs[ self.sample_idxes[i]])
            self.sample_pos == num_samples

        return s_list


if __name__ == '__main__':
    dataset = Dataset()
