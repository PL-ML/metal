
from metal.common.constants import OR_DERIVE_TYPE, AND_DERIVE_TYPE, XOR_DERIVE_TYPE, NOT_DERIVE_TYPE, T_DERIVE_TYPE, NT_DERIVE_TYPE
from metal.common.constants import OP_NAMES, OP_NAME2TYPE, OP_GLOBAL_N2T, TYPE2NAME, SPEC_COVER_TYPE


class GraphNode(object):
    def __init__(self, index, node_type, name = None):
        self.index = index
        self.node_type = node_type
        
        self.name = name

        self.in_edge_list = []
        self.out_edge_list = []

    def add_in_edge(self, src, edge_type):
        self.in_edge_list.append((edge_type, src))

    def add_out_edge(self, dst, edge_type):
        self.out_edge_list.append((edge_type, dst))

class ExprNode(object):
    def __init__(self, pg_node):
        self.pg_node = None
        if type(pg_node) is not GraphNode:
            self.name = pg_node
        else:
            self.pg_node = pg_node
            self.name = pg_node.name
        self.children = []
        self.state = None

    def clone(self):
        if self.pg_node is None:
            root = ExprNode(self.name)
        else:
            root = ExprNode(self.pg_node)
        
        for c in self.children:
            root.children.append(c.clone())
        
        return root

    def __str__(self):
        pass

    def to_smt2(self):
        return self.name 

    def has_internal_implications(self,pg):
        return False 



    def has_trivial_pattern(self):
        return False

    def to_z3(self):
        pass

    def get_vars(self, st):
        pass

    def to_py(self):
        pass


class GrammarGraph(object):
    def __init__(self, sygus_instance):        
        self.node_list = []
        self.unique_nodes = {}
        self.edge_list = []


        self.nt_nodes = []
        self.t_nodes = []
        self.op_nodes = [] # operators are similar to raw/ssa variables
        self.global_or = self.add_node("global", "or")
        self.global_and = self.add_node("global", "and")
        self.global_xor = self.add_node("global", "xor")
        self.global_not = self.add_node("global", "not")
        self.global_opd = {"or" : self.global_or, "and":self.global_and, "xor":self.global_xor, "not" : self.global_not}

        self.cfg_mapping = { 
            "global_or" : self.global_or.index,
            "global_and" : self.global_and.index,
            "global_xor" : self.global_xor.index,
            "global_not" : self.global_not.index,
         }
        
        cfg = sygus_instance.get_grammar()
        self.traverse_grammar_ast(cfg)

        self.spec_embedding = self.add_node("spec_embedding", "spec")
        self.spec_nodes = []
        spec = sygus_instance.get_spec()
        self.traveser_spec_ast(spec)


    def traverse_helper(self, cfg, nt, vis):
        if nt in vis:
            return vis[nt]

        # check nt is terminal or not
        if nt in cfg.terminals:
            node = self.add_node("Terminal", nt)
            self.t_nodes.append(node)
            vis[nt] = node
            self.cfg_mapping[ nt ] = [ node.index ]
            return node
        
        node = self.add_node("nonTerminal", nt)
        self.nt_nodes.append(node)
        self.cfg_mapping[ nt ] = [ node.index ]

        rules = cfg.productions[nt]
        for r in rules:
            app = r[0]
            if app not in OP_NAMES:
                b_node = self.traverse_helper(cfg, app, vis)
                self.cfg_mapping[ nt ].append( b_node.index )
                self.add_double_dir_edge(node.index, b_node.index, T_DERIVE_TYPE, T_DERIVE_TYPE+1)
                continue
            
            app_node = self.add_node(app, "%s_%s" % (nt,app))
            self.op_nodes.append(app_node)
            self.cfg_mapping[ nt ].append( app_node.index )
            self.add_double_dir_edge(node.index, app_node.index, NT_DERIVE_TYPE, NT_DERIVE_TYPE + 1)

            # add edges between specific application of OP and its global symbol
            g_op = self.global_opd[app]
            ty = OP_GLOBAL_N2T[app]
            self.add_double_dir_edge(g_op.index, app_node.index, ty, ty+1)

            ty = OP_NAME2TYPE[app]
            for x in r[1:]:
                b_node = self.traverse_helper(cfg, x, vis)
                self.add_double_dir_edge(app_node.index, b_node.index, ty, ty + 1)
        
        vis[nt] = node
        return node 


    def traverse_grammar_ast(self, cfg):
        visited = {}
        self.traverse_helper(cfg, cfg.start, visited)


    def find_node_by_type_name(self, ty, name):
        for nd in self.node_list:
            if nd.node_type == ty and nd.name == name:
                return nd
        return None

    def traveser_spec_ast(self, spec):
        if len(spec.args) == 0:
            #current node is terminal, which should be found in grammar ast
            nd = self.find_node_by_type_name("Terminal", spec.app)
            assert nd, "cannot find node for " + spec.app
            return nd
        else:
            app = spec.app 
            app_node = self.add_node(app, None)#"spec_" + app)
            self.spec_nodes.append(app_node)

            # spec cover edge
            self.add_double_dir_edge(self.spec_embedding.index, app_node.index, SPEC_COVER_TYPE, SPEC_COVER_TYPE+1)

            # add edges between specific application of OP and its global symbol
            g_op = self.global_opd[app]
            ty = OP_GLOBAL_N2T[app]
            self.add_double_dir_edge(g_op.index, app_node.index, ty, ty+1)

            ty = OP_NAME2TYPE[app]
            for x in spec.args:
                b_node = self.traveser_spec_ast(x)
                self.add_double_dir_edge(app_node.index, b_node.index, ty, ty + 1)

            return app_node

    def num_nodes(self):
        return len(self.node_list)

    def num_edges(self):
        return len(self.edge_list)

    def add_node(self, node_type, name = None):                
        idx = len(self.node_list)
        node = GraphNode(idx, node_type=node_type, name=name)
        self.node_list.append(node)
        if name is not None:
            key = node_type + '-' + name
            # print("add key:", key)
            assert key not in self.unique_nodes, "key=" + key + " unique_nodes:" + str(self.unique_nodes)
            self.unique_nodes[key] = node
        return node

    def add_directed_edge(self, src_idx, dst_idx, edge_type):
        x = self.node_list[src_idx]
        y = self.node_list[dst_idx]
        x.add_out_edge(y, edge_type)
        y.add_in_edge(x, edge_type)
        self.edge_list.append((src_idx, dst_idx, edge_type))

    def add_double_dir_edge(self, src_idx, dst_idx, edge_type_forward, edge_type_backward):
        self.add_directed_edge(src_idx, dst_idx, edge_type_forward)
        self.add_directed_edge(dst_idx, src_idx, edge_type_backward)

    def dump_dot(self):

        print("node:", self.node_list)
        print("cfg_mapping:", self.cfg_mapping)

        print("digraph G{")
        for nd in self.node_list:
            print('\t%s[label="%s"];' % (nd.index, nd.name) )
        
        for e in self.edge_list:
            src, dst, ty = e
            print('\t%d -> %d [label="%s"];' % (src, dst, TYPE2NAME[ty]))

        print("}")
        

if __name__ == '__main__':
    pass
