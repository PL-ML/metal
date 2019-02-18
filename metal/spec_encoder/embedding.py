from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMEmbed(nn.Module):
    def __init__(self, latent_dim, numOf_node_types):
        """
        this class transform a SpecTree to a numOf_nodes * latent_dim tensor, i.e. the external memory when running RL.
        It maintains a embedding table for each node type. When given a SpecTree, it sees a tree as seq of node_types
        and maps them into embeddings with the embedding table, then it encodes them by applying lstm that processes
        sequentially through the seq.


        latent_dim:
            length of the node embedding
        num_node_types:
            number of nodes types, used to get a embedding table containing initial embedding for each node type
        """
        super(LSTMEmbed, self).__init__()
        self.latent_dim = latent_dim
        self.numOf_node_types = numOf_node_types

        self.w2v = nn.Embedding(numOf_node_types, latent_dim)
        self.lstm = nn.LSTMCell(latent_dim, latent_dim)

    def forward(self, specsample_list, istraining=True):
        """

        specsample_list:
            list of SpecSample objects

        return
        ------
        tree_embed: a n * numOf_nodes * latent_dim tensor, containing embeddings for n SpecTree

        """
        tree_embed_list = []
        if type(specsample_list) is not list:
            specsample_list = [specsample_list]

        for spectree in [specsample.spectree for specsample in specsample_list]:
            # generate embedding for each encountered node type in this tree
            embeddings = self.w2v(Variable(torch.LongTensor(spectree.node_type_seq), requires_grad=False))

            hx = Variable(torch.Tensor(1, self.latent_dim).zero_(), requires_grad=False)
            cx = Variable(torch.Tensor(1, self.latent_dim).zero_(), requires_grad=False)

#            hx = embeddings.new_zeros(1, self.latent_dim, requires_grad=False)
#            cx = embeddings.new_zeros(1, self.latent_dim, requires_grad=False)

            # encode with seq lstm
            node_embeddings = []
            for i in range(spectree.numOf_nodes):
                hx, cx = self.lstm(embeddings[i].view(1, -1), (hx, cx))
                node_embeddings.append(hx)

            # # append placeholders to make same shape
            # assert self.num_node_feats - spectree.numOf_nodes >= 0
            # node_embeddings.append([embeddings.new_zeros(1, self.latent_dim, requires_grad=False) for _ in
            #                         range(self.num_node_feats - spectree.numOf_nodes)])

            node_embeddings.append(hx) # TODO ?
            node_embeddings = torch.cat(node_embeddings, dim=0)

            tree_embed_list.append(node_embeddings)

        return torch.cat(tree_embed_list, dim=0)


from metal.spec_encoder.s2v_lib import S2VLIB, S2VGraph
from metal.common.pytorch_util import weights_init, gnn_spmm, get_torch_version
from metal.common.constants import NUM_GRAMMAR_EDGE_TYPES
import torch.nn.functional as F

class EmbedMeanField(nn.Module):
    def __init__(self, latent_dim, num_node_feats, max_lv = 3):
        super(EmbedMeanField, self).__init__()
        self.latent_dim = latent_dim        
        self.num_node_feats = num_node_feats        

        self.max_lv = max_lv

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)

        self.conv_param_list = []
        self.merge_param_list = []
        for i in range(self.max_lv):
            self.conv_param_list.append(nn.Linear(latent_dim, NUM_GRAMMAR_EDGE_TYPES * latent_dim))
            self.merge_param_list.append( nn.Linear(NUM_GRAMMAR_EDGE_TYPES * latent_dim, latent_dim) )

        self.conv_param_list = nn.ModuleList(self.conv_param_list)
        self.merge_param_list = nn.ModuleList(self.merge_param_list)

        self.state_gru = nn.GRUCell(latent_dim, latent_dim)

        weights_init(self)

    def forward(self, graph_list, istraining=True): 
        node_feat = S2VLIB.ConcatNodeFeats(graph_list)        
        sp_list = S2VLIB.PrepareMeanField(graph_list)
        version = get_torch_version()
        if not istraining:
            if version >= 0.4:
                torch.set_grad_enabled(False)
            else:
                node_feat = Variable(node_feat.data, volatile=True)
        
        h = self.mean_field(node_feat, sp_list)

        if not istraining: # recover
            if version >= 0.4:
                torch.set_grad_enabled(True)

        return h

    def mean_field(self, node_feat, sp_list):
        input_node_linear = self.w_n2l(node_feat)
        input_message = input_node_linear
        input_potential = F.tanh(input_message)

        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            conv_feat = self.conv_param_list[lv](cur_message_layer)
            chunks = torch.split(conv_feat, self.latent_dim, dim=1)
            
            msg_list = []
            for i in range(NUM_GRAMMAR_EDGE_TYPES):
                t = gnn_spmm(sp_list[i], chunks[i])
                msg_list.append( t )
            
            msg = F.tanh( torch.cat(msg_list, dim=1) )
            cur_input = self.merge_param_list[lv](msg)# + input_potential

            cur_message_layer = cur_input + cur_message_layer
            # cur_message_layer = self.state_gru(cur_input, cur_message_layer)
            cur_message_layer = F.tanh(cur_message_layer)
            lv += 1

        return cur_message_layer


if __name__ == '__main__':

    table = nn.Embedding(5, 3)
    emb = table(torch.tensor([1, 2, 3]))
    y = emb.new_zeros(1, 3, requires_grad=False)

    print(table(torch.tensor(range(5))))
    print(emb)
    print(y)

    # random.seed(cmd_args.seed)
    # np.random.seed(cmd_args.seed)
    # torch.manual_seed(cmd_args.seed)
    #
    # s2v_graphs = []
    # pg_graphs = []
    # with open(cmd_args.data_root + '/list.txt', 'r') as f:
    #     for row in f:
    #         with open(cmd_args.data_root + '/' + row.strip() + '.json', 'r') as gf:
    #             graph_json = json.load(gf)
    #             pg_graphs.append(ProgramGraph(graph_json))
    # for g in pg_graphs:
    #     s2v_graphs.append(S2VGraph(g))
    #
    # print(len(s2v_graphs))
    # # mf = EmbedMeanField(128, len(node_type_dict))
    # if cmd_args.ctx == 'gpu':
    #     mf = mf.cuda()
    #
    # embedding = mf(s2v_graphs[0:2])
    # embed2 = mf(s2v_graphs[0:1])
    # embed3 = mf(s2v_graphs[1:2])
    # ee = torch.cat([embed2, embed3], dim=0)
    # diff = torch.sum(torch.abs(embedding - ee))
    # print(diff)
    #
    # r = range(len(s2v_graphs))
    # for i in tqdm(range(1000)):
    #     random.shuffle(r)
    #     glist = []
    #     for j in range(20):
    #         glist.append(s2v_graphs[r[j]])
    #
    #     embedding = mf(glist)
