import sys, os
sys.path.append(os.path.abspath('..'))

import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import utils.extraction
import networkx as nx

from utils.datareader import DataReader, GraphData
from utils.bkdcdd import select_cdd_graphs, select_cdd_nodes
from utils.mask import gen_mask, recover_mask
import main.benign as benign
import trojan.GTA as gta
from trojan.input import gen_input
from trojan.prop import train_model, evaluate
from config import parse_args
from main.benign import node_level_run

class GraphBackdoor:
    def __init__(self, args) -> None:
        self.args = args
        
        assert torch.cuda.is_available(), 'no GPU available'
        self.cpu = torch.device('cpu')
        self.cuda = torch.device('cuda')

    def run(self):
        # train a benign GNN
        self.benign_dr, self.benign_model, training_nodes_index, testing_nodes_index = benign.node_level_run(self.args)
        model = copy.deepcopy(self.benign_model).to(self.cuda)
        # pick up initial candidates
        bkd_gids_test, bkd_nids_test, bkd_nid_groups_test = self.bkd_cdd(training_nodes_index, testing_nodes_index, 'test')
        test_transform_nodes = set()
        for gid in bkd_gids_test:
            for node_group in bkd_nid_groups_test[gid]:
                test_transform_nodes = test_transform_nodes.union(set(node_group))
                nei = get_neighbors(node_group, self.benign_dr.data['adj_list'][gid], 2)
                nei = set(nei)
                test_transform_nodes = test_transform_nodes.union(nei)
        test_transform_nodes = set(testing_nodes_index).intersection(test_transform_nodes)
        test_unchanged_nodes = set(testing_nodes_index).difference(test_transform_nodes)
        
        test_nodes_in_target_class = set()
        test_nodes_notin_target_class = set()
        for node_index in test_transform_nodes:
            if self.benign_dr.data['nlabels'][0][node_index] == args.target_class:
                test_nodes_in_target_class.add(node_index)
            else:
                test_nodes_notin_target_class.add(node_index)
        
        nodenums = [adj.shape[0] for adj in self.benign_dr.data['adj_list']]
        nodemax = max(nodenums)
        featdim = np.array(self.benign_dr.data['features'][0]).shape[1]

        # init two generators for topo/feat
        toponet = gta.GraphTrojanNet(nodemax, self.args.gtn_layernum)
        featnet = gta.GraphTrojanNet(featdim, self.args.gtn_layernum)

        
        # init test data
        # NOTE: for data that can only add perturbation on features, only init the topo value
        init_dr_test = self.init_trigger(
            self.args, copy.deepcopy(self.benign_dr), bkd_gids_test, bkd_nid_groups_test, 0.0, 0.0)
        bkd_dr_test = copy.deepcopy(init_dr_test)
        
        topomask_test, featmask_test = gen_mask(
            init_dr_test, bkd_gids_test, bkd_nid_groups_test)
        Ainput_test, Xinput_test = gen_input(self.args, init_dr_test, bkd_gids_test)
        

        for rs_step in range(self.args.resample_steps):   # for each step, choose different sample
            
            # randomly select new graph backdoor samples
            bkd_gids_train, bkd_nids_train, bkd_nid_groups_train = self.bkd_cdd(training_nodes_index, testing_nodes_index,'train')

            train_transform_nodes = set()
            for gid in bkd_gids_train:
                for node_group in bkd_nid_groups_train[gid]:
                    train_transform_nodes = train_transform_nodes.union(set(node_group))
                    nei = get_neighbors(node_group, self.benign_dr.data['adj_list'][gid], 2)
                    nei = set(nei)
                    train_transform_nodes = train_transform_nodes.union(nei)
            train_transform_nodes = set(training_nodes_index).intersection(train_transform_nodes)
            train_unchanged_nodes = set(training_nodes_index).difference(train_transform_nodes)
            
            # positive/negtive sample set
            pset = bkd_gids_train
            nset = list([0])

            # if self.args.pn_rate != None:
            #     if len(pset) > len(nset):
            #         repeat = int(np.ceil(len(pset)/(len(nset)*self.args.pn_rate)))
            #         nset = list(nset) * repeat
            #     else:
            #         repeat = int(np.ceil((len(nset)*self.args.pn_rate)/len(pset)))
            #         pset = list(pset) * repeat
            
            # init train data
            # NOTE: for data that can only add perturbation on features, only init the topo value
            init_dr_train = self.init_trigger(
                self.args, copy.deepcopy(self.benign_dr), bkd_gids_train, bkd_nid_groups_train, 0.0, 0.0)
            for node_index in train_transform_nodes:
                (init_dr_train.data['nlabels'])[0][node_index] = args.target_class
            bkd_dr_train = copy.deepcopy(init_dr_train)

            ####evaluate edge centrality
            total_train_EC, total_test_EC = 0.0, 0.0
            total_train_NC, total_test_NC = 0.0, 0.0
            gdata = GraphData(self.benign_dr, {0})
            A = nx.from_numpy_matrix(gdata.adj_list[0])
            degree_cen = nx.degree_centrality(A)
            node_cen = nx.eigenvector_centrality(A)
            for sub_group in bkd_nid_groups_train[0]:
                for i in sub_group:
                    total_train_EC += degree_cen[i]
                    total_train_NC += node_cen[i]
            
            for sub_group in bkd_nid_groups_test[0]:
                for i in sub_group:
                    total_test_EC += degree_cen[i]
                    total_test_NC += node_cen[i]
            print(total_train_EC, total_test_EC)
            print(total_train_NC, total_test_NC)

            

            topomask_train, featmask_train = gen_mask(
                init_dr_train, bkd_gids_train, bkd_nid_groups_train)
            Ainput_train, Xinput_train = gen_input(self.args, init_dr_train, bkd_gids_train)
            
            for bi_step in range(self.args.bilevel_steps):
                print("Resampling step %d, bi-level optimization step %d" % (rs_step, bi_step))
                
                toponet, featnet = gta.train_gtn(
                    self.args, model, toponet, featnet,
                    pset, nset, topomask_train, featmask_train, 
                    init_dr_train, bkd_dr_train, Ainput_train, Xinput_train)
                
                # get new backdoor datareader for training based on well-trained generators
                for gid in bkd_gids_train:
                    rst_bkdA = toponet(
                        Ainput_train[gid], topomask_train[gid], self.args.topo_thrd, 
                        self.cpu, self.args.topo_activation, 'topo')
                    # rst_bkdA = recover_mask(nodenums[gid], topomask_train[gid], 'topo')
                    # bkd_dr_train.data['adj_list'][gid] = torch.add(rst_bkdA, init_dr_train.data['adj_list'][gid])
                    bkd_dr_train.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]].detach().cpu(), 
                        init_dr_train.data['adj_list'][gid])
                
                    rst_bkdX = featnet(
                        Xinput_train[gid], featmask_train[gid], self.args.feat_thrd, 
                        self.cpu, self.args.feat_activation, 'feat')
                    # rst_bkdX = recover_mask(nodenums[gid], featmask_train[gid], 'feat')
                    # bkd_dr_train.data['features'][gid] = torch.add(rst_bkdX, init_dr_train.data['features'][gid]) 
                    bkd_dr_train.data['features'][gid] = torch.add(
                        rst_bkdX[:nodenums[gid]].detach().cpu(), init_dr_train.data['features'][gid]) 
                    
                # train GNN
                train_model(self.args, bkd_dr_train, model, train_transform_nodes, train_unchanged_nodes)
                
                #----------------- Evaluation -----------------#
                for gid in bkd_gids_test:
                    rst_bkdA = toponet(
                        Ainput_test[gid], topomask_test[gid], self.args.topo_thrd, 
                        self.cpu, self.args.topo_activation, 'topo')
                    # rst_bkdA = recover_mask(nodenums[gid], topomask_test[gid], 'topo')
                    # bkd_dr_test.data['adj_list'][gid] = torch.add(rst_bkdA, 
                    #     torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))
                    bkd_dr_test.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]], 
                        torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))
                
                    rst_bkdX = featnet(
                        Xinput_test[gid], featmask_test[gid], self.args.feat_thrd, 
                        self.cpu, self.args.feat_activation, 'feat')
                    # rst_bkdX = recover_mask(nodenums[gid], featmask_test[gid], 'feat')
                    # bkd_dr_test.data['features'][gid] = torch.add(
                    #     rst_bkdX, torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))
                    bkd_dr_test.data['features'][gid] = torch.add(
                        rst_bkdX[:nodenums[gid]], torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))
                    
                # # graph originally in target label
                # yt_gids = [gid for gid in bkd_gids_test 
                #         if self.benign_dr.data['labels'][gid]==self.args.target_class] 
                # # graph originally notin target label
                # yx_gids = list(set(bkd_gids_test) - set(yt_gids))
                # clean_graphs_test = list(set(self.benign_dr.data['splits']['test'])-set(bkd_gids_test))

                # feed into GNN, test success rate
                bkd_acc = evaluate(self.args, bkd_dr_test, model, test_transform_nodes, 'success')
                flip_rate = evaluate(self.args, bkd_dr_test, model,test_nodes_notin_target_class, 'flip')
                clean_acc = evaluate(self.args, bkd_dr_test, model, test_unchanged_nodes, 'clean')
                
                # save gnn
                if rs_step == 0 and (bi_step==self.args.bilevel_steps-1 or abs(bkd_acc-100) <1e-4):
                    if self.args.save_bkd_model:
                        save_path = self.args.bkd_model_save_path
                        os.makedirs(save_path, exist_ok=True)
                        save_path = os.path.join(save_path, '%s-%s-%f.t7' % (
                            self.args.model, self.args.dataset, self.args.train_ratio, 
                            self.args.bkd_gratio_trainset, self.args.bkd_num_pergraph, self.args.bkd_size))
                    
                        torch.save({'model': model.state_dict(),
                                    'asr': bkd_acc,
                                    'flip_rate': flip_rate,
                                    'clean_acc': clean_acc,
                                }, save_path)
                        print("Trojaning model is saved at: ", save_path)
                    
                if abs(bkd_acc-100) <1e-4:
                    # bkd_dr_tosave = copy.deepcopy(bkd_dr_test)
                    print("Early Termination for 100% Attack Rate")
                    break
        print('Done')
        return training_nodes_index, testing_nodes_index, self.benign_dr, bkd_dr_test, model, test_transform_nodes, test_nodes_notin_target_class, test_unchanged_nodes


    def bkd_cdd(self, training_nodes_index, testing_nodes_index, subset: str):
        # - subset: 'train', 'test'
        # find graphs to add trigger (not modify now)
        bkd_gids = [0]
        # find trigger nodes per graph
        # same sequence with selected backdoored graphs
        bkd_nids, bkd_nid_groups = select_cdd_nodes(
            self.args, bkd_gids, self.benign_dr.data['adj_list'], training_nodes_index, testing_nodes_index, subset)

        assert len(bkd_gids)==len(bkd_nids)==len(bkd_nid_groups)

        return bkd_gids, bkd_nids, bkd_nid_groups


    @staticmethod
    def init_trigger(args, dr: DataReader, bkd_gids: list, bkd_nid_groups: list, init_edge: float, init_feat: float):
        if init_feat == None:
            init_feat = - 1
            print('init feat == None, transferred into -1')
        
        # (in place) datareader trigger injection
        for i in tqdm(range(len(bkd_gids)), desc="initializing trigger..."):
            gid = bkd_gids[i]           
            for group in bkd_nid_groups[i] :
                # change adj in-place
                src, dst = [], []
                for v1 in group:
                    for v2 in group:
                        if v1!=v2:
                            src.append(v1)
                            dst.append(v2)
                a = np.array(dr.data['adj_list'][gid])
                a[src, dst] = init_edge
                dr.data['adj_list'][gid] = a.tolist()
                featdim = len(dr.data['features'][0][0])
                a = np.array(dr.data['features'][gid])
                a[group] = np.ones((len(group), featdim)) * init_feat
                dr.data['features'][gid] = a.tolist()
                #for j in range(len(src)):
                #    dr.data['adj_list'][gid][src[j], dst[j]] = init_edge

                # change features in-place
                #featdim = len(dr.data['features'][0][0])
                #dr.data['features'][gid][group] = np.ones((len(group), featdim)) * init_feat
            
                # change graph/node labels
                assert args.target_class is not None
                for v in group:
                    dr.data['nlabels'][gid][v] = args.target_class
                 #dr.data['labels'][gid] = args.target_class

        return dr 


def get_neighbors(nodes, np_adj, khops):
    def find_neighbors(nodes, adj, j, result):
        _nodes = set()
        if j == 0:
            return
        all_node_num = adj.shape[0]
        for v in nodes:
            for u in range(all_node_num):
                if adj[v, u] == 1:
                    result.add(u)
                    _nodes.add(u)
        find_neighbors(_nodes, adj, j-1, result)
    
    result = set()
    find_neighbors(nodes, np_adj, khops, result)
    neighbors = np.array(list(result))
    #neighbors = neighbors[np.where(~np.in1d(neighbors, np.array(nodes)))]
    return neighbors


def attack_extraction_model(args):
    attack = GraphBackdoor(args)
    training_nodes, testing_nodes, benign_dr, bkd_dr_test, model, test_transform_nodes, test_nodes_notin_target_class, test_unchanged_nodes = attack.run()

    surrogate_train_subset = utils.extraction.split_subset(training_nodes, 0.5)
    train_emb = utils.extraction.evaluate_target_response(args, model, benign_dr, surrogate_train_subset, 'embeddings')
    train_labels = utils.extraction.evaluate_target_response(args, model, benign_dr, surrogate_train_subset, 'labels')
    test_labels = utils.extraction.evaluate_target_response(args, model, benign_dr, testing_nodes, 'labels')
    data = benign_dr, surrogate_train_subset, testing_nodes, train_emb, train_labels, test_labels
    surrogate_model = utils.extraction.train_surrogate_model(args, data)


    target_bkd_acc = evaluate(args, bkd_dr_test, model, test_transform_nodes, 'success')
    target_flip_rate = evaluate(args, bkd_dr_test, model,test_nodes_notin_target_class, 'flip')
    target_clean_acc = evaluate(args, bkd_dr_test, model, test_unchanged_nodes, 'clean')

    surrogate_bkd_acc = evaluate(args, bkd_dr_test, surrogate_model, test_transform_nodes, 'success')
    surrogate_flip_rate = evaluate(args, bkd_dr_test, surrogate_model,test_nodes_notin_target_class, 'flip')
    surrogate_clean_acc = evaluate(args, bkd_dr_test, surrogate_model, test_unchanged_nodes, 'clean')

    
if __name__ == '__main__':
    args = parse_args()
    attack_extraction_model(args)
    #attack = GraphBackdoor(args)
    #attack.run()