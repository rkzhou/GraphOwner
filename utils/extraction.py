import sys, os
sys.path.append(os.path.abspath('..'))

import torch
import random
import math
import copy
import numpy as np
import torch.nn.functional as F

from config import parse_args
from utils.datareader import GraphData, DataReader
from torch.utils.data import DataLoader
from model.sagesurrogate import SAGEEMB

from main.benign import node_level_run


class Classification(torch.nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        self.fc1 = torch.nn.Linear(emb_size, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
    

def split_subset(nodes_index, split_ratio):
    random.shuffle(nodes_index)
    subset_num = math.floor(len(nodes_index) * split_ratio)
    extraction_nodes_index = nodes_index[:subset_num]
    
    return extraction_nodes_index


def evaluate_target_response(args, model, dr, eval_nodes_index, response:str):
    assert torch.cuda.is_available(), 'no GPU available'
    cpu = torch.device('cpu')
    cuda = torch.device('cuda')

    model.eval()
    model = model.to(cuda)
    gid = {0}
    gdata = GraphData(dr, gid)
    loader = DataLoader(gdata, batch_size=args.batch_size, shuffle=False)

    for batch_id, data in enumerate(loader):
        for i in range(len(data)):
            data[i] = data[i].to(cuda)
        output, emb = model(data)
        output = output.detach()
        emb = emb.detach()

        if response == 'embeddings':
            target_response = torch.zeros((len(eval_nodes_index), emb.shape[1]))
            for i in range(len(eval_nodes_index)):
                target_response[i] = emb[eval_nodes_index[i]]
        elif response == 'labels':
            target_response = torch.zeros(len(eval_nodes_index), dtype=torch.long)
            predict_fn = lambda output: output.max(1, keepdim=True)[1]
            pred = predict_fn(output)
            for i in range(len(eval_nodes_index)):
                target_response[i] = pred[eval_nodes_index[i]]
        
    return target_response


def train_surrogate_model(args, data):
    dr, surrogate_training_nodes_index, testing_nodes_index, train_emb, train_labels, test_labels = data

    assert torch.cuda.is_available(), 'no GPU available'
    cpu = torch.device('cpu')
    cuda = torch.device('cuda')

    gid = {0}
    gdata = GraphData(dr, gid)
    loader = DataLoader(gdata, batch_size=args.batch_size, shuffle=False)

    #prepare model
    in_dim = loader.dataset.num_features
    out_dim = train_emb.shape[1]
    n_classes = len(np.unique(loader.dataset.node_labels))

    surrogate_model = SAGEEMB(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    surrogate_model = surrogate_model.to(cuda)

    loss_fcn = torch.nn.MSELoss()
    loss_fcn = loss_fcn.to(cuda)
    loss_clf = torch.nn.CrossEntropyLoss()
    loss_clf = loss_clf.to(cuda)

    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr = args.lr)
    clf = Classification(out_dim, n_classes)
    clf = clf.to(cuda)
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    optimizer_classification = torch.optim.SGD(clf.parameters(), lr=0.01)

    epoch_num = 400

    print('Model Extracting')
    for epoch in range(epoch_num):
        surrogate_model.train()

        for batch_id, data in enumerate(loader):
            for i in range(len(data)):
                data[i] = data[i].to(cuda)
            train_emb = train_emb.to(cuda)
            train_labels = train_labels.to(cuda)
            embeddings, _ = surrogate_model(data)
            part_embeddings = torch.zeros((len(surrogate_training_nodes_index), embeddings.shape[1]), device=cuda)

            for i in range(len(surrogate_training_nodes_index)):
                part_embeddings[i] = embeddings[surrogate_training_nodes_index[i]]
            loss = torch.sqrt(loss_fcn(part_embeddings, train_emb))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_classification.zero_grad()
            logists = clf(part_embeddings.detach())
            loss_sup = loss_clf(logists, train_labels)
            loss_sup.backward()
            optimizer_classification.step()
        
        if (epoch + 1) % 10 == 0:
            surrogate_model.eval()
            
            acc_correct = 0
            fide_correct = 0
            for batch_id, data in enumerate(loader):
                for i in range(len(data)):
                    data[i] = data[i].to(cuda)
                embeddings, _ = surrogate_model(data)
                outputs = clf(embeddings.detach())
                pred = predict_fn(outputs)
                
                for i in range(len(testing_nodes_index)):
                    if pred[testing_nodes_index[i]] == data[2][0][testing_nodes_index[i]]:
                        acc_correct += 1
                    if pred[testing_nodes_index[i]] == test_labels[i]:
                        fide_correct += 1
                
            accuracy = acc_correct * 100.0 / len(testing_nodes_index)
            fidelity = fide_correct * 100.0 / len(test_labels)
            print('Accuracy of model extraction is {:.4f} and fidelity is {:.4f}'.format(accuracy, fidelity))
    

    return surrogate_model





if __name__ == '__main__':
    # args = parse_args()
    # dr, model, training_nodes, testing_nodes = node_level_run(args)
    # surrogate_train_subset = split_subset(training_nodes, 0.5)
    # target_emb = evaluate_target_response(args, model, dr, surrogate_train_subset, 'embeddings')
    # target_labels = evaluate_target_response(args, model, dr, testing_nodes, 'labels')
    # data = dr, surrogate_train_subset, testing_nodes, target_emb, target_labels
    # train_surrogate_model(args, data)
    pass