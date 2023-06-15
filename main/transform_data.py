import torch_geometric.datasets as dt
import torch

if __name__ == '__main__':
    #test_dataset = dt.FacebookPagePage('../dataset/Facebook/')
    data = torch.load('../dataset/Facebook/processed/data.pt')
    print(data)

    f = open('../dataset/Facebook/Facebook_A.txt', 'w+')
    for i in range(data[0]['edge_index'].shape[1]):
        source_node = data[0]['edge_index'][0][i]
        target_node = data[0]['edge_index'][1][i]
        f.write('{}, {}\n'.format(source_node, target_node))
    f.close()

    f = open('../dataset/Facebook/Facebook_graph_indicator.txt', 'w+')
    for i in range(data[0]['x'].shape[0]):
        f.write('1\n')
    f.close()

    f = open('../dataset/Facebook/Facebook_node_attributes.txt', 'w+')
    for i in range(data[0]['x'].shape[0]):
        for j in range(data[0]['x'].shape[1]):
            f.write(str(data[0]['x'][i][j].item()))
            if j != (data[0]['x'].shape[1] - 1):
                f.write(', ')
            else:
                f.write('\n')
    f.close()
    
    f = open('../dataset/Facebook/Facebook_node_labels.txt', 'w+')
    for i in range(data[0]['y'].shape[0]):
        f.write(str(data[0]['y'][i].item()))
        f.write('\n')
    f.close()