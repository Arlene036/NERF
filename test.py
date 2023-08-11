from model import MoleculeVAE
import torch
import argparse
from main import load_data
from dataset import TransformerDataset
import pickle
from torch.utils.data import DataLoader

def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data_path', type=str, help='path of dataset', default='./')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size.256')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the order of atoms')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers to generate data.')
    parser.add_argument('--prefix', type=str, default='data',
                        help='data prefix')

    parser.add_argument('--name', type=str, default='tmp',
                        help='model name, crucial for test and checkpoint initialization')
    parser.add_argument('--vae', action='store_true', default=False, help='use vae')
    parser.add_argument('--depth', type=int, default=6, help='depth')
    parser.add_argument('--dim', type=int, default=192, help='dim')

    parser.add_argument('--save_path', type=str, default='./CKPT/', help='path of save prefix')
    parser.add_argument('--train', action='store_true', default=False, help='do training.')
    parser.add_argument('--save', action='store_true', default=True, help='Save model.')
    parser.add_argument('--eval', action='store_true', default=False, help='eval model.')
    parser.add_argument('--test', action='store_true', default=True, help='test model.')
    parser.add_argument('--recon', action='store_true', default=False, help='test reconstruction only.')

    parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train. 200')
    parser.add_argument('--local_rank', type=int, default=0, help='rank')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--temperature', type=float, default=1, nargs='+', help='temperature.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--beta', type=float, default=0.1, help='the weight of kl')
    parser.add_argument('--checkpoint', type=str, default=None, nargs='*',
                        help='initialize from a checkpoint, if None, do not restore')
    parser.add_argument('--world_size', type=int, default=1, help='number of processes')

    args = parser.parse_args()



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    m = MoleculeVAE(args, 100).to(device)


    file_path = 'H:/NERF/data/USPTO-15K/valid_1_data.pickle'
    loaded_data = read_pickle_file(file_path)
    mini_dataset = TransformerDataset(args.shuffle, loaded_data)
    mini_dataloader = DataLoader(mini_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn = TransformerDataset.collate_fn)


    for data in mini_dataloader:
        for key in data:
            data[key] = data[key].to(device)
        output = m('sample', data, temperature=0)
        # print(output)
        break



