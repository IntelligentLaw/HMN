import argparse

import datetime
from utils.datasets import *
import torch
from model.model import HMN
import train

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('--dev_batch_size', default=1, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=3000, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--hidden-size', default=128, type=int)
    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--learning-rate', default=0.0015, type=float)
    parser.add_argument('--print-freq', default=1500, type=int)
    parser.add_argument('--test-freq', default=1, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)

    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    parser.add_argument('-snapshot', type=str, default=None,
                        help='filename of model snapshot [default: None]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('-embed_num', type=int, default=100000, help='the num of vocabulary size')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    args = parser.parse_args()


    time_str = datetime.datetime.now().isoformat()
    sys.stdout = Logger("./output/{}.txt".format(time_str))

    print("\nLoading data...")
    accu_path = "/home/iot/fanyu/Attention-BiDAF/new_data_with_parent/accu_dict.pkl"
    law_path = "/home/iot/fanyu/Attention-BiDAF/new_data_with_parent/law_dict.pkl"
    word_path = "/home/iot/fanyu/Attention-BiDAF/new_data_with_parent/word_dict_10w.pkl"
    parent_path = "/home/iot/fanyu/Attention-BiDAF/new_data_with_parent/parent_dict.pkl"
    train_data_path = "/home/iot/fanyu/BiLabelEmbedding/new_data_with_parent/preprocess_good/transform_data_train_dir/"
    dev_data_path = "/home/iot/fanyu/BiLabelEmbedding/new_data_with_parent/preprocess_good/transform_data_test_dir/"


    train_iter, dev_iter, word_num, law_num, parent_num = make_data(train_data_path, dev_data_path,
                                                          law_path, parent_path, word_path, args.batch_size, args.dev_batch_size)
    args.model_name = 'TEST'
    args.save_dir = "accu_snapshot"
    args.embed_num = word_num
    args.class_num = law_num
    args.parent_num = parent_num
    args.law_num = law_num
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, args.model_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


    model = HMN(args)

    if args.snapshot is not None:
        print("\nLoading model from {}...".format(args.snapshot))
        model.load_state_dict(torch.load(args.snapshot))
    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    if args.cuda:
        print("cuda")
        torch.cuda.set_device(args.gpu)
        model = model.cuda()

    print('training start!')
    train.train(train_iter, dev_iter, model, args)
    print('training finished!')


if __name__ == '__main__':
    main()
