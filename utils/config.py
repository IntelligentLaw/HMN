#coding=utf8
import argparse

def common_args(model_name):
    parser = argparse.ArgumentParser(description='classifier')
    # model name
    parser.add_argument('-model-name', type=str, default = model_name, help = 'the model name')

    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=10000, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=256, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=50,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=300, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    # parser.add_argument('-save-dir', type=str, default='accu_snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # data
    # parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
     # device
    parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')

    # common model args
    parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-linear_hidden_size', type=int, default=128, help='number of fc linear hidden size [default: 128]')
    parser.add_argument('-seq_len', type=int, default=1500, help="the len of sequence length")
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    parser.add_argument('-embed_num', type=int, default=150000, help='the num of vocabulary size')

    # CNN model args
    parser.add_argument('-kernel-num', type=int, default=128, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='1,2,3,4',
                        help='comma-separated kernel size to use for convolution')
    # LSTM model args
    parser.add_argument('-hidden_size', type=int, default=128, help='number of fc linear hidden size [default: 128]')
    parser.add_argument('-num_layers', type=int, default=2, help='number of fc layers [default: 2]')

    # RCNN model args

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print(1)
    args = common_args("common_args")
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
