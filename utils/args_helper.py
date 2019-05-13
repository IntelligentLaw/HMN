import argparse


def common_args(model_name):
    parser = argparse.ArgumentParser(description='model_args')

    # model name
    parser.add_argument('-model_name', type=str, default=model_name, help='where to save the snapshot')

    # learning
    parser.add_argument('-lr_dynamic', type=bool, default=True, help="if make the learning rate dynamic")
    parser.add_argument('-lr', type=float, default=5e-3, help='initial learning rate [default: 0.001]')
    parser.add_argument('-lr2', type=float, default=1e-3, help='the embedding layer learning rate ')
    parser.add_argument('-min_lr', type=float, default=1e-5, help='the min learning rate of training, when ')
    parser.add_argument('-lr_decay', type=float, default=0.95, help='the decay of learning rate')
    parser.add_argument('-decay_every', type=float, default= 1500, help='decay calculate every 3000 steps')
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    parser.add_argument('-early-stop', type=int, default=4000, help='iteration numbers to stop without performance increasing')
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')

    # common model
    parser.add_argument('-kmax-pooling', type=int, default=2, help='k max pooling size [default: 2]')
    parser.add_argument('-linear_hidden_size', type=int, default=512, help='fc linear hidden size [default: 400]')
    parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
    parser.add_argument("-static", type=bool, default=False, help = "fix the embedding [default: False]")
    parser.add_argument("-add_pretrain_embedding",  type=bool, default=False)
    parser.add_argument('-pretrain_embedding_path', type=str, default="./new_data/pretrainEmbeddingMatrix.npz", help="the pretrain embedding path")
    parser.add_argument('-hidden_dim', type=int, default=300, help='hidden size [default: 300]')


    # CNN model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

    # LSTM model
    parser.add_argument('-lstm_hidden_dim', type=int, default=128, help='lstm hidden dim [default: 128]')
    parser.add_argument('-lstm_num_layers', type=int, default=1, help='the num of layers [default: 1]')
    parser.add_argument('-bidirectional', type=bool, default=True, help='the bidirectional')

    # Word Attention + Sentence Attention
    parser.add_argument('-word_gru_hidden', type=int, default=128, help='the num of layers [default: 1]')
    parser.add_argument('-sent_gru_hidden', type=int, default=128, help='the num of layers [default: 1]')

    # device
    parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    args = parser.parse_args()

    return args


