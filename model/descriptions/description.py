import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.DynamicGRU import DynamicGRU

class LawDscription(nn.Module):
    def __init__(self, args):
        super(LawDscription, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        self.parent_num = [17,54,20,13,55,3,14,7]
        self.lstm_hidden_dim = args.hidden_size
        self.embed = nn.Embedding(V, D, padding_idx=0)
        self.lstmLabel = nn.LSTM(D, args.hidden_size, bidirectional=True, batch_first=True)

        self.label_dynamic_gru = DynamicGRU(input_dim=D,
                                         output_dim=self.lstm_hidden_dim,
                                         bidirectional=True,
                                         batch_first=True)


    def cal(self,input_list, last_hidden):
        """
        use CosineSimilarity to calculate the reputation
        :param input_list
        :param last_hidden
        :return:father
        :return:output_list(one father class's subclasses)
        """
        #B is the num of subclasses
        #[B,H]--[B*B,L,H]
        b = (last_hidden.unsqueeze(1)).repeat(len(last_hidden), input_list.size()[1], 1)
        #[B,L,H]--[B*B,L,H]
        a = [(law).unsqueeze(0).repeat(len(input_list), 1, 1) for law in input_list]
        a = torch.cat(a)

        # #[B*B,L,L]
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        #[B*B,L,1]
        cosine = cos(a,b).unsqueeze(2)
        # [B*B,L,H]*[B*B,L,1]
        attention = a*cosine
        # [B, B, L, H]
        part_list = attention.view(len(input_list), len(input_list), -1, self.args.hidden_size)
        # [B,B,L,H]----[B,1,L,H]-----[B,L,H]  all subclasses's attention
        avgpool = F.avg_pool3d(part_list, (len(input_list), 1, 1)).squeeze()

        # weight = torch.div(avgpool,input_list)

        output_list = input_list-avgpool
        father = torch.cat([avgpool])

        return father, output_list
    def forward(self, law_text, law_length):
        """
        calculate the expression,and compress the expression into sequence level(parent class and subclass)
        :param law_text
        :param law_length
        :return:father_list
        :return:all_list
        """
        x = self.embed(law_text)
        label_lstm_out, (label_ht, label_ct) = self.label_dynamic_gru(x, law_length)
        output = label_lstm_out[:, :, :self.lstm_hidden_dim] + label_lstm_out[:, :, self.lstm_hidden_dim:]

        last_hideen = []
        for i,length in enumerate(law_length):
            last_hideen.append(output[i,length-1,:])
        last_hideen = torch.stack(last_hideen, dim=0)

        i=0
        all_list = []
        father_list = []
        for j,size in enumerate(self.parent_num):
            hidden = last_hideen[i:i + size]
            input_list = output[i:i+size]
            i = i + size
            father, part_list = self.cal(input_list, hidden)
            part_list = part_list.transpose(1, 2)
            label_part_out = F.max_pool1d(part_list, part_list.size(2)).squeeze(2)
            all_list.append(label_part_out)
            father_list.append(father)

        father_list = torch.cat(father_list)
        label_out_transpose = father_list.transpose(1, 2)
        # (LS, H, L) -> (LS, H)
        father_list = F.max_pool1d(label_out_transpose, label_out_transpose.size(2)).squeeze(2)
        return father_list, all_list