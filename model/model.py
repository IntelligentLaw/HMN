from model.attentions.RSABlock import *
from model.layers.FCLayer import *
from model.descriptions.description import LawDscription
from model.model_utils import LayerNorm
from model.layers.DynamicGRU import DynamicGRU

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class HMN(nn.Module):
    def __init__(self, args):
        super(HMN, self).__init__()
        self.args = args
        C = args.class_num
        self.lstm_hidden_dim = args.hidden_size
        self.decription = LawDscription(args)
        self.RSANModel_Sub = RSANModel_Sub(args)
        self.RSANModel = RSANModel(args)
        self.coatt1 = RSANModel(args)
        self.coatt2 = RSANModel(args)
        self.coatt3 = RSANModel(args)
        self.coatt4 = RSANModel(args)
        self.coatt5 = RSANModel(args)
        self.coatt6 = RSANModel(args)
        self.coatt7 = RSANModel(args)
        self.coatt8 = RSANModel(args)
        self.final_fc = FCLayer(self.lstm_hidden_dim*2 , 8, type="deep")
        self.fc1 = FCLayer(self.lstm_hidden_dim*2 , 17, type="deep")
        self.fc2 = FCLayer(self.lstm_hidden_dim*2 , 54, type="deep")
        self.fc3 = FCLayer(self.lstm_hidden_dim*2 , 20, type="deep")
        self.fc4 = FCLayer(self.lstm_hidden_dim*2 , 13, type="deep")
        self.fc5 = FCLayer(self.lstm_hidden_dim*2 , 55, type="deep")
        self.fc6 = FCLayer(self.lstm_hidden_dim*2 , 3, type="deep")
        self.fc7 = FCLayer(self.lstm_hidden_dim*2 , 14, type="deep")
        self.fc8 = FCLayer(self.lstm_hidden_dim*2 , 7, type="deep")

    def forward(self, inputs=None, inputs_length=None, label_inputs=None, all_list=None,
                label_inputs_length=None, label_des=None, classify=None,flag=None,label1=None):
        """
        :param inputs
        :param inputs_length
        :param label_inputs(calulate the label expression)
        :param label_inputs_length
        :param all_list(all subclasses label expression)
        :param label_des(parent class label expression)
        :param classify(classify  input by parent class)
        :param flag(flag equals 1 means train and 0 mean test)
        """
        if inputs is None:
            label_out, all_list= self.decription(label_inputs, label_inputs_length)
            return label_out, all_list

        if flag is not None and flag==0:
            label_repeat_out = label_des.repeat((inputs.size(0), 1, 1))
            fact_out = self.RSANModel_Sub(inputs,inputs_length)
            output_feature = self.RSANModel(fact_out,inputs_length,label_repeat_out)
            logits = self.final_fc(output_feature)

            law_list = []
            for i,item in enumerate(classify):
                if len(item)>0:
                    law_part = all_list[i].unsqueeze(0).repeat(len(item), 1,1)
                else:
                    law_part = torch.Tensor([])
                law_list.append(law_part)


            evidence = []
            evidence_len = []
            logits_law = [[] for i in range(8)]
            for item in classify:
                part_evilen = inputs_length[item]
                evidence_len.append(part_evilen)
                if len(part_evilen)>0:
                    maxlen = max(part_evilen)
                    part_evi = fact_out[item]
                    part_evi = part_evi[:,:maxlen,:]
                    evidence.append(part_evi)
                else:
                    evidence.append(fact_out[item])

            if len(evidence_len[0])>0 :
                logits_law[0] = self.coatt1(evidence[0], evidence_len[0], law_list[0])
                logits_law[0] = self.fc1(logits_law[0])
            if len(evidence_len[1])> 0:
                logits_law[1] = self.coatt2(evidence[1], evidence_len[1], law_list[1])
                logits_law[1] = self.fc2(logits_law[1])
            if len(evidence_len[2])> 0:
                logits_law[2] = self.coatt3(evidence[2], evidence_len[2], law_list[2])
                logits_law[2] = self.fc3(logits_law[2])
            if len(evidence_len[3])> 0:
                logits_law[3] = self.coatt4(evidence[3], evidence_len[3], law_list[3])
                logits_law[3] = self.fc4(logits_law[3])
            if len(evidence_len[4]) > 0:
                logits_law[4] = self.coatt5(evidence[4], evidence_len[4], law_list[4])
                logits_law[4] = self.fc5(logits_law[4])
            if len(evidence_len[5])> 0:
                logits_law[5] = self.coatt6(evidence[5], evidence_len[5], law_list[5])
                logits_law[5] = self.fc6(logits_law[5])
            if len(evidence_len[6])> 0:
                logits_law[6] = self.coatt7(evidence[6], evidence_len[6], law_list[6])
                logits_law[6] = self.fc7(logits_law[6])
            if len(evidence_len[7])> 0:
                logits_law[7] = self.coatt8(evidence[7], evidence_len[7], law_list[7])
                logits_law[7] = self.fc8(logits_law[7])
            return logits, logits_law

        if flag is not None and flag == 1:
            parent_size = [[0, 17], [17, 71], [71, 91], [91, 104], [104, 159], [159, 162], [162, 176], [176, 183]]

            label_repeat_out = label_des.repeat((inputs.size(0), 1, 1))
            fact_out = self.RSANModel_Sub(inputs,inputs_length)
            output_feature = self.RSANModel(fact_out,inputs_length,label_repeat_out)
            logits = self.final_fc(output_feature)

            #predict the parent label
            output1 = (F.sigmoid(logits) > 0.5).squeeze()
            predict_label = torch.ByteTensor([0 for i in range(183)])
            if self.args.cuda:
                predict_label = predict_label.cuda()

            for index in range(8):
                # if predict the parent class correct, continue
                if output1[index] == True and label1.squeeze()[index] == 1:
                    sub = all_list[index].unsqueeze(0)

                    if index == 0:
                        output2 = self.coatt1(fact_out, inputs_length, sub)
                        output2 = self.fc1(output2)
                        logits2 = (F.sigmoid(output2) > 0.5).squeeze()
                        predict_label[parent_size[index][0]:parent_size[index][1]] = logits2
                    elif index == 1:
                        output2 = self.coatt2(fact_out, inputs_length, sub)
                        output2 = self.fc2(output2)
                        logits2 = (F.sigmoid(output2) > 0.5).squeeze()
                        predict_label[parent_size[index][0]:parent_size[index][1]] = logits2
                    elif index == 2:
                        output2 = self.coatt3(fact_out, inputs_length, sub)
                        output2 = self.fc3(output2)
                        logits2 = (F.sigmoid(output2) > 0.5).squeeze()
                        predict_label[parent_size[index][0]:parent_size[index][1]] = logits2
                    elif index == 3:
                        output2 = self.coatt4(fact_out, inputs_length, sub)
                        output2 = self.fc4(output2)
                        logits2 = (F.sigmoid(output2) > 0.5).squeeze()
                        predict_label[parent_size[index][0]:parent_size[index][1]] = logits2
                    elif index == 4:
                        output2 = self.coatt5(fact_out, inputs_length, sub)
                        output2 = self.fc5(output2)
                        logits2 = (F.sigmoid(output2) > 0.5).squeeze()
                        predict_label[parent_size[index][0]:parent_size[index][1]] = logits2
                    elif index == 5:
                        output2 = self.coatt6(fact_out, inputs_length, sub)
                        output2 = self.fc6(output2)
                        logits2 = (F.sigmoid(output2) > 0.5).squeeze()
                        predict_label[parent_size[index][0]:parent_size[index][1]] = logits2
                    elif index == 6:
                        output2 = self.coatt7(fact_out, inputs_length, sub)
                        output2 = self.fc7(output2)
                        logits2 = (F.sigmoid(output2) > 0.5).squeeze()
                        predict_label[parent_size[index][0]:parent_size[index][1]] = logits2
                    elif index == 7:
                        output2 = self.coatt8(fact_out, inputs_length, sub)
                        output2 = self.fc8(output2)
                        logits2 = (F.sigmoid(output2) > 0.5).squeeze()
                        predict_label[parent_size[index][0]:parent_size[index][1]] = logits2

            return output1.unsqueeze(0), predict_label.unsqueeze(0)

class RSANModel_Sub(nn.Module):
    def __init__(self, args):
        super(RSANModel_Sub, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.class_num = args.class_num
        # lstm args
        self.lstm_hidden_dim = args.hidden_size
        self.num_layers = 1
        self.fact_dynamic_lstm = DynamicGRU(input_dim=D,
                                             output_dim=self.lstm_hidden_dim,
                                             num_layers=self.num_layers,
                                             bidirectional=True,
                                             batch_first=True)
        # attention is all you need args
        self.num_headers = 8
        self.rsa_layers = 3

        # embedding layer
        self.embed = nn.Embedding(V, D, padding_idx=0)
        self.lstmInput = nn.LSTM(D, args.hidden_size, bidirectional=True, batch_first=True)

    def forward(self, inputs, input_length):
        ####1.  BEGIN embedding
        # DEVNOTES: embed fact descrption and label
        # (B, L) -> (B, L, E)
        embed_fact = self.embed(inputs)

        ####2.  BEGIN lstm embedding
        # DEVNOTES: lstm embedding
        fact_lstm_out, h = self.lstmInput(embed_fact)
        # (B, L, 2*H) -> (B, L, H)
        fact_out = fact_lstm_out[:, :, :self.lstm_hidden_dim] + fact_lstm_out[:, :, self.lstm_hidden_dim:]
        return fact_out

class RSANModel(nn.Module):
    def __init__(self, args, pretrained_embeddings=None):
        super(RSANModel, self).__init__()
        self.args = args
        self.pretrained_embeddings = pretrained_embeddings
        C = args.class_num
        self.class_num = args.class_num
        # lstm args
        self.lstm_hidden_dim = args.hidden_size
        self.num_layers = 1

        # attention is all you need args
        self.num_headers = 8
        self.rsa_layers = 3

        self.norm = LayerNorm(args.hidden_size)

        self.rsa_blocks = RSABlock()

        # self.final_fc = nn.Linear(args.hidden_size * 2, C)
        self.final_fc = FCLayer(self.lstm_hidden_dim*2, C, type="deep")


    def forward(self, inputs, inputs_length, label_inputs):

        """
        :param inputs
        :param inputs_length
        :param label_inputs
        :param label_inputs_length
        :return:output_feature
        """
        docs_len = Variable(torch.LongTensor([label_inputs.size(1)] * inputs_length.size(0))).cuda()

        FactAoA = inputs
        LabelAoA = label_inputs

        # (B,L,H), (B,LS,H)
        FactAoA, LabelAoA = self.rsa_blocks(FactAoA, inputs_length, LabelAoA, docs_len)
        FactAoA = self.norm(FactAoA)
        LabelAoA = self.norm(LabelAoA)

        # simple version
        # (B, L, H) -> (B, H)
        FactAoA_output = torch.mean(FactAoA,dim=1)
        # (B, LS, H) -> (B, H)
        LabelAoA_output = torch.mean(LabelAoA,dim=1)
        # (B, H) + (B, H) -> (B, 2H)
        output_feature = torch.cat((FactAoA_output, LabelAoA_output), dim=-1)

        return output_feature