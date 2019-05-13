import torch
from torchvision import transforms
from sklearn import metrics
import torch.nn.functional as F
import numpy as np
from utils.datasets import *
import Make_Law_Label
from sklearn.metrics import confusion_matrix
import datetime
def cal_precision_recall(parent,y_true, y_pred):
    macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
    macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
    print("===== parent is {}, ma precision is {}, ma recall is {}".format(parent, macro_precision, macro_recall))

def cal_metric(y_true, y_pred):
    ma_p, ma_r, ma_f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
    # mi_p, mi_r, mi_f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
    acc = metrics.accuracy_score(y_true,y_pred)
    jaccard = metrics.jaccard_similarity_score(y_true, y_pred)
    hamming_loss = metrics.hamming_loss(y_true, y_pred)
    # average_f1 = (ma_f1 + mi_f1)/2 * 100
    return [(ma_p, ma_r, ma_f1), acc, jaccard, hamming_loss]

def cal_metrics(y_batch, y_predictions, loss):

    f1_score_macro = metrics.f1_score(np.array(y_batch), y_predictions, average='macro')
    macro_precision = metrics.precision_score(np.array(y_batch), y_predictions, average='macro')
    macro_recall = metrics.recall_score(np.array(y_batch), y_predictions, average='macro')
    # metrics.auc()
    f1_score_micro = metrics.f1_score(np.array(y_batch), y_predictions, average='micro')
    micro_precision = metrics.precision_score(np.array(y_batch), y_predictions, average='micro')
    micro_recall = metrics.recall_score(np.array(y_batch), y_predictions, average='micro')
    average_f1 = (f1_score_macro + f1_score_micro)/2 * 100

    time_str = datetime.datetime.now().isoformat()
    print("the time is : {}. the loss is: {}. the average f1 score is : {}".format(time_str, loss.data[0], average_f1))
    print("macro precision is: {}. macro recall is: {}.micro precision is: {}. micro recall is: {}.".format(macro_precision, macro_recall, micro_precision, micro_recall))
    return average_f1

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    parent_size = [[0, 17], [17, 71], [71, 91], [91, 104], [104, 159], [159, 162], [162, 176], [176, 183]]
    law_text, law_length, law_order, parent2law = Make_Law_Label.makelaw()
    steps = 0
    best_f1_score = 0.0
    sum_loss = 0
    model.train()
    for epoch in range(1, args.epoch + 1):
        start_test_time = datetime.datetime.now()
        print("==================== epoch:{} ====================".format(epoch))
        for batch in train_iter:
            text, text_lens, label1, label2, law = batch
            text, label1, label2= Variable(text),Variable(label1), Variable(label2)

            article_text, article_len = Variable(law_text), Variable(law_length)
            if args.cuda:
                text, label1 = text.cuda(), label1.cuda()
                label2 = label2.cuda()
                text_lens = text_lens.cuda()
                # law_text = law_text.cuda()
                article_text, article_len = article_text.cuda(), article_len.cuda()

            # we have parent classifier and sub classifier,sperate input by parent class and train
            parent_index = torch.nonzero(label1)
            classify = [[] for i in range(8)]
            label2_list = []
            for index in parent_index:
                classify[index[1]] = classify[index[1]] + [index[0]]
            # classify[5]= [i for i in range(len(text))]
            classify = [torch.LongTensor(item) for item in classify]
            for i, item in enumerate(classify):
                if(len(item)==1):
                    classify[i] =classify[i].repeat(2)
                    item = item.repeat(2)
                label2_part = label2[item]
                if len(label2_part) > 0:
                    label2_part = label2_part[:, parent_size[i][0]: parent_size[i][1]]
                label2_list.append(label2_part)

            optimizer.zero_grad()

            # label_des, all_list= model(label_inputs=article_text, label_inputs_length=article_len,epoch=epoch,step=steps)
            label_des, all_list = model(label_inputs=article_text, label_inputs_length=article_len)

            logits,logits_list= model(inputs=text, inputs_length=text_lens, label_des=label_des,
                           all_list=all_list, classify=classify,flag=0)
            # print(steps)
            loss1 = torch.nn.functional.binary_cross_entropy_with_logits(logits, label1)
            loss2 = 0
            if len(label2_list[0]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[0], label2_list[0])
            if len(label2_list[1]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[1], label2_list[1])
            if len(label2_list[2]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[2], label2_list[2])
            if len(label2_list[3]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[3], label2_list[3])
            if len(label2_list[4]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[4], label2_list[4])
            if len(label2_list[5]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[5], label2_list[5])
            if len(label2_list[6]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[6], label2_list[6])
            if len(label2_list[7]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[7], label2_list[7])

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()


            sum_loss = sum_loss + loss.data
            steps = steps + 1
            if steps % args.print_freq == 0:
                print("##################### step is : {} ########################".format(steps))
                for i in range(8):
                    if len(logits_list[i])>0:
                        logits_numpy = (F.sigmoid(logits_list[i]).cpu().data.numpy() > 0.5).astype('int')
                        label_numpy = label2_list[i].cpu().data.numpy()
                        cal_precision_recall(i+1,label_numpy, logits_numpy)

            # if steps % args.test_freq == 0:
            #     eval(dev_iter, model, args, label_des,all_list)

            #     if valid_average_f1 > best_f1_score:
            #         best_f1_score = valid_average_f1
            #         last_step = steps
            #         if args.save_best:
            #             save(model, args.save_dir, args.save_dir.split("/")[0] + "_best", steps)
            # if steps % args.save_interval == 0:
            #     save(model, args.save_dir, args.save_dir.split("/")[0], steps)
        end_test_time = datetime.datetime.now()
        print("Train : epoch {}, time cost {}".format(epoch + 1, end_test_time - start_test_time))
        print("Train : sum loss {}, average loss {}".format(sum_loss, sum_loss / (steps)))
        sum_loss = 0
        steps = 0
        eval(dev_iter, model, args,label_des, all_list)
        if (epoch) % 5 == 0:
            adjust_learning_rate(optimizer)
            print("lr dec 5")

def eval(dev_iter, model, args,label_des,all_list):
    model.eval()
    avg_loss = 0.0
    avg_f1 = 0.0
    batch_num = 0
    pre_label1_list = []
    label1_list = []
    pre_label2_list = []
    label2_list = []
    start_test_time = datetime.datetime.now()
    print("======================== Evaluation =====================")
    for batch in dev_iter:
        batch_num = batch_num + 1

        text, text_lens, label1, label2, law = batch
        text, label2 = Variable(text), Variable(label2)

        if args.cuda:
            text, label2 = text.cuda(), label2.cuda()
            label1 = label1.cuda()
            text_lens = text_lens.cuda()

        logits,logits2 = model(inputs=text, inputs_length=text_lens, label_des=label_des,all_list=all_list,flag=1,label1=label1)

        pre_numpy1 = logits.cpu().data.numpy().astype('int')
        label1_numpy = label1.cpu().data.numpy()
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(logits2, label2)
        logits_numpy = logits2.cpu().data.numpy().astype('int')
        label_numpy = label2.cpu().data.numpy()

        pre_label1_list.append(pre_numpy1)
        label1_list.append(label1_numpy)

        pre_label2_list.append(logits_numpy)
        label2_list.append(label_numpy)
        # if batch_num == 100:
        #     break
    pre_label1_list = np.concatenate(pre_label1_list)
    pre_label2_list = np.concatenate(pre_label2_list)
    label1_list = np.concatenate(label1_list)
    label2_list = np.concatenate(label2_list)

    pre_sumlist = np.concatenate((pre_label1_list,pre_label2_list),1)
    label_sumlist = np.concatenate((label1_list,label2_list),1)

    parent_size = [[0, 17], [17, 71], [71, 91], [91, 104], [104, 159], [159, 162], [162, 176], [176, 183]]
    for j,item in enumerate(parent_size):
        cal_precision_recall(j+1,label2_list[:,item[0]:item[1]], pre_label2_list[:,item[0]:item[1]])

    (pma_p, pma_r, pma_f1), pacc, pjaccard, phamming_loss = cal_metric(label1_list, pre_label1_list)
    (ma_p, ma_r, ma_f1), acc, jaccard, hamming_loss = cal_metric(label2_list,pre_label2_list)

    (sma_p, sma_r, sma_f1), sacc, sjaccard, shamming_loss = cal_metric(label_sumlist, pre_sumlist)
    print(label_sumlist.shape)
    model.train()

    end_test_time = datetime.datetime.now()
    print("TestP: time cost {}".format(end_test_time - start_test_time))
    print("TestP: macro precision: {} macro recall: {}  ma f1 {}".format(pma_p, pma_r, pma_f1))
    print("TestP: Acc is {}".format(pacc))
    print("TestP: hamming is {}".format(phamming_loss))
    print("TestP: jaccard is {} ".format(pjaccard))

    # print("Test : time cost {}".format(end_test_time - start_test_time))
    print("TestC : macro precision: {} macro recall: {}  ma f1 {}".format(ma_p, ma_r, ma_f1))
    print("TestC : Acc is {}".format(acc))
    print("TestC : hamming is {}".format(hamming_loss))
    print("TestC : jaccard is {} ".format(jaccard))

    print("TestS : macro precision: {} macro recall: {}  ma f1 {}".format(sma_p, sma_r, sma_f1))
    print("TestS : Acc is {}".format(sacc))
    print("TestS : hamming is {}".format(shamming_loss))
    print("TestS : jaccard is {} ".format(sjaccard))
    model.train()
    # print("average loss is {}, average f1 is {}".format(avg_loss, avg_f1))
    return avg_loss, avg_f1
def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    print(1)