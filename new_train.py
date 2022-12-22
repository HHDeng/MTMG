import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from new_model import Net
from data_load import NerDataset, pad, HParams, ddi_measure, get_triplets, find_entities,get_final_output
import os
import numpy as np
from pytorch_pretrained_bert import BertConfig,BertAdam
import parameters
from collections import OrderedDict
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
import copy
import torch.nn.functional as F

# prepare biobert dict
# tmp_d = torch.load(parameters.BERT_WEIGHTS, map_location='cpu')
tmp_d = torch.load(parameters.BERT_WEIGHTS)
state_dict = OrderedDict()
for i in list(tmp_d.keys())[:199]:
    x = i
    if i.find('bert') > -1:
        x = '.'.join(i.split('.')[1:])
    state_dict[x] = tmp_d[i]


def train(model, iterator, optimizer, criterion, adjustment):
    model.train()
    for i, batch in enumerate(iterator):
        x, is_heads, seqlens, mask, is_subjects, ner_tags_id, ddi_tags_ids,tags_mask, word_seqlens, ddi_ord_id,enc_tags,pdc_tags= batch

        optimizer.zero_grad()
        NER_logits, DDI_logits,ENC_logits,PDC_logits,loss = model(x, mask, is_subjects, is_heads, ner_tags_id, ddi_tags_ids,tags_mask,seqlens,enc_tags,pdc_tags)  # logits: (N, T, VOCAB), y: (N, T)
        # logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        p_ENC = ENC_logits.argmax(-1).cpu().numpy().tolist()
        y_ENC = enc_tags.numpy() .tolist()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()

        if i % 10 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")


def eval(model, iterator, f, adjustment,epoch):
    model.eval()
    label_freq = {}
    NER_P, NER_Y = [], []
    ENC_P,ENC_Y = [], []
    PDC_P,PDC_Y = [], []
    Words, Is_heads, Tags, P, T, Is_subject,output_triplets,target_triplets,Idx= [], [], [], [], [], [],[],[],[]
    TP, M_tp, I_tp, E_tp, A_tp, F_tp = 0, 0, 0, 0, 0, 0
    TP_FP, M_tp_fp, I_tp_fp, E_tp_fp, A_tp_fp, F_tp_fp = 0, 0, 0, 0, 0, 0
    TP_FN, M_tp_fn, I_tp_fn, E_tp_fn, A_tp_fn, F_tp_fn = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, is_heads, seqlens, mask, is_subjects, NER_tags_id, DDI_tags_ids,tags_mask, word_seqlens, ddi_ord_id,enc_tags,pdc_tags= batch

            NER_logits, DDI_logits,ENC_logits,PDC_logits,_ = model(x, mask, is_subjects, is_heads, NER_tags_id, DDI_tags_ids,tags_mask, seqlens,enc_tags,pdc_tags)  # y_hat: (N, T)
            # if hp.is_use_adj:
            #     logits = logits + adjustment.to(logits.device)

            #y_p = torch.argmax(F.log_softmax(logits, dim=2), dim=2).cpu().numpy().tolist()

            #drug entities predicted results
            y_p_NER = NER_logits.argmax(-1).cpu().numpy().tolist()
            y_t_NER = NER_tags_id.numpy().tolist()
            tags_mask = tags_mask.numpy().tolist()

            #ENC and PDC tasks predicted results
            p_ENC = ENC_logits.argmax(-1).cpu().numpy().tolist()
            y_ENC = enc_tags.squeeze(1).numpy().tolist()
            ENC_P.extend(p_ENC)
            ENC_Y.extend(y_ENC)
            p_PDC = PDC_logits.argmax(-1).cpu().numpy().tolist()
            y_PDC = pdc_tags.squeeze(1).numpy().tolist()
            PDC_P.extend(p_PDC)
            PDC_Y.extend(y_PDC)

            for i, p_ner in enumerate(y_p_NER):
                for j, m in enumerate(p_ner):
                    if tags_mask[i][j] == 1:
                       NER_P.append(y_p_NER[i][j])
                       NER_Y.append(y_t_NER[i][j])

            #drug-drug interactions extracted resluts
            for i,p_ddi_logits in enumerate(DDI_logits):
                output_tris,target_tris = ddi_measure(p_ddi_logits,DDI_tags_ids[i],word_seqlens[i],ddi_ord_id[i],tags_mask[i])
                output_triplets.append(output_tris)
                target_triplets.append(target_tris)

   # print(1)
    M_tp, I_tp, E_tp, A_tp, F_tp, M_tp_fp, I_tp_fp, E_tp_fp, A_tp_fp, F_tp_fp,M_tp_fn, I_tp_fn, E_tp_fn, A_tp_fn, F_tp_fn=get_final_output(output_triplets,target_triplets)
    label_ENC = [0,1]
    label_PDC = [0,1]
    P_ENC = np.array(ENC_P)
    T_ENC = np.array(ENC_Y)
    P_PDC = np.array(PDC_P)
    T_PDC = np.array(PDC_Y)
    f1_nozero_ENC = f1_score(T_ENC, P_ENC, labels=label_ENC, average='micro')
    f1_nozero_PDC = f1_score(T_PDC, P_PDC, labels=label_PDC, average='micro')
    print("-------------------Auxiliary eval------------------------")
    print("f1_ENC=%.6f" % f1_nozero_ENC)
    print("f1_PDC=%.6f" % f1_nozero_PDC)
    P = np.array(NER_P)
    T = np.array(NER_Y)
    num_proposed = len(P[P > 0])
    num_correct = (np.logical_and(T == P, T > 0)).astype(int).sum()
    num_gold = len(T[T > 0])
    label_NER = [1, 2, 3, 4, 5, 6, 7, 8]
    label_DDI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    label_DDI_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    label_DDI_head = [1, 2, 3, 4, 5]
    precision_nozero_NER = precision_score(T, P, labels=label_NER, average='micro')
    recall_nozero_NER = recall_score(T, P, labels=label_NER, average='micro')
    f1_nozero_NER = f1_score(T, P, labels=label_NER, average='micro')
    report = classification_report(NER_Y, NER_P, labels=label_NER, digits=6)
    print("-------------------DNER eval------------------------")

    print("precision_nozero=%.6f" % precision_nozero_NER)
    print("recall_nozero=%.6f" % recall_nozero_NER)
    print("f1_nozero=%.6f" % f1_nozero_NER)
    DNER_P = precision_nozero_NER
    DNER_R = recall_nozero_NER
    DNER_F1 = f1_nozero_NER
    print("---------------------DDI eval--------------------------")
    print(A_tp,A_tp_fp,A_tp_fn)
    print(E_tp, E_tp_fp, E_tp_fn)
    print(M_tp, M_tp_fp, M_tp_fn)
    print(I_tp, I_tp_fp,I_tp_fn)
    print(F_tp, F_tp_fp, F_tp_fn)
    tp1 = A_tp + E_tp + M_tp +I_tp
    tp_fp1 = A_tp_fp +E_tp_fp +M_tp_fp +I_tp_fp
    tp_fn1 = A_tp_fn + E_tp_fn + M_tp_fn + I_tp_fn
    tp2 = tp1 + F_tp
    tp_fp2 = tp_fp1 + F_tp_fp
    tp_fn2= tp_fn1 + F_tp_fn
    print("Advise_F1=%.6f" % (2 * A_tp / (A_tp_fp + A_tp_fn)))
    print("Effect_F1=%.6f" % (2 * E_tp / (E_tp_fp + E_tp_fn)))
    print("Mechanism_F1=%.6f" % (2 * M_tp / (M_tp_fp + M_tp_fn)))
    print("Int_F1=%.6f" % (2 * I_tp / (I_tp_fp + I_tp_fn)))
    if tp_fp1 != 0:
        print("DDI_P=%.6f" % (tp1 / tp_fp1))
        print("DDI_R=%.6f" % (tp1 / tp_fn1))
    print("DDI_F1=%.6f" % (2 * tp1 / (tp_fp1 + tp_fn1)))
    DDI_P = tp1 / tp_fp1
    DDI_R = tp1 / tp_fn1
    DDI_F1 = 2 * tp1 / (tp_fp1 + tp_fn1)
    print("overall_DDI_F1=%.6f" % (2 * tp2 / (tp_fp2 + tp_fn2)))
    print("-------------------------------------------------")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    # final = f + ".P%.6f_R%.6f_F%.6f" % (precision_nozero, recall_nozero, f1_nozero) + '.txt'
    # with open(final, 'w') as fout:
    #     result = open(f, "r").read()
    #     fout.write(f"{result}\n")
    #
    #     fout.write(f"precision={precision_nozero}\n")
    #     fout.write(f"recall={recall_nozero}\n")
    #     fout.write(f"f1={f1_nozero}\n")
    #
    # os.remove(f)
    outfile = open("./recordall.txt", 'a', encoding='utf-8')
    outfile.write(
        "'\rEval--Epoch: %d, DNER_F1: %.4f DDI_F1: %.4f, ENC_F1: %.4f,  PDC_F1: %.4f" % (
        epoch, DNER_F1,DDI_F1,f1_nozero_ENC,f1_nozero_PDC))
    # print("precision=%.2f" % precision)
    # print("recall=%.2f" % recall)
    # print("f1=%.2f" % f1)
    return precision, recall, f1


if __name__ == "__main__":

    train_dataset = NerDataset("./joint_train.txt", 'NER')  
    eval_dataset = NerDataset("./joint_test.txt", 'NER')
    hp = HParams('NER')

    # Define model
    config = BertConfig(vocab_size_or_config_json_file=parameters.BERT_CONFIG_FILE)
    model = Net(config=config, bert_state_dict=state_dict, vocab_len=len(hp.VOCAB), device=hp.device)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    # update with already pretrained weight

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.eval_batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
    num_train_optimization_steps = int(len(train_iter)*hp.n_epochs)

    optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
    #optimizer = BertAdam(model.parameters(), lr=hp.lr, t_total=num_train_optimization_steps, warmup=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    label_freq = {}
    for i, batch in enumerate(train_iter):
        x, is_heads,seqlens, _, _, tags_id, _, _, _ ,_,_,_= batch
        tags_id = tags_id.numpy().tolist()
        for j in tags_id:
            for key in j:
                if key != 0:
                    label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    print(label_freq_array)
    adjustments = torch.from_numpy(np.log(label_freq_array ** 1.0 + 1e-12))
    zero = torch.zeros((1))
    adjustments = torch.cat((zero, adjustments))
    # criterion = PriorLoss(prior)
    for epoch in range(1, hp.n_epochs + 1):
        train(model, train_iter, optimizer, criterion, adjustments)
        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
        fname = os.path.join('checkpoints', str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname, adjustments,epoch)
        torch.save(model.state_dict(), f"{fname}.pt")
