import numpy as np
from torch.utils import data
import torch.nn as nn
import parameters
import torch
from pytorch_pretrained_bert import BertTokenizer
import copy

class HParams:
    def __init__(self, vocab_type):
        self.VOCAB_DICT = {
            'NER': ('<PAD>', 'B-drug', 'B-drug_n', 'B-group', 'B-brand',
                    'I-drug', 'I-drug_n', 'I-group', 'I-brand', 'O'),
            'DDI': ('<PAD>', 'B_S', 'B_false', 'B_mechanism', 'B_int', 'B_effect', 'B_advise',
                    'I_S', 'I_false', 'I_mechanism', 'I_effect', 'O'),
            'HeadDDI': ('<PAD>', 'mechanism', 'int', 'effect', 'advise', 'false', 'O')
        }
        self.VOCAB = self.VOCAB_DICT[vocab_type]
        self.tag2idx = {v: k for k, v in enumerate(self.VOCAB)}
        self.idx2tag = {k: v for k, v in enumerate(self.VOCAB)}
        self.tag2idx_hddi = {v: k for k, v in enumerate(self.VOCAB_DICT['HeadDDI'])}
        self.tag2idx_ddi = {v: k for k, v in enumerate(self.VOCAB_DICT['DDI'])}
        self.batch_size = 16
        self.eval_batch_size = 8
        self.lr = 0.001
        self.n_epochs = 30
        self.is_use_adj = False
        self.tokenizer = BertTokenizer(vocab_file=parameters.VOCAB_FILE, do_lower_case=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NerDataset(data.Dataset):
    def __init__(self, path, vocab_type):
        self.hp = HParams(vocab_type)
        instances = open(path).read().strip().split('\n\n')
        sents = []
        tags_li = []
        enc_tags = []
        pdc_tags = []
        for entry in instances:
            words = []
            tags = []
            lines = entry.splitlines()
            etag,ptag = lines[0].split("\t")
            enc_tags.append(int(etag))
            pdc_tags.append(int(ptag))
            del lines[0]
            t_num = len(lines[0].split("\t"))-1
            for idx in range(t_num):
                temp = []
                temp.append("<PAD>")
                tags.append(temp)
            for line in lines:
                w_t = line.split("\t")
                words.append(w_t[0])
                for idx in range(t_num):
                    tags[idx].append(w_t[idx+1])
            for idx in range(t_num):
                tags[idx].append("<PAD>")
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(tags)

        self.sents, self.tags_li = sents, tags_li
        self.enc_tags,self.pdc_tags = enc_tags,pdc_tags
    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]  # words, tags: string list
        # We give credits only to the first piece.
        x, y = [], []  # list of ids
        t_num = len(tags)
        enc_tag = []
        pdc_tag = []
        enc_tag.append(self.enc_tags[idx])
        pdc_tag.append(self.pdc_tags[idx])
        ner_tags_id = []
        ddi_tags_id = []
        ddi_org_id = []  #used for analysis entities
        is_heads = []  # list. 1: the token is the first piece of a word
        is_subjects = [] #use to generate the embedding of subject  in stage 2
        for i in range(t_num):
            if i == 0:
                continue
            is_sub = []
            ddi_tags = []
            ord_id = []
            for w,t in zip(words,tags[i]):
                tokens = self.hp.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                if t in ("B_S","I_S"):
                    sub = [1.0] * len(tokens)
                else:
                    sub = [0.0] * len(tokens)
                    # stage 2 only predict the head of objective entities，so just need the tag "B"
                ord_id.append(self.hp.tag2idx_ddi[t])
                if t in ('B_S', 'I_S'):
                    t = "O"
                if t in ('B_false', 'B_mechanism', 'B_int', 'B_effect', 'B_advise', 'I_false', 'I_mechanism', 'I_effect'):
                    t = t.split("_")[1]
                ddi_tags.append(self.hp.tag2idx_hddi[t])
                is_sub.extend(sub)
            is_subjects.append(is_sub)
            ddi_tags_id.append(ddi_tags)
            ddi_org_id.append(ord_id)
        for w, t in zip(words, tags[0]):
            tokens = self.hp.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.hp.tokenizer.convert_tokens_to_ids(tokens)
            is_head = [1] + [0] * (len(tokens) - 1)
            ner_tags_id.append(self.hp.tag2idx[t])
            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision

            x.extend(xx)
            is_heads.extend(is_head)


        # seqlen
        token_seqlen = len(x)
        word_seqlen = len(words)
        return x, is_heads, is_subjects, ner_tags_id, ddi_tags_id,token_seqlen,word_seqlen,ddi_org_id,enc_tag,pdc_tag


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]

    enc_tags = f(-2)
    pdc_tags = f(-1)
    seqlens = f(-5)
    word_seqlens = f(-4)
    maxlen = np.array(seqlens).max()
    ddi_org_id = f(-3)
    ddi_tags_ids = []
    for ids in f(4):
        temp = []
        for id in ids:
            temp.append(id + [0]*(maxlen-len(id)))
        ddi_tags_ids.append(temp)

    is_subjects = []
    for sub in f(2):
        temp = []
        for s in sub:
            temp.append(s + [0]*(maxlen - len(s)))
        is_subjects.append(temp)
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
    f2 = lambda x, seqlen: [sample[x] + [1] * (seqlen - len(sample[x])) for sample in batch]
    # 0: <pad>
    is_heads = f2(1, maxlen)
    ner_tags_id = f(3, maxlen)

    x = f(0, maxlen)
    mask = [[float(i != 0) for i in ii] for ii in x]
    tags_mask = [[i != 0 for i in ii] for ii in ner_tags_id]
    f = torch.LongTensor
    f1 = torch.FloatTensor
    return f(x), f(is_heads),seqlens, f(mask), is_subjects, f(ner_tags_id),ddi_tags_ids,f(tags_mask), word_seqlens, ddi_org_id,f(enc_tags),f(pdc_tags)
def find_entities(ord_id):
    sub_idx = []
    en_idx = []
    flag = False
    temp = []
    for idx,n in enumerate(ord_id):
        if sub_idx == None:
            continue
        if n == 1:
            sub_idx.append(idx)
        elif n == 7 and ord_id[idx+1] != 7:
            sub_idx.append(idx)
        if n in (2,3,4,5,6) and ord_id[idx+1] not in (8,9,10):
            temp.append(idx)
            en_idx.append(temp.copy())
            temp.clear()
        elif n in (2,3,4,5,6)  and ord_id[idx+1] in (8,9,10):
            temp.append(idx)
            flag = True
        if n == 11 and flag == True:
            temp.append(idx-1)
            en_idx.append(temp.copy())
            temp.clear()
            flag = False
    return sub_idx,en_idx


def ddi_measure(output, targets, length,ord_id,tag_mask):
    # tp,m_tp,i_tp,e_tp,a_tp,f_tp = 0,0,0,0,0,0
    # tp_fp,m_tp_fp,i_tp_fp,e_tp_fp,a_tp_fp,f_tp_fp = 0,0,0,0,0,0
    # tp_fn,m_tp_fn,i_tp_fn,e_tp_fn,a_tp_fn,f_tp_fn= 0,0,0,0,0,0

    output_tris = []
    target_tris = []

    for i in range(len(output)):
        out = output[i].squeeze(0).argmax(-1).cpu().numpy().tolist()
        sub_idx, en_idx = find_entities(ord_id[i])
        out = out[:length]
        target = targets[i][:length]
        out_triplets = get_triplets(out,sub_idx, en_idx)
        target_triplets = get_triplets(target,sub_idx, en_idx)
        for target_tri in target_triplets:
            if target_tri not in target_tris:
                target_tris.append(target_tri)
        for output_tri in out_triplets:
            if output_tri not in output_tris:
                output_tris.append(output_tri)
    return output_tris,target_tris
def get_final_output(output_triplets,target_triplets):
    tp,m_tp,i_tp,e_tp,a_tp,f_tp = 0,0,0,0,0,0
    tp_fp,m_tp_fp,i_tp_fp,e_tp_fp,a_tp_fp,f_tp_fp = 0,0,0,0,0,0
    tp_fn,m_tp_fn,i_tp_fn,e_tp_fn,a_tp_fn,f_tp_fn= 0,0,0,0,0,0
    for i,t_tris in enumerate(target_triplets):
        for t_tri in t_tris:
            #print(target_triplet)
            if t_tri[2] == 1:
                m_tp_fn += 1
            elif t_tri[2] == 2:
                i_tp_fn += 1
            elif t_tri[2] == 3:
                e_tp_fn += 1
            elif t_tri[2] == 4:
                a_tp_fn += 1
            elif t_tri[2] == 5:
                f_tp_fn += 1
        o_tris = output_triplets[i]
        if len(o_tris) == 0:
            continue
        tp_fp += len(o_tris)
        for o_tri in o_tris:
            if o_tri[2] == 1:
                m_tp_fp += 1
            elif o_tri[2] == 2:
                i_tp_fp += 1
            elif o_tri[2] == 3:
                e_tp_fp += 1
            elif o_tri[2] == 4:
                a_tp_fp += 1
            elif o_tri[2] == 5:
                f_tp_fp += 1
        for t_tri in t_tris:
            #print(target_triplet)
            for o_tri in o_tris:
                if o_tri == t_tri:
                    if o_tri[2] == 1:
                        m_tp += 1
                    elif o_tri[2] == 2:
                        i_tp += 1
                    elif o_tri[2] == 3:
                        e_tp += 1
                    elif o_tri[2] == 4:
                        a_tp += 1
                    else:
                        f_tp += 1
    return m_tp,i_tp,e_tp,a_tp,f_tp,m_tp_fp,i_tp_fp,e_tp_fp,a_tp_fp,f_tp_fp,m_tp_fn,i_tp_fn,e_tp_fn,a_tp_fn,f_tp_fn


def get_triplets(tags,sub_idx,en_idx):
    temp = {}
    triplets = []
    for idx, tag in enumerate(tags):
        if tag in (1,2,3,4,5):
            for en in en_idx:
                if idx == en[0]:
                    if idx > sub_idx[0]:
                        triplets.append((sub_idx, en, tag))
                        break

                    # if idx <= sub_idx[0]:
                    #     triplets.append((en, sub_idx, tag))
                    #     break
                    # else:
                    #     triplets.append((sub_idx, en, tag))
                    #     break

                # if len(en)>1:
                #     if idx > en[0] and idx <= en[1]:
                #         if idx <= sub_idx[0] and (en, sub_idx, tag) not in triplets:
                #             triplets.append((en, sub_idx, tag))
                #         elif idx > sub_idx[0] and (sub_idx, en, tag) not in triplets:
                #             triplets.append((sub_idx, en, tag))
    return triplets

class PriorLoss(nn.Module):
    def __init__(self, prior=None, num_classes=None, reduction="mean", eps=1e-12, tau=1.0):
        super(PriorLoss, self).__init__()
        self.loss_CL = torch.nn.CrossEntropyLoss(ignore_index=0)
        if not prior: prior = np.array([1 / num_classes for _ in range(num_classes)])  # 如果不存在就设置为num
        if type(prior) == list: prior = np.array(prior)
        self.log_prior = torch.from_numpy(np.log(prior ** tau + eps))
        self.eps = eps
        self.tau = tau

    def forward(self, logits, labels):
        logits = logits + self.log_prior.to(labels.device)
        loss = self.loss_CL(logits, labels)
        return loss
