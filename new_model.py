import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
import numpy as np
class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs
class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.GELU = nn.GELU()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.GELU(x)
        return self.linear(x)
class Net(nn.Module):
    def __init__(self, config, bert_state_dict, vocab_len, device = 'cpu'):
        super().__init__()
        self.bert = BertModel(config)
        if bert_state_dict is not None:
            self.bert.load_state_dict(bert_state_dict)
        #self.bert.eval()
        self.rnn_ddi = nn.LSTM(bidirectional=True, num_layers=2, input_size=768*3, hidden_size=768*3//2, batch_first=True)
        self.rnn_ner = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        self.biaffine = Biaffine(768,7,bias_x=True,bias_y=True)
        self.FC = FCLayer(768*2,768)
        self.ner_fc = nn.Linear(768, vocab_len)
        self.ddi_fc = nn.Linear(768,7)
        self.enc_fc = nn.Linear(768, 2)
        self.pdc_fc = nn.Linear(768, 2)
        self.fc2 = nn.Linear(768*2,768)
        self.fc1 = nn.Linear(768,768)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.device = device
        self.task = 'ddi'
        self.cln = LayerNorm(768,768,conditional=True)

    def ddi_forward(self,is_subjects,ddi_tag_ids,tags_mask,enc,criterion,cls_enc):
        ddi_loss = torch.tensor(0.0).to(enc.device)
        ddi_logits = []
        for i in range(len(is_subjects)):
            is_subject = torch.FloatTensor(is_subjects[i]).to(self.device)
            ddi_tag_id = torch.LongTensor(ddi_tag_ids[i]).to(self.device)
            length_tensor = (is_subject != 0).sum(dim=0)
            #is_subject = is_subject.unsqueeze(1)  # b*1*maxlen
            sum_vector = torch.matmul(is_subject, enc.squeeze(0))
            avg_vector = sum_vector / (length_tensor.float())
            out = self.cln(enc.unsqueeze(0),avg_vector.unsqueeze(0).unsqueeze(0))
            logits = self.ddi_fc(out)
            ddi_logits.append(logits)
            active_loss = tags_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]

            active_tags = ddi_tag_id.view(-1)[active_loss]
            loss = criterion(active_logits, active_tags.to(active_logits.device))
            ddi_loss = torch.add(ddi_loss, loss)
        ddi_loss = torch.div(ddi_loss,len(is_subjects))
        return ddi_logits,ddi_loss
    def forward(self, x,mask,is_subjects,is_head,ner_tags_id,ddi_tags_id,tags_masks,seqlens,enc_tags,pdc_tags):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''

        x = x.to(self.device)
        mask = mask.to(self.device)
        maxlen = np.array(seqlens).max()
        is_head = is_head.to(self.device)
        DDI_logits = []
        DDI_loss = torch.tensor(0.0).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0,reduction='mean')
        criterion2 = torch.nn.CrossEntropyLoss(reduction='mean')
        encoded_layers, cls_enc = self.bert(input_ids=x, attention_mask=mask)
        with torch.no_grad():

            #enc = encoded_layers[-1]
            batch_size, max_len, f_dim = encoded_layers[-1].shape
            valid_ouput = torch.zeros(batch_size, max_len, f_dim, dtype=torch.float32, device=self.device)

            enc = encoded_layers[-1]
            for i in range(batch_size):
                jj = -1
                for j in range(maxlen):
                    if is_head[i][j] == 1:
                        jj += 1
                        valid_ouput[i][jj] = enc[i][j]
            enc = self.dropout(valid_ouput)
            cls = torch.mean(enc,dim=1)

        enc, _ = self.rnn_ner(enc)
        for i in range(batch_size):
            ddi_logits, ddi_loss = self.ddi_forward(is_subjects[i], ddi_tags_id[i], tags_masks[i],
                                                    enc[i], criterion,cls[i])
            DDI_loss = torch.add(DDI_loss,ddi_loss)
            DDI_logits.append(ddi_logits)
        DDI_loss = torch.div(DDI_loss,batch_size)
        cls = torch.mean(enc, dim=1)
        ENC_logits = self.enc_fc(cls)
        PDC_logits = self.pdc_fc(cls)
        ENC_loss = criterion2(ENC_logits, enc_tags.squeeze(1).to(ENC_logits.device))
        PDC_loss = criterion2(PDC_logits, pdc_tags.squeeze(1).to(PDC_logits.device))
        NER_logits = self.ner_fc(enc)
        active_loss = tags_masks.view(-1) == 1
        active_logits = NER_logits.view(-1, NER_logits.shape[-1])[active_loss]

        active_tags = ner_tags_id.view(-1)[active_loss]
        NER_loss = criterion(active_logits, active_tags.to(active_logits.device))
        return NER_logits,DDI_logits,ENC_logits,PDC_logits,NER_loss+DDI_loss+0.1*ENC_loss+0.1*PDC_loss