import sys
from pytorch_pretrained_bert.module import BertModel
import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np
import copy
import torch.nn.functional as F
import math
from src.vae import *



def gelu(x):
  return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))
class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings = config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')
        self.num_treatment = config.get('num_treatment')
        self.device = config.get('device')
        self.year_vocab_size = config.get('year_vocab_size')

        if config.get('poolingSize') is not None:
            self.poolingSize = config.get('poolingSize')
        if config.get('MEM') is not None:
            self.MEM = config.get('MEM')
        else:
            self.MEM=False
        if config.get('unsupSize') is not None:
            self.unsupSize = config.get('unsupSize')
        if config.get('unsupVAE') is not None:
            self.unsupVAE = True
        else:
            self.unsupVAE = False
        if config.get("vaeinchannels") is not None:
            self.vaeinchannels = config.get('vaeinchannels')
        if config.get("vaelatentdim") is not None:
            self.vaelatentdim = config.get('vaelatentdim')
        if config.get("vaehidden") is not None:
            self.vaehidden = config.get('vaehidden')
        if config.get("klpar") is not None:
            self.klpar = config.get('klpar')
        else:
            self.klpar = 1
        if config.get('BetaD') is not None:
            self.BetaD = config.get('BetaD')
        else:
            self.BetaD = False

class BertEmbeddingsUnsup(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddingsUnsup, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size).\
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))
        self.year_embeddings = nn.Embedding(config.year_vocab_size, config.hidden_size)
        self.unsuplist = config.unsupSize
        sumInputTabular = sum([el[1] for el in self.unsuplist])
        self.unsupEmbeddings =  nn.ModuleList([nn.Embedding(el[0], el[1]) for el in self.unsuplist])
        self.unsupLinear = nn.Linear(sumInputTabular, config.hidden_size)

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, year_ids=None):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)

        tabularVar = word_ids[:,:len(self.unsuplist)]
        word_ids = word_ids[:,len(self.unsuplist):]


        word_embed = self.word_embeddings(word_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)
        year_embed = self.year_embeddings(year_ids)
        tabularVar = tabularVar.transpose(1,0)
        tabularVarembed = torch.cat([self.unsupEmbeddings[eliter](el) for eliter, el in enumerate(tabularVar)], dim=1)
        tabularVarembed = self.unsupLinear(tabularVarembed).unsqueeze(1)
        embeddings = word_embed + age_embed  + year_embed + posi_embeddings
        embeddings = torch.cat((tabularVarembed, embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos/(10000**(2*idx/hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos/(10000**(2*idx/hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)

class BertLastPooler(nn.Module):
    def __init__(self, config):
        super(BertLastPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.poolingSize)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the last token. this is unlike first token pooling, just switched the ordering of the data
        first_token_tensor = hidden_states[:, -1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertVAEPooler(nn.Module):
    def __init__(self, config):
        super(BertVAEPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.poolingSize)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class TBEHRT(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(TBEHRT, self).__init__(config)
        if 'cuda' in config.device:

            self.device = int(config.device[-1])

            self.otherDevice =  self.device
        else:
            self.device = 'cpu'
            self.otherDevice = self.device
        self.bert = SimpleBEHRT (config)

        # self.bert = BertModel(config)
        self.treatmentC = nn.Linear(config.poolingSize, config.num_treatment)
        self.OutcomeT1_1 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT2_1 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT3_1 = nn.Linear(config.poolingSize, 2)

        self.OutcomeT1_2 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT2_2 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT3_2 = nn.Linear(config.poolingSize, 2)
        self.config = config
        self.dropoutVAE = nn.Dropout(config.hidden_dropout_prob)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.treatmentC.to(self.otherDevice)

        self.OutcomeT1_1.to(self.otherDevice)
        self.OutcomeT2_1.to(self.otherDevice)

        self.OutcomeT3_1.to(self.otherDevice)


        self.OutcomeT1_2.to(self.otherDevice)
        self.OutcomeT2_2.to(self.otherDevice)

        self.OutcomeT3_2.to(self.otherDevice)
        self.gelu = nn.ELU()
        self.num_labels = num_labels
        self.num_treatment = config.num_treatment
        self.logS = nn.LogSoftmax()

        self.VAEpooler = BertVAEPooler(config)
        self.VAEpooler.to(self.otherDevice)
        self.VAE = VAE(config)
        self.VAE.to(self.otherDevice)
        self.treatmentW = 1.0
        self.MEM = False
        self.config = config
        if config.MEM is True:
            self.MEM = True
            print('turning on the MEM....')
            self.cls = Bert.modeling.BertOnlyMLMHead(config, self.bert.bert.embeddings.word_embeddings.weight)
            self.cls.to(self.otherDevice)
        self.apply(self.init_bert_weights)
        print("full init completed...")

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, year_ids = None, attention_mask=None, masked_lm_labels=None,
                outcomeT=None, treatmentCLabel=None, fullEval=False, vaelabel = None):
        batchs = input_ids.shape[0]
        embed4MLM, pooled_out = self.bert(input_ids, age_ids, seg_ids, posi_ids , year_ids,  attention_mask,
                                                output_all_encoded_layers=False, fullmask = None)

        treatmentCLabel = treatmentCLabel.to(self.otherDevice)
        pooled_outVAE = self.VAEpooler(embed4MLM)
        pooled_outVAE = self.dropoutVAE(pooled_outVAE)

        outcomeT = outcomeT.to(self.otherDevice)
        outputVAE = self.VAE(pooled_outVAE, vaelabel)

        if self.MEM==True:
            prediction_scores = self.cls(embed4MLM[:,1:])

            if masked_lm_labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        else:
            prediction_scores = torch.ones([batchs, self.config.vocab_size]).to(self.otherDevice)
            masked_lm_loss = torch.tensor([0.0]).to(self.otherDevice)

        treatmentOut = self.treatmentC(pooled_out)
        tloss_fuct = nn.NLLLoss(reduction='mean')
        lossT = tloss_fuct(self.logS(treatmentOut).view(-1, self.config.num_treatment), treatmentCLabel.view(-1))
        pureSoft = nn.Softmax(dim=1)

        treatmentOut = pureSoft(treatmentOut)

        out1 = self.gelu(self.OutcomeT1_1(pooled_out))
        out2 = self.gelu(self.OutcomeT2_1(out1))
        logits0 = (self.OutcomeT3_1(out2))

        out12 = self.gelu(self.OutcomeT1_2(pooled_out))
        out22 = self.gelu(self.OutcomeT2_2(out12))
        logits1 = (self.OutcomeT3_2(out22))



        outcome1loss = nn.CrossEntropyLoss(reduction='none')
        outcome0loss = nn.CrossEntropyLoss(reduction='none')
        lossRaw0 =  outcome0loss(logits0,outcomeT.squeeze(-1))
        lossRaw1 = outcome1loss(logits1, outcomeT.squeeze(-1))

        trueLoss1 = torch.mean(lossRaw1*(treatmentCLabel.type(torch.FloatTensor).squeeze(-1)).to(self.otherDevice))
        trueLoss0 = torch.mean(lossRaw0*(1-treatmentCLabel.type(torch.FloatTensor).squeeze(-1)).to(self.otherDevice))

        tloss = trueLoss0 +trueLoss1 + self.treatmentW*lossT


        outlog1 = pureSoft(logits1)[:,1]
        outlog0 = pureSoft(logits0)[:,1]
        outlogits = torch.cat((outlog0.view(-1, 1).unsqueeze(0), outlog1.view(-1, 1).unsqueeze(0)), dim=0 )
        fulTout = treatmentOut
        outputTreatIndex = treatmentCLabel
        outlabelsfull = outcomeT


        # vae loss
        vaeloss = self.VAE.loss_function(outputVAE)
        vaeloss_total = vaeloss['loss']
        masked_lm_loss=  masked_lm_loss+ vaeloss_total
        return masked_lm_loss, tloss, prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.contiguous().view(  -1), fulTout, outputTreatIndex, outlogits, outlabelsfull, outputTreatIndex, 0, vaeloss


class TARNET_MEM(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(TARNET_MEM, self).__init__(config)
        if 'cuda' in config.device:

            self.device = int(config.device[-1])

            self.otherDevice =  self.device
        else:
            self.device = 'cpu'
            self.otherDevice = self.device
        self.bert = SimpleBEHRT (config)

        # self.bert = BertModel(config)
        self.treatmentC = nn.Linear(config.poolingSize, config.num_treatment)
        self.OutcomeT1_1 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT2_1 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT3_1 = nn.Linear(config.poolingSize, 2)


        self.OutcomeT1_2 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT2_2 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT3_2 = nn.Linear(config.poolingSize, 2)
        self.config = config
        self.dropoutVAE = nn.Dropout(config.hidden_dropout_prob)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.treatmentC.to(self.otherDevice)

        self.OutcomeT1_1.to(self.otherDevice)
        self.OutcomeT2_1.to(self.otherDevice)

        self.OutcomeT3_1.to(self.otherDevice)


        self.OutcomeT1_2.to(self.otherDevice)
        self.OutcomeT2_2.to(self.otherDevice)

        self.OutcomeT3_2.to(self.otherDevice)
        self.gelu = nn.ELU()
        self.num_labels = num_labels
        self.num_treatment = config.num_treatment
        self.logS = nn.LogSoftmax()





        self.VAEpooler = BertVAEPooler(config)
        self.VAEpooler.to(self.otherDevice)
        self.VAE = VAE(config)
        self.VAE.to(self.otherDevice)
        self.treatmentW = 1.0
        self.MEM = False
        self.config = config
        if config.MEM is True:
            self.MEM = True
            print('turning on the MEM....')
            self.cls = Bert.modeling.BertOnlyMLMHead(config, self.bert.bert.embeddings.word_embeddings.weight)
            self.cls.to(self.otherDevice)
        self.apply(self.init_bert_weights)
        print("full init completed...")

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, year_ids = None, attention_mask=None, masked_lm_labels=None,
                outcomeT=None, treatmentCLabel=None, fullEval=False, vaelabel = None):
        batchs = input_ids.shape[0]
        embed4MLM, pooled_out = self.bert(input_ids, age_ids, seg_ids, posi_ids , year_ids,  attention_mask,
                                                output_all_encoded_layers=False, fullmask = None)

        treatmentCLabel = treatmentCLabel.to(self.otherDevice)
        #
        pooled_outVAE = self.VAEpooler(embed4MLM)
        pooled_outVAE = self.dropoutVAE(pooled_outVAE)

        outcomeT = outcomeT.to(self.otherDevice)
        outputVAE = self.VAE(pooled_outVAE, vaelabel)

        masked_lm_loss = torch.tensor([0.0]).to(self.otherDevice)

        if self.MEM==True:
            prediction_scores = self.cls(embed4MLM[:,1:])

            if masked_lm_labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        else:
            prediction_scores = torch.ones([batchs, self.config.vocab_size]).to(self.otherDevice)
            masked_lm_loss = torch.tensor([0.0]).to(self.otherDevice)


        treatmentOut = self.treatmentC(pooled_out)
        tloss_fuct = nn.NLLLoss(reduction='mean')
        pureSoft = nn.Softmax(dim=1)

        treatmentOut = pureSoft(treatmentOut)

        out1 = self.gelu(self.OutcomeT1_1(pooled_out))
        out2 = self.gelu(self.OutcomeT2_1(out1))
        logits0 = (self.OutcomeT3_1(out2))

        out12 = self.gelu(self.OutcomeT1_2(pooled_out))
        out22 = self.gelu(self.OutcomeT2_2(out12))
        logits1 = (self.OutcomeT3_2(out22))


        outcome1loss = nn.CrossEntropyLoss(reduction='none')
        outcome0loss = nn.CrossEntropyLoss(reduction='none')
        lossRaw0 =  outcome0loss(logits0,outcomeT.squeeze(-1))
        lossRaw1 = outcome1loss(logits1, outcomeT.squeeze(-1))

        trueLoss1 = torch.mean(lossRaw1*(treatmentCLabel.type(torch.FloatTensor).squeeze(-1)).to(self.otherDevice))
        trueLoss0 = torch.mean(lossRaw0*(1-treatmentCLabel.type(torch.FloatTensor).squeeze(-1)).to(self.otherDevice))

        tloss = trueLoss0 +trueLoss1
        outlog1 = pureSoft(logits1)[:,1]

        outlog0 = pureSoft(logits0)[:,1]


        outlogits = torch.cat((outlog0.view(-1, 1).unsqueeze(0), outlog1.view(-1, 1).unsqueeze(0)), dim=0 )
        fulTout = treatmentOut

        outputTreatIndex = treatmentCLabel
        outlabelsfull = outcomeT







        # vae loss
        vaeloss = self.VAE.loss_function(outputVAE)
        vaeloss_total = vaeloss['loss']
        masked_lm_loss=  masked_lm_loss+ vaeloss_total
        return masked_lm_loss, tloss, prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.contiguous().view(  -1), fulTout, outputTreatIndex, outlogits, outlabelsfull, outputTreatIndex, 0, vaeloss




class SimpleBEHRT(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels=1):
        super(SimpleBEHRT, self).__init__(config)

        self.bert = BEHRTBASE(config)
        if 'cuda' in config.device:

            self.device = int(config.device[-1])

            self.otherDevice =  self.device
        else:
            self.device = 'cpu'
            self.otherDevice = self.device


        self.bert.to(self.device)

        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pooler = BertLastPooler(config)
        self.pooler.to(self.otherDevice)


        self.config = config
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids = None, posi_ids=None, year_ids=None, attention_mask=None,output_all_encoded_layers = False, fullmask = None):
        batchS = input_ids.shape[0]
        sequence_output, embedding_outputLSTM, embedding_outputLSTM2, attention_maskLSTM = self.bert(input_ids, age_ids ,seg_ids, posi_ids, year_ids,  attention_mask, fullmask = fullmask)

        pooled_out = self.pooler(embedding_outputLSTM)
        pooled_out = self.dropout(pooled_out)
        return embedding_outputLSTM, pooled_out

class BEHRTBASE(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BEHRTBASE, self).__init__(config)

        self.embeddings = BertEmbeddingsUnsup(config=config)

        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.config = config

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids = None, posi_ids=None, year_ids=None,  attention_mask=None, fullmask = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)

        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        encodermask = attention_mask
        encodermask = encodermask.unsqueeze(1).unsqueeze(2)
        encodermask = encodermask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encodermask = (1.0 - encodermask) * -10000.0



        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        embedding_output = self.embeddings(input_ids, age_ids,  seg_ids, posi_ids, year_ids)

        encoded_layers = self.encoder(embedding_output,
                                      encodermask,
                                      output_all_encoded_layers=False)
        sequenceOut = encoded_layers[-1]
        return [0], sequenceOut, [0], [0]


class DRAGONNET(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(DRAGONNET, self).__init__(config)
        if 'cuda' in config.device:

            self.device = int(config.device[-1])

            self.otherDevice =  self.device
        else:
            self.device = 'cpu'
            self.otherDevice = self.device
        self.bert = SimpleBEHRT (config)

        self.treatmentC = nn.Linear(config.poolingSize, config.num_treatment)
        self.OutcomeT1_1 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT2_1 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT3_1 = nn.Linear(config.poolingSize, 2)


        self.OutcomeT1_2 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT2_2 = nn.Linear(config.poolingSize, config.poolingSize)
        self.OutcomeT3_2 = nn.Linear(config.poolingSize, 2)
        self.config = config
        self.dropoutVAE = nn.Dropout(config.hidden_dropout_prob)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.treatmentC.to(self.otherDevice)

        self.OutcomeT1_1.to(self.otherDevice)
        self.OutcomeT2_1.to(self.otherDevice)

        self.OutcomeT3_1.to(self.otherDevice)


        self.OutcomeT1_2.to(self.otherDevice)
        self.OutcomeT2_2.to(self.otherDevice)

        self.OutcomeT3_2.to(self.otherDevice)
        self.gelu = nn.ELU()
        self.num_labels = num_labels
        self.num_treatment = config.num_treatment
        self.logS = nn.LogSoftmax()



        self.VAEpooler = BertVAEPooler(config)
        self.VAEpooler.to(self.otherDevice)
        self.VAE = VAE(config)
        self.VAE.to(self.otherDevice)
        self.treatmentW = 1.0
        self.MEM = False
        self.config = config
        if config.MEM is True:
            self.MEM = True
            print('turning on the MEM....')
            self.cls = Bert.modeling.BertOnlyMLMHead(config, self.bert.bert.embeddings.word_embeddings.weight)
            self.cls.to(self.otherDevice)
        self.apply(self.init_bert_weights)
        print("full init completed...")

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, year_ids = None,  attention_mask=None, masked_lm_labels=None,
                outcomeT=None, treatmentCLabel=None, fullEval=False, vaelabel = None):
        batchs = input_ids.shape[0]
        embed4MLM, pooled_out = self.bert(input_ids, age_ids, seg_ids, posi_ids , year_ids,  attention_mask,
                                                output_all_encoded_layers=False, fullmask = None)

        treatmentCLabel = treatmentCLabel.to(self.otherDevice)
        #
        pooled_outVAE = self.VAEpooler(embed4MLM)
        pooled_outVAE = self.dropoutVAE(pooled_outVAE)

        outcomeT = outcomeT.to(self.otherDevice)
        outputVAE = self.VAE(pooled_outVAE, vaelabel)

        if self.MEM==True:
            prediction_scores = self.cls(embed4MEM[:,1:])

            if masked_lm_labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        else:
            prediction_scores = torch.ones([batchs, self.config.vocab_size]).to(self.otherDevice)
            masked_lm_loss = torch.tensor([0.0]).to(self.otherDevice)

        treatmentOut = self.treatmentC(pooled_out)
        tloss_fuct = nn.NLLLoss(reduction='mean')
        lossT = tloss_fuct(self.logS(treatmentOut).view(-1, self.config.num_treatment), treatmentCLabel.view(-1))
        pureSoft = nn.Softmax(dim=1)

        treatmentOut = pureSoft(treatmentOut)

        out1 = self.gelu(self.OutcomeT1_1(pooled_out))
        out2 = self.gelu(self.OutcomeT2_1(out1))
        logits0 = (self.OutcomeT3_1(out2))

        out12 = self.gelu(self.OutcomeT1_2(pooled_out))
        out22 = self.gelu(self.OutcomeT2_2(out12))
        logits1 = (self.OutcomeT3_2(out22))
        outcome1loss = nn.CrossEntropyLoss(reduction='none')
        outcome0loss = nn.CrossEntropyLoss(reduction='none')
        lossRaw0 =  outcome0loss(logits0,outcomeT.squeeze(-1))
        lossRaw1 = outcome1loss(logits1, outcomeT.squeeze(-1))
        trueLoss1 = torch.mean(lossRaw1*(treatmentCLabel.type(torch.FloatTensor).squeeze(-1)).to(self.otherDevice))
        trueLoss0 = torch.mean(lossRaw0*(1-treatmentCLabel.type(torch.FloatTensor).squeeze(-1)).to(self.otherDevice))

        tloss = trueLoss0 +trueLoss1 + self.treatmentW*lossT
        outlog1 = pureSoft(logits1)[:,1]

        outlog0 = pureSoft(logits0)[:,1]


        outlogits = torch.cat((outlog0.view(-1, 1).unsqueeze(0), outlog1.view(-1, 1).unsqueeze(0)), dim=0 )
        fulTout = treatmentOut

        outputTreatIndex = treatmentCLabel
        outlabelsfull = outcomeT



        # vae loss
        vaeloss = self.VAE.loss_function(outputVAE)
        vaeloss_total = vaeloss['loss']
        masked_lm_loss=  masked_lm_loss+ vaeloss_total
        return masked_lm_loss, tloss, prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.contiguous().view(  -1), fulTout, outputTreatIndex, outlogits, outlabelsfull, outputTreatIndex, 0, vaeloss
