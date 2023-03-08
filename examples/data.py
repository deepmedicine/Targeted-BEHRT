from torch.utils.data.dataset import Dataset
import torch
from src.utils import *
# data for var autoencoder deep unsup learning with tbehrt


class TBEHRT_data_formation(Dataset):
    def __init__(self, token2idx, dataframe, code= 'code', age = 'age', year = 'year' , static= 'static' , max_len=1000,expColumn='explabel', outcomeColumn='label',  max_age=110, yvocab=None, list2avoid=None, MEM=True):
        """
            The dataset class for the pytorch coded model, Targeted BEHRT

            token2idx - the dict that maps tokens in EHR to numbers /index
            dataframe - the pandas dataframe that has the code,age,year, and any static columns
            code - name of code column
            age - name of age column
            year - name of year column
            static - name of static column
            max_len - length of sequence
            yvocab - the year vocab for the year based sequence of variables
            expColumn - the exposure column for dataframe
            outcomeColumn - the outcome column
            MEM - the masked EHR modelling flag for unsupervised learning
            list2avoid - list of tokens /diseases to not include in the MEM masking procedure

              """

        if list2avoid is None:
            self.acceptableVoc = token2idx
        else:
            self.acceptableVoc = {x: y for x, y in token2idx.items() if x not in list2avoid}
            print("old Vocab size: ", len(token2idx), ", and new Vocab size: ", len(self.acceptableVoc))
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe[code]
        self.age = dataframe[age]
        self.year = dataframe[year]
        if outcomeColumn is None:
            self.label = dataframe.deathLabel
        else:
            self.label = dataframe[outcomeColumn]
        self.age2idx, _ = age_vocab(110, year, symbol=None)

        if expColumn is None:
            self.treatmentLabel = dataframe.diseaseLabel
        else:
            self.treatmentLabel = dataframe[expColumn]
        self.year2idx = yvocab
        self.codeS = dataframe[static]
        self.MEM = MEM
    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data

        age = self.age[index]

        code = self.code[index]
        year = self.year[index]

        age = age[(-self.max_len + 1):]
        code = code[(-self.max_len + 1):]
        year = year[(-self.max_len + 1):]


        treatmentOutcome = torch.LongTensor([self.treatmentLabel[index]])

        # avoid data cut with first element to be 'SEP'
        labelOutcome = self.label[index]

        
        # moved CLS to end as opposed to beginning.
        code[-1] = 'CLS'

        mask = np.ones(self.max_len)
        mask[:-len(code)] = 0
        mask = np.append(np.array([1]), mask)


        tokensReal, code2 = code2index(code, self.vocab)
        # pad age sequence and code sequence
        year = seq_padding_reverse(year, self.max_len, token2idx=self.year2idx)

        age = seq_padding_reverse(age, self.max_len, token2idx=self.age2idx)

        if self.MEM == False:
            tokens, codeMLM, labelMLM = nonMASK(code, self.vocab)
        else:
            tokens, codeMLM, labelMLM = randommaskreal(code, self.acceptableVoc)

        # get position code and segment code
        tokens = seq_padding_reverse(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        code2 = seq_padding_reverse(code2, self.max_len, symbol=self.vocab['PAD'])

        codeMLM = seq_padding_reverse(codeMLM, self.max_len, symbol=self.vocab['PAD'])
        labelMLM = seq_padding_reverse(labelMLM, self.max_len, symbol=-1)

        outCodeS = [int(xx) for xx in self.codeS[index]]
        fixedcovar = np.array(outCodeS )
        labelcovar = np.array(([-1] * len(outCodeS)) + [-1, -1])
        if self.MEM == True:
            fixedcovar, labelcovar = covarUnsupMaker(fixedcovar)
        code2 = np.append(fixedcovar, code2)
        codeMLM = np.append(fixedcovar, codeMLM)



        # code2 is the fixed static covariates while the codeMLM are the longutidunal one
        return torch.LongTensor(age), torch.LongTensor(code2), torch.LongTensor(codeMLM), torch.LongTensor(
            position), torch.LongTensor(segment), torch.LongTensor(year), \
               torch.LongTensor(mask), torch.LongTensor(labelMLM), torch.LongTensor(
            [labelOutcome]), treatmentOutcome,  torch.LongTensor(labelcovar)

    
    def __len__(self):
        return len(self.code)

