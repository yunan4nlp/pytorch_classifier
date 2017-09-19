import torch.nn as nn
import torch.autograd
import torch.nn.init as init
import numpy

class RNNLabeler(nn.Module):
    def __init__(self, hyperParams):
        super(RNNLabeler,self).__init__()

        self.hyperParams = hyperParams
        if hyperParams.wordEmbFile == "":
            self.wordEmb = nn.Embedding(hyperParams.wordNum, hyperParams.wordEmbSize)
            self.wordDim = hyperParams.wordEmbSize
        else:
            self.wordEmb, self.wordDim = self.load_pretrain(hyperParams.wordEmbFile, hyperParams.wordAlpha)
        self.wordEmb.weight.requires_grad = hyperParams.wordFineTune

        self.dropOut = nn.Dropout(hyperParams.dropProb)
        self.LSTM = nn.LSTM(input_size=self.wordDim,
                           hidden_size=hyperParams.rnnHiddenSize,
                           dropout=hyperParams.dropProb,
                           batch_first=True, num_layers=2, bidirectional=True)
        self.linearLayer = nn.Linear(hyperParams.rnnHiddenSize * 2, hyperParams.labelSize, bias=False)
        init.kaiming_uniform(self.linearLayer.weight)
        self.outputLayer = nn.Linear(hyperParams.rnnHiddenSize * 2, hyperParams.labelSize, bias=False)
        init.kaiming_uniform(self.outputLayer.weight)

    def init_hidden(self, batch):
        if self.hyperParams.useCuda:
            return (torch.autograd.Variable(torch.zeros(4, batch, self.hyperParams.rnnHiddenSize)).cuda(),
                    torch.autograd.Variable(torch.zeros(4, batch, self.hyperParams.rnnHiddenSize)).cuda())
        else:
            return (torch.autograd.Variable(torch.zeros(4, batch, self.hyperParams.rnnHiddenSize)),
                    torch.autograd.Variable(torch.zeros(4, batch, self.hyperParams.rnnHiddenSize)))

    def load_pretrain(self, file, alpha):
        f = open(file, encoding='utf-8')
        allLines = f.readlines()
        indexs = []
        info = allLines[0].strip().split(' ')
        embDim = len(info) - 1
        emb = nn.Embedding(self.hyperParams.wordNum, embDim)
        init.uniform(emb.weight,
                     a=-numpy.sqrt(3 / embDim),
                     b=numpy.sqrt(3 / embDim))
        oov_emb = torch.zeros(1, embDim).type(torch.FloatTensor)
        for line in allLines:
            info = line.strip().split(' ')
            wordID = alpha.from_string(info[0])
            if wordID >= 0:
                indexs.append(wordID)
                for idx in range(embDim):
                    val = float(info[idx + 1])
                    emb.weight.data[wordID][idx] = val
                    oov_emb[0][idx] += val
        f.close()
        count = len(indexs)
        for idx in range(embDim):
            oov_emb[0][idx] /= count

        unkID = alpha.from_string(self.hyperParams.unk)
        print('UNK ID: ', unkID)
        if unkID != -1:
            for idx in range(embDim):
                emb.weight.data[unkID][idx] = oov_emb[0][idx]

        print("Load Embedding file: ", file, ", size: ", embDim)
        oov = 0
        for idx in range(alpha.m_size):
            if idx not in indexs:
                oov += 1
        print("OOV Num: ", oov, "Total Num: ", alpha.m_size,
              "OOV Ratio: ", oov / alpha.m_size)
        print("OOV ", self.hyperParams.unk, "use avg value initialize")
        return emb, embDim

    def p_change(self, feat):
        print(feat)

    def forward(self, feat, batch = 1):
        sentSize = len(feat.data[0])
        wordRepresents = self.wordEmb(feat)
        wordRepresents = self.dropOut(wordRepresents)
        LSTMHidden = self.init_hidden(batch)
        LSTMOutputs, _ = self.LSTM(wordRepresents, LSTMHidden)
        #print(LSTMOutputs.permute(0, 2, 1).size())
        max_pool = torch.nn.functional.max_pool1d(LSTMOutputs.permute(0, 2, 1), sentSize)
        max_pool = torch.cat(max_pool.permute(0, 2, 1), 0)
        output = self.linearLayer(max_pool)
        return output








