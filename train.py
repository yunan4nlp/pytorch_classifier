import random
import  torch
from optparse import OptionParser
import torch.nn
import torch.autograd
import torch.nn.functional
from read import Reader
from instance import Feature
from instance import Example
from hyperparams import HyperParams
from model import RNNLabeler
from eval import Eval


class Labeler:
    def __init__(self):
        self.word_state = {}
        self.label_state = {}
        self.hyperParams = HyperParams()

    def createAlphabet(self, trainInsts, devInsts, testInsts):
        print("create alpha.................")
        for inst in trainInsts:
            for w in inst.words:
                if w not in self.word_state:
                    self.word_state[w] = 1
                else:
                    self.word_state[w] += 1

            l = inst.label
            if l not in self.label_state:
                self.label_state[l] = 1
            else:
                self.label_state[l] += 1

        print("word state:", len(self.word_state))
        self.addTestAlpha(devInsts)
        print("word state:", len(self.word_state))
        self.addTestAlpha(testInsts)
        print("word state:", len(self.word_state))

        self.word_state[self.hyperParams.unk] = self.hyperParams.wordCutOff + 1
        self.word_state[self.hyperParams.padding] = self.hyperParams.wordCutOff + 1

        self.hyperParams.wordAlpha.initial(self.word_state, self.hyperParams.wordCutOff)
        self.hyperParams.wordAlpha.set_fixed_flag(True)

        self.hyperParams.wordNum = self.hyperParams.wordAlpha.m_size

        self.hyperParams.unkWordID = self.hyperParams.wordAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.paddingID = self.hyperParams.wordAlpha.from_string(self.hyperParams.padding)

        self.hyperParams.labelAlpha.initial(self.label_state)
        self.hyperParams.labelAlpha.set_fixed_flag(True)
        self.hyperParams.labelSize = self.hyperParams.labelAlpha.m_size

        print("Label num: ", self.hyperParams.labelSize)
        print("Word num: ", self.hyperParams.wordNum)
        print("Padding ID: ", self.hyperParams.paddingID)
        print("UNK ID: ", self.hyperParams.unkWordID)

        outstr = open("C:\\Users\\yunan\\Desktop\\wordalpha", encoding='utf8', mode='w')
        for idx in range(self.hyperParams.wordAlpha.m_size):
            outstr.write(self.hyperParams.wordAlpha.from_id(idx))
            outstr.write('\n')
        outstr.close()

    def addTestAlpha(self, insts):
        print("Add test alpha.............")
        if self.hyperParams.wordFineTune == False:
            for inst in insts:
                for w in inst.words:
                    if (w not in self.word_state):
                        self.word_state[w] = 1
                    else:
                        self.word_state[w] += 1

    def extractFeature(self, inst):
        feat = Feature()
        feat.sentLen = len(inst.words)
        feat.wordIndexs = torch.autograd.Variable(torch.LongTensor(1, feat.sentLen))
        for idx in range(len(inst.words)):
            w = inst.words[idx]
            wordId = self.hyperParams.wordAlpha.from_string(w)
            if wordId == -1:
                wordId = self.hyperParams.unkWordID
            feat.wordIndexs.data[0][idx] = wordId
        return feat

    def instance2Example(self, insts):
        exams = []
        for inst in insts:
            example = Example()
            example.labelIndex = torch.autograd.Variable(torch.LongTensor(1))
            example.feat = self.extractFeature(inst)
            l = inst.label
            labelId = self.hyperParams.labelAlpha.from_string(l)
            example.labelIndex.data[0] = labelId
            exams.append(example)
        return exams

    def getBatchFeatLabel(self, exams):
        maxSentSize = 0
        batch = len(exams)
        for e in exams:
            if maxSentSize < e.feat.sentLen:
                maxSentSize = e.feat.sentLen
        if maxSentSize > 40:
            maxSentSize = 40
        batch_feats = torch.autograd.Variable(torch.LongTensor(batch, maxSentSize))
        batch_labels = torch.autograd.Variable(torch.LongTensor(batch))

        for idx in range(len(batch_feats.data)):
            e = exams[idx]
            batch_labels.data[idx] = e.labelIndex.data[0]
            for idy in range(maxSentSize):
                if idy < e.feat.sentLen:
                    batch_feats.data[idx][idy] = e.feat.wordIndexs.data[0][idy]
                else:
                    batch_feats.data[idx][idy] = self.hyperParams.paddingID
        if self.hyperParams.useCuda:
            return batch_feats.cuda(), batch_labels.cuda(), batch
        else:
            return batch_feats, batch_labels, batch


    def train(self, train_file, dev_file, test_file):
        self.hyperParams.show()
        torch.set_num_threads(self.hyperParams.thread)
        reader = Reader()

        trainInsts = reader.readInstances(train_file, self.hyperParams.maxInstance)
        devInsts = reader.readInstances(dev_file, self.hyperParams.maxInstance)
        testInsts = reader.readInstances(test_file, self.hyperParams.maxInstance)

        print("Training Instance: ", len(trainInsts))
        print("Dev Instance: ", len(devInsts))
        print("Test Instance: ", len(testInsts))

        self.createAlphabet(trainInsts, devInsts, testInsts)

        trainExamples = self.instance2Example(trainInsts)
        devExamples = self.instance2Example(devInsts)
        testExamples = self.instance2Example(testInsts)

        self.model = RNNLabeler(self.hyperParams)
        if self.hyperParams.useCuda:
            self.model.cuda()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.hyperParams.learningRate)

        indexes = []
        train_num = len(trainExamples)
        for idx in range(train_num):
            indexes.append(idx)

        batchBlock = len(trainExamples) // self.hyperParams.batch
        if train_num % self.hyperParams.batch != 0:
            batchBlock += 1
        for iter in range(self.hyperParams.maxIter):
            print('###Iteration' + str(iter) + "###")
            random.shuffle(indexes)
            self.model.train()
            for updateIter in range(batchBlock):
                #self.model.zero_grad()
                optimizer.zero_grad()
                exams = []
                start_pos = updateIter * self.hyperParams.batch
                end_pos = (updateIter + 1) * self.hyperParams.batch
                if end_pos > train_num:
                    end_pos = train_num
                for idx in range(start_pos, end_pos):
                    exams.append(trainExamples[indexes[idx]])
                feats, labels, batch = self.getBatchFeatLabel(exams)
                output = self.model(feats, batch)
                loss = torch.nn.functional.cross_entropy(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm(parameters, self.hyperParams.clip)
                optimizer.step()
                if (updateIter + 1) % self.hyperParams.verboseIter == 0:
                    print('current: ', idx + 1, ", cost:", loss.data[0])

            self.model.eval()
            eval_dev = Eval()
            for idx in range(len(devExamples)):
                predictLabel = self.predict(devExamples[idx])
                devInsts[idx].evalACC(predictLabel, eval_dev)
            print("dev: ", end='')
            eval_dev.getACC()

            eval_test = Eval()
            for idx in range(len(testExamples)):
                predictLabel = self.predict(testExamples[idx])
                testInsts[idx].evalACC(predictLabel, eval_test)
            print("test: ", end='')
            eval_test.getACC()

    def predict(self, exam):
        output = self.model(exam.feat.wordIndexs)
        labelID = self.getMaxIndex(output)
        return self.hyperParams.labelAlpha.from_id(labelID)

    def getMaxIndex(self, tag_score):
        max = tag_score.data[0][0]
        maxIndex = 0
        for idx in range(1, self.hyperParams.labelSize):
            if tag_score.data[0][idx] > max:
                max = tag_score.data[0][idx]
                maxIndex = idx
        return maxIndex


parser = OptionParser()
parser.add_option("--train", dest="trainFile",
                  help="train dataset")

parser.add_option("--dev", dest="devFile",
                  help="dev dataset")

parser.add_option("--test", dest="testFile",
                  help="test dataset")


random.seed(0)
torch.manual_seed(0)
(options, args) = parser.parse_args()
l = Labeler()
l.train(options.trainFile, options.devFile, options.testFile)

