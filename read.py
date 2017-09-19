from instance import  Instance
import re

class Reader:

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def readInstances(self, path, maxInst = -1):
        insts = []
        r = open(path)
        for line in r.readlines():
            inst = Instance()
            info = line.strip().split("||| ")
            sent = self.clean_str(info[0].strip())
            inst.words = sent.split(' ')
            inst.label = info[1]
            if maxInst == -1 or (len(insts) < maxInst):
                insts.append(inst)
        r.close()
        return insts

