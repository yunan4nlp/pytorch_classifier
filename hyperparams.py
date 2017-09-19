class HyperParams:
    def __init__(self):
        self.wordNum = 0
        self.labelSize = 0

        self.unk = '-unk-'
        self.padding = '-padding-'
        self.unkWordID = 0
        self.paddingID = 0

        self.clip = 10
        self.maxIter = 20
        self.verboseIter = 20
        self.wordCutOff = 0
        self.wordEmbSize = 100
        self.wordFineTune = True
        #self.wordEmbFile = "E:\\py_workspace\\classifier\\data\\glove.twitter.27B.200d.txt"
        self.wordEmbFile = ""
        self.dropProb = 0.5
        self.rnnHiddenSize = 100
        self.hiddenSize = 100
        self.thread = 1
        self.learningRate = 0.001
        self.maxInstance = 10
        self.batch = 1
        self.useCuda = False

        self.wordAlpha = Alphabet()
        self.labelAlpha = Alphabet()
    def show(self):
        print('wordCutOff = ', self.wordCutOff)
        print('wordEmbSize = ', self.wordEmbSize)
        print('wordFineTune = ', self.wordFineTune)
        print('rnnHiddenSize = ', self.rnnHiddenSize)
        print('learningRate = ', self.learningRate)
        print('clip = ', self.clip)
        print('batch = ', self.batch)

        print('maxInstance = ', self.maxInstance)
        print('maxIter =', self.maxIter)
        print('thread = ', self.thread)
        print('verboseIter = ', self.verboseIter)


class Alphabet:
    def __init__(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = {}

    def from_id(self, qid, defineStr = ''):
        if int(qid) < 0 or self.m_size <= qid:
            return defineStr
        else:
            return self.id2string[qid]

    def from_string(self, str):
        if str in self.string2id:
            return self.string2id[str]
        else:
            if not self.m_b_fixed:
                newid = self.m_size
                self.id2string.append(str)
                self.string2id[str] = newid
                self.m_size += 1
                if self.m_size >= self.max_cap:
                    self.m_b_fixed = True
                return newid
            else:
                return -1

    def clear(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = {}

    def set_fixed_flag(self, bfixed):
        self.m_b_fixed = bfixed
        if (not self.m_b_fixed) and (self.m_size >= self.max_cap):
            self.m_b_fixed = True

    def initial(self, elem_state, cutoff = 0):
        for key in elem_state:
            if  elem_state[key] > cutoff:
                self.from_string(key)
        self.set_fixed_flag(True)

