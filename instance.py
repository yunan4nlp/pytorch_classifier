from eval import Eval

class Instance:
    def __init__(self):
        self.words = []
        self.label = ''

    def show(self):
        print(self.words, self.label)

    def evalACC(self, predict_labels, eval):
            if self.label in predict_labels:
                eval.correct_num += 1
            eval.predict_num += 1

class Feature:
    def __init__(self):
        self.wordIndexs = []
        self.sentLen = 0

class Example:
    def __init__(self):
        self.feat = Feature()
        self.labelIndex = -1

    def show(self):
        print(self.feat.wordIndexs)
        print(self.labelIndex)


