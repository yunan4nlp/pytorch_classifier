from instance import  Instance

class Reader:
    def readInstances(self, path, maxInst = -1):
        insts = []
        r = open(path)
        for line in r.readlines():
            inst = Instance()
            info = line.strip().split("||| ")
            inst.words = info[0].strip().split(' ')
            inst.label = info[1]
            if maxInst == -1 or (len(insts) < maxInst):
                insts.append(inst)
        r.close()
        return insts
