import pickle

class HMM(object):

    def __init__(self, load=False):
        self.states = ['B', 'M', 'E', 'S']
        self.savepath = 'r_hmm_data.pkl'
        self.transprobmatrix = {}            # 状态转移概率矩阵
        self.observematrix = {}              # 发射概率矩阵
        self.initmatrix = {}                 # 初始状态分布概率矩阵
        self.wordcount = {}                  # 统计词频
        self.numberofline = 0
        if load:
            self.loadpara()

    # 将相关信息写入临时文件，方便后续使用，使用时读该文件即可
    def savepara(self):
        with open(self.savepath, 'wb') as f:
            pickle.dump(self.transprobmatrix, f)
            pickle.dump(self.observematrix, f)
            pickle.dump(self.initmatrix, f)
            pickle.dump(self.wordcount, f)
            pickle.dump(self.numberofline, f)

    def loadpara(self):
        with open(self.savepath, 'rb') as f:
            self.transprobmatrix = pickle.load(f)
            self.observematrix = pickle.load(f)
            self.initmatrix = pickle.load(f)
            self.wordcount = pickle.load(f)
            self.numberofline = pickle.load(f)

    # 初始化参数
    def initpara(self, trained=False):
        if trained:
            self.loadpara()
        else:
            for state in self.states:
                self.transprobmatrix[state] = {s: 0.0 for s in self.states}
                self.observematrix[state] = {}
                self.initmatrix[state] = 0.0
                self.wordcount[state] = 0.0

    def labelmark(self, text):
        length = len(text)
        if length == 1:
            return ['S']
        else:
            return ['B'] + ['M'] * (length - 2) + ['E']

    # 从语料中获取词频
    def traincorpus(self, file_path, trained=False):
        self.initpara(trained)
        # setofwords = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                words = [word for word in line.strip() if word != ' ']        # 字的集合
                self.numberofline += 1
                # upgrade words'set
                # setofwords.update(words)
                getwords = line.split()               # 词的集合
                word_state = []
                for word in getwords:                 # 给句子中每个词打标签
                    word_state.extend(self.labelmark(word))
                for index, status in enumerate(word_state):
                    self.wordcount[status] += 1       # 统计每个状态的频数
                    if index == 0:
                        self.initmatrix[status] += 1  # 第一个字为初始状态
                    # 统计状态转移概率和观测概率
                    else :
                        self.transprobmatrix[word_state[index - 1]][status] += 1
                        self.observematrix[status][words[index]] = self.observematrix[status].get(words[index], 0) + 1

    # 词频转化为概率
    def calculate(self):
        initmatrix = {k: l / self.numberofline for k, l in self.initmatrix.items()}
        transprobmatrix = {k: {k1: l1 / self.wordcount[k] for k1, l1 in l.items()} for k, l in self.transprobmatrix.items()}
        observematrix = {k: {k1: l1 / self.wordcount[k] for k1, l1 in l.items()} for k, l in self.observematrix.items()}
        return initmatrix, transprobmatrix, observematrix

    def viterbi(self, text):
        initmatrix, transprobmatrix, observematrix = self.calculate()
        localprob = [{}]               # 局部概率
        optimalpath = {}               # 最优路径（使得概率最大）
        # 初始化，计算初始时所有状态的局部概率，显然最优路径为每个状态自己
        for state in self.states:
            localprob[0][state] = initmatrix[state] * observematrix[state].get(text[0], 0)
            optimalpath[state] = [state]
        # 递归计算每个时刻的局部概率和最优路径
        for t in range(1, len(text)):
            localprob.append({})
            new_path = {}
            for state in self.states:
                # 这里乘观测概率是为了方便计算局部概率，最大值则是我们要的局部概率
                (tempprob, temp_state) = max([(localprob[t - 1][y0] * transprobmatrix[y0][state] * observematrix[state].get(text[t], 0), y0) for y0 in self.states])
                localprob[t][state] = tempprob                         # 迭代更新
                new_path[state] = optimalpath[temp_state] + [state]    # 更新最优路径
            optimalpath = new_path            # 更新该时刻所有状态的最优路径
        # 观测序列的概率即为该时刻的局部概率
        optimalpathprob, last_state = max([(localprob[len(text) - 1][y0], y0) for y0 in self.states])
        return optimalpath[last_state]

    def cut(self, text, best_path):
        begin, end = 0, 0
        cut_string = []
        for index, char in enumerate(text):
            signal = best_path[index]
            if signal == 'B':
                begin = index
            elif signal == 'E':
                cut_string.append(text[begin:index + 1])
                end = index + 1
            elif signal == 'S':
                cut_string.append(char)
                end = index + 1
        if end < len(text):
            cut_string.append(text[end:])
        return cut_string

    def use_cut(self, text):
        best_path = self.viterbi(text)        # 使用viterbi算法得到了最优路径
        return self.cut(text, best_path)

if __name__=='__main__':
    f = open('testset.txt', 'r', encoding='utf-8')
    lines = f.readlines()             # 读取测试集数据
    testcase = list()
    for line in lines:
        line = line.strip('\n')
        testcase.append(line)
    print(len(testcase))
    log = open('log.txt', mode='a', encoding='utf-8')
    for i in range(len(testcase)):
        training = HMM(load=True)
        print("第{}行分词结果已写入文件".format(i))
        print(training.use_cut(testcase[i]), file=log)
    # training = HMM(load=True)
    # initmatrix, transprobmatrix, observematrix = training.calculate()
    # print("初始化状态分布结果：{}".format(initmatrix))
    # print("转移概率矩阵：{}".format(transprobmatrix))
    # print("发射概率矩阵：{}".format(observematrix))

