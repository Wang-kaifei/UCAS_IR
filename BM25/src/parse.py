import pandas
import gensim
from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

class Parser:
    """Parser类，CorpusParser和QueryParser共享单词预处理部分的方法"""
    def __init__(self, stop_word_path):
        self.stop_words = self.get_stopwords(stop_word_path)
        
    def get_stopwords(self, filename):
        """读取停用词文件，存储为列表"""
        res = []
        with open (filename, 'r') as f:
            lines = f.readlines()
            f.close()
        for word in lines:
            res.append(word.rstrip())
        return res
    
    def delet_words(self, src):
        """从src中去除停用表单词"""
        porter_stemmer = PorterStemmer()
        wordnet_lemmatizer = WordNetLemmatizer()
        src = [wordnet_lemmatizer.lemmatize(porter_stemmer.stem(i.lower())) for i  in src]
        res = []
        for word in src:
            if word not in self.stop_words:
                res.append(word)
        return res
    

class CorpusParser(Parser):

    def __init__(self, filename, stop_word_path = ""):
        if stop_word_path: 
            super(CorpusParser, self).__init__(stop_word_path)
        else:
            print("no stop_word")
            self.stop_words = []
        self.filename = filename
        self.corpus = {}

    def parse(self):
        """读取每个查询的待选文档
        思路：建立一个三层字典key=qid value = pid - p
        每个value都是一个小字典，存储pid - p"""
        q_p_dic = {}
        p_dic = {}
        df_tsv = pandas.read_table(self.filename, index_col=None, names=['qid','pid','q','p'])
        print("Read passage end.")
        for i in range(df_tsv.shape[0] - 1):
            p_dic[df_tsv['pid'][i]] = super().delet_words(df_tsv['p'][i].strip().split()) #存储小字典
            if df_tsv['qid'][i] != df_tsv['qid'][i + 1]: #如果下一行是对应的下一个q
                q_p_dic[df_tsv['qid'][i]] = p_dic  #存储字典
                p_dic = {} #清空（新建）
        #存储最后一个查询对应的字典
        p_dic[df_tsv['pid'][df_tsv.shape[0] - 1]] = super().delet_words(df_tsv['p'][df_tsv.shape[0] - 1].strip().split()) #存储小字典
        q_p_dic[df_tsv['qid'][df_tsv.shape[0] - 1]] = p_dic 
        
        self.corpus = q_p_dic

    def get_corpus(self):
        return self.corpus


class QueryParser(Parser):

    def __init__(self, filename, w2v_model_path = "", stop_word_path = ""):
        
        if stop_word_path:   #获取停用词表
#            print("aaa", stop_word_path)
            super(QueryParser, self).__init__(stop_word_path)
        else:
            self.stop_words = []
        self.w2v_model = None
        if w2v_model_path:
            self.get_word_vec(w2v_model_path)
        else:
            print("no model_word")
        self.filename = filename
        self.queries = {}  #qid-q
        
    def get_word_vec(self, w2v_model_path):
         #获取词向量模型
        if w2v_model_path.split('/')[-1].split('.')[-1] == "txt":
            self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path,binary = False)
        else: 
            self.w2v_model  = gensim.models.word2vec.Word2Vec.load(w2v_model_path)

        
    def parse(self):
        #读取全部查询到list: self.queries
        df_tsv = pandas.read_table(self.filename, index_col=None, names=['qid','q'])
        #以字典形式存储到queries中，key=qid value = q words (list, 存储单词) 
        for i in range(df_tsv.shape[0]):
            q_words = super().delet_words(df_tsv['q'][i].strip().split())
            if self.w2v_model != None:
                q_words = self.query_extension(q_words)
            self.queries[df_tsv['qid'][i]] = q_words
            q_words = []
    
    def query_extension(self, q_words, k = 5):
        """查询扩展"""
        sim = []
#        print(len(q_words))
        for word in q_words:
            if word in self.w2v_model:
                sim_word_list = self.w2v_model.wv.most_similar_cosmul(word, topn = k)
                extend_list = []
                for e_word, _ in sim_word_list:
                    if _ > 0.75:
                        extend_list.append(e_word)
                sim.extend(super().delet_words(extend_list))
        q_words.extend(sim)
#        print(len(q_words))
        return q_words
    
    def get_queries(self):
        return self.queries


