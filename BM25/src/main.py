from parse import Parser, QueryParser, CorpusParser
from query import QueryProcessor
import numpy as np
import argparse
import os

"""需要循环每个q，对于每个查询，每次都更新p计算池"""

def std_out(qid, pid, rank, score):
    """生成标准输出行
    <查询ID> Q0 <文档ID> <文档排序><文档评分> <系统ID>"""
    tmp = (qid, pid, rank, score)
    line = '{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tMY-BM25\n'.format(*tmp)
    return line



def main(passage_path, query_path, output_path, stop_words_path, w2v_path):
    
    #停用词路径，词向量模型路径
    stop_words_path = stop_words_path
    w2v_path = w2v_path
    query_path = query_path
    passage_path = passage_path
    #读取查询文件，构建查询表
    qp = QueryParser(query_path, w2v_path, stop_words_path)
    qp.parse()
    queries = qp.get_queries() #获取 qid-q
    print("Read queries end.")
    #读取passage文件
    cp = CorpusParser(passage_path, stop_words_path)
    cp.parse()
    corpus = cp.get_corpus() #获取 qid-{pid-p}
    print("Build passage pools end.")
    
    #每个查询根据各自候选池 进行PM25打分
    results = {}
    i = 1
    for qid in queries:
        print("No. %s query has been scored..." % i)
        proc = QueryProcessor(queries[qid], corpus[qid])
        results[qid] = proc.run_query() #得到一个pid-score的字典
        i += 1

    #排序后写出
    res_string = ""
    for qid in results:
        #对于每组结果按照分数排序
        sorted_res = sorted(results[qid].items(), key = lambda kv:(kv[1], kv[0]))
        sorted_res.reverse()
        rank = 1
        last_score = 0 #最后一个得分
        for i in sorted_res:  #pid - score
            res_string += std_out(qid, i[0], rank, i[1])
            rank += 1
        #遍历所有的pid，如果没有被写出则直接跟在最后
        if len(sorted_res) > 0:  #防止文章池太小(eg:5)没有找到一篇匹配文章的情况
            last_score = sorted_res[len(sorted_res) - 1][1]
            pids = np.array(sorted_res)[:, 0]
            for pid in corpus[qid]:
                if pid not in pids:
                    last_score -= 0.1  #每个减0.1
                    res_string += std_out(qid, pid, rank, last_score - 0.1)
                    rank += 1
        else:
            for pid in corpus[qid]:
                last_score -= 0.1  #每个减0.1
                res_string += std_out(qid, pid, rank, last_score - 0.1)
                rank += 1

    with open(os.path.join(output_path), "w") as f:
        f.write(res_string)
        f.close()

if __name__ == '__main__':
    
    test = "../trec_eval-9.0.7/trec_eval"
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_file_path", type = str, help='Input the test passages path!')
    parser.add_argument("queries_file_path", type = str, help='Input the test queries path!')
    parser.add_argument("output_dir", type = str, help='Input the output direction!')
    parser.add_argument("--stop_words_path", help="Optionally : Input the test stop-words file path!", type=str)
    parser.add_argument("--w2v_path", help="Optionally : Input the test words vector file path!", type=str)
    args = parser.parse_args()
    print("I am working hard ^-^....")
    outpath = os.path.join(args.output_dir, 'BM25.txt')
    main(args.corpus_file_path, args.queries_file_path,  outpath, args.stop_words_path, args.w2v_path)
    
    print("Let's evaluate the results...")

    os.system("../trec_eval-9.0.7/trec_eval -m ndcg_cut %s %s" % ("../2019qrels-pass.txt", outpath))

#    print (args.corpus_file_path)
#    print (args.queries_file_path)
#    print (args.stop_words_path)
#    print (args.w2v_path)

#    stop_words_path = '../stop_words/stopwords5.txt'
#    w2v_path = '/Users/kaifeiwang/Desktop/BM25-master/model/word2vec50.txt'
#    query_path = '/Users/kaifeiwang/Desktop/IR_2020_Project/msmarco-test2019-43-queries.tsv'
#    passage_path = '/Users/kaifeiwang/Desktop/IR_2020_Project/msmarco-passagetest2019-43-sorted-top1000.tsv'
#    out = '/Users/kaifeiwang/Desktop/'

