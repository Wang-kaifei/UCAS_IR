from invdx import build_data_structures
from rank import score_BM25

#查询类
class QueryProcessor:
    
    def __init__(self, querie, corpus):
        self.querie = querie
        self.index, self.dlt = build_data_structures(corpus)

    def run_query(self):
        query_result = {}
        for term in self.querie:
            if term in self.index:
                doc_dict = self.index[term] #get all terms for every passage
                for docid, freq in doc_dict.items(): #for each document and its word frequency
                    score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt), dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length()) # calculate score
                    if docid in query_result: #this document has already been scored once
                        query_result[docid] += score
                    else:
                        query_result[docid] = score
        return query_result #pid - score
