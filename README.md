# 国科大现代信息检索
### 项目为国科大何苯老师现代信息检索课程大作业相关内容。
## 题目要求
完成TREC 2019 Deep Learning Passage Ranking竞赛中的rerank子任务。  
考虑到学生的硬件设备不足，老师给了从官方数据中随机抽取的部分样本，使用该数据统一评测。  
## 数据介绍
![image](https://github.com/Wang-kaifei/UCAS_IR/blob/main/IMG/data.png)
<br>训练数据有：
>Train Passages 每个片段内容映射成数字(pid)  
>Train Queries 每个查询内容映射成数字(qid)  
>Train Triples 三元组，每一行为(pid, pos_qid, neg_qid)  

测试数据有：
>Test Queries (查询, qid) 二元组  
>Test TopFile (qid, pid, 查询, 片段) 四元组  

对于每个在Test Queries文件中的查询(qid)，都能在Test TopFile文件中找到一组对应的passage(pid)。  
而这组passage是随机排列的，模型的功能是将这组passage按照与该查询的相关度从高到低排序。  
## BM25
选择BM25模型作为baseline，只用到了Test Queries和Test TopFile文件，具体见BM25文件夹下内容。

