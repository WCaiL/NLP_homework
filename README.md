# NLP_homework
# 深度学习与自然语言处理课程大作业
## 作业1
**题目要求**：第一部分：通过中文语料库来验证Zipf's Law. 第二部分：阅读《Entropy Of English》，计算中文(分别以词和字为单位) 的平均信息熵。
第一部分的的代码为zipf_law.py，第二部分的代码为entropy.py

## 作业2
**题目要求**：从下面链接给定的语料库中均匀抽取1000个段落作为数据集（每个段落可以有 K 个 token, K 可以取20，100，500, 1000, 3000），每个段落的标签就是对应段落所属的小说。利用LDA模型在给定的语料库上进行文本建模，主题数量为 T，并把每个段落表示为主题分布后进行分类（分类器自由选择），分类结果使用 10 次交叉验证（i.e. 900 做训练，剩余100 做测试循环十次）。实现和讨论如下的方面：（1）在设定不同的主题个数T的情况下，分类性能是否有变化？；（2）以"词"和以"字"为基本单元下分类结果有什么差异？（3）不同的取值的K的短文本和长文本，主题模型性能上是否有差异？
代码为LDA.py

## 作业3
**题目要求**：利用给定语料库（金庸小说语料如下链接），利用1～2 种神经语言模型（如：基于Word2Vec ， LSTM， GloVe等模型）来训练词向量，通过计算词向量之间的语意距离、某一类词语的聚类、某些段落直接的语意关联、或者其他方法来验证词向量的有效性。训练代码为train_LSTM.py，验证词向量代码为eval_embedding.py，预训练模型见release。

## 作业4
**题目要求**：利用给定语料库（金庸语小说语料链接见作业三），用Seq2Seq与Transformer两种不同的模型来实现文本生成的任务（给定开头后生成武侠小说的片段或者章节），并对比与讨论两种方法的优缺点 
