import jieba
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

inf = open("./datasets_cn/inf.txt", "r", encoding="gb18030").read()  # gb18030 utf-8
inf = inf.split(',')
counts = {}
stop = [line.strip() for line in open('cn_stopwords.txt', encoding="utf-8").readlines()]
stop.append(' ')
stop.append('\n')
stop.append('\u3000')
# extra_characters = {"，", "。", "\n", "“", "”", "：", "；", "？", "（", "）", "！", "…", "\u3000"}

for name in inf:
    with open("./datasets_cn/" + name + ".txt", "r", encoding="gb18030") as f:
        txt = f.read()
        print('正在分析的文件是：', name)
    words = jieba.lcut(txt)
    for word in words:
        counts[word] = counts.get(word,0)+1
    for word in stop:
        if word in counts:
            del counts[word]

items = list(counts.items())
items.sort(key=lambda x: x[1], reverse=True)
print(items[:10])
sort_list = sorted(counts.values(), reverse=True)


plt.title('Zipf-Law',fontsize=16)
plt.xlabel('排名',fontsize=14)
plt.ylabel('频率',fontsize=14)
x = [i for i in range(len(sort_list))]
plt.plot(x, sort_list , 'r')
plt.savefig('./Zipf_Law.jpg')
plt.show()

plt.title('Zipf-Law对数坐标轴',fontsize=16)
plt.xlabel('排名',fontsize=14)
plt.ylabel('频率',fontsize=14)
plt.yticks([pow(10,i) for i in range(0,4)])
plt.xticks([pow(10,i) for i in range(0,4)])
x = [i for i in range(len(sort_list))]
plt.yscale('log')
plt.xscale('log')
plt.plot(x, sort_list , 'r')
plt.savefig('./Zipf_Law对数坐标轴.jpg')
plt.show()