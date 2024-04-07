# 参考链接https://blog.csdn.net/qq_45688080/article/details/130669630
import os
import jieba
import math
import collections
 
# 获取N元语言模型
def getNmodel(phrase_model, n, words_list):
    if n == 1:
        for i in range(len(words_list)):
            phrase_model[words_list[i]] = phrase_model.get(words_list[i], 0) + 1
    else:
        for i in range(len(words_list) - (n - 1)):
            if n == 2:
                condition_t = words_list[i]
            else:
                condition = []
                for j in range(n-1):
                    condition.append(words_list[i + j])
                condition_t = tuple(condition)
            phrase_model[(condition_t, words_list[i+n-1])] = phrase_model.get((condition_t, words_list[i+n-1]), 0) + 1
    return phrase_model
# 获取N元信息熵
def getNentropy(n, clean_zh_file_content):
    if n == 1:
        phrase_model = getNmodel({}, 1, clean_zh_file_content)
        model_lenth = len(clean_zh_file_content)
        entropy = sum(
            [-(phrase[1] / model_lenth) * math.log(phrase[1] / model_lenth, 2) for phrase in phrase_model.items()])
    elif n>1:
        phrase_model_pre = getNmodel({}, n-1, clean_zh_file_content)
        phrase_model = getNmodel({}, n, clean_zh_file_content)
        phrase_n_len = sum([phrase[1] for phrase in phrase_model.items()])
        entropy = 0
        for n_phrase in phrase_model.items():
            p_xy = n_phrase[1] / phrase_n_len
            p_x_y = n_phrase[1] /  phrase_model_pre[n_phrase[0][0]]
            entropy+=(-p_xy * math.log(p_x_y, 2))
    return entropy
 
import matplotlib.pyplot as plt
# 画图
def draw_img(imgs_folder, zh_file_entropy, type="word"):
    x_axis = [key for key in zh_file_entropy.keys()] # 书名为x轴
    entropy_one = [value[0] for key,value in zh_file_entropy.items()] # 信息熵为y轴
    entropy_two = [value[1] for key,value in zh_file_entropy.items()]
    entropy_three = [value[2] for key, value in zh_file_entropy.items()]
    entropy = []
    entropy.append(entropy_one)
    entropy.append(entropy_two)
    entropy.append(entropy_three)
    # 解决图片中中文乱码解决
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    # 遍历画图
    for index in range(len(entropy)):
        for i in range(len(x_axis)):
            plt.bar(x_axis[i], entropy[index][i], width=0.5)
        if type=="word":
            plt.title(str(index+1)+"元信息熵词分析")
        else:
            plt.title(str(index + 1) + "元信息熵字分析")
        # 设置x轴标签名
        plt.xlabel("书名")
        # 设置y轴标签名
        plt.ylabel("信息熵")
        # 显示
        plt.xticks(fontsize=7)
        if type=="word":
            plt.savefig(os.path.join(imgs_folder,str(index)+"word.jpg"))
        else:
            plt.savefig(os.path.join(imgs_folder, str(index) + "char.jpg"))
        plt.show()
        
if __name__ == "__main__":
    imgs_folder = "./result_img"
    inf = open("./datasets_cn/inf.txt", "r", encoding="gb18030").read()  # gb18030 utf-8
    inf = inf.split(',')
    counts = {}
    stop = [line.strip() for line in open('cn_stopwords.txt', encoding="utf-8").readlines()]
    stop.append(' ')
    stop.append('\n')
    stop.append('\u3000')
    # extra_characters = {"，", "。", "\n", "“", "”", "：", "；", "？", "（", "）", "！", "…", "\u3000"}
    zh_file_word_entropy = collections.defaultdict(list) # 用来记录n元词组信息熵
    zh_file_char_entropy = collections.defaultdict(list) # 用来记录n元字组信息熵
    for name in inf:
        with open("./datasets_cn/" + name + ".txt", "r", encoding="gb18030") as f:
            txt = f.read()
            print('正在分析的文件是：', name)
            words = jieba.lcut(txt)
            cleaned_words = []
            cleaned_chars = []
            for word in words:
                if word in stop:
                    continue
                cleaned_words.append(word)
            for char in txt:
                if char in stop:
                    continue
                cleaned_chars.append(char)
            # 计算该本小说的n元信息熵 此处：1-3元
            for i in range(1, 4):
                entropy_word = getNentropy(i, cleaned_words)
                entropy_char = getNentropy(i, cleaned_chars)
                zh_file_word_entropy[name].append(entropy_word)
                zh_file_char_entropy[name].append(entropy_char)
    # 输出不同小说的n元信息熵
    print(zh_file_word_entropy)
    print(zh_file_char_entropy)
    sum = [0, 0, 0]
    for key, value in zh_file_word_entropy.items():
        sum[0] += value[0]
        sum[1] += value[1]
        sum[2] += value[2]
    print("1元信息熵平均值：", sum[0] / len(zh_file_word_entropy))
    print("2元信息熵平均值：", sum[1] / len(zh_file_word_entropy))
    print("3元信息熵平均值：", sum[2] / len(zh_file_word_entropy))
    sum = [0, 0, 0]
    for key, value in zh_file_char_entropy.items():
        sum[0] += value[0]
        sum[1] += value[1]
        sum[2] += value[2]
    print("1元信息熵平均值：", sum[0] / len(zh_file_char_entropy))
    print("2元信息熵平均值：", sum[1] / len(zh_file_char_entropy))
    print("3元信息熵平均值：", sum[2] / len(zh_file_char_entropy))
    # 画图
    draw_img(imgs_folder, zh_file_word_entropy, type="word")
    draw_img(imgs_folder, zh_file_char_entropy, type="char")
    print("Finish!")