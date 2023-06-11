import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 预处理数据
def text_parse(big_string):
    token_list = big_string.split()
    return [tok.lower() for tok in token_list if len(tok) > 2]

# 去除列表中重复元素，并以列表形式返回
def create_vocab_list(data_set):
    vocab_set = set({})
    for d in data_set:
        vocab_set = vocab_set | set(d)
    return list(vocab_set)

# 统计每一文档（或邮件）在单词表中出现的次数，并以列表形式返回
def words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec

# 朴素贝叶斯主程序
doc_list, class_list, x = [], [], []
for i in range(1, 24):
    # 读取第i篇垃圾文件，并以列表形式返回
    word_list = text_parse(open('Emails/Training/spam/{0}.txt'.format(i)).read())
    doc_list.append(word_list)
    class_list.append(1)

    # 读取第i篇非垃圾文件，并以列表形式返回
    word_list = text_parse(open('Emails/Training/normal/{0}.txt'.format(i)).read())
    doc_list.append(word_list)
    class_list.append(0)

# 将数据向量化
vocab_list = create_vocab_list(doc_list)
for word_list in doc_list:
    x.append(words_to_vec(vocab_list, word_list))

# 分割数据为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, class_list, test_size=0.25)
x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), \
    np.array(y_train), np.array(y_test)

# 训练模型
nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)

# 测试模型效果
y_pred = nb_model.predict(x_test)

# 输出预测情况
print("正确值：{0}".format(y_test))
print("预测值：{0}".format(y_pred))
print("准确率：%f%%" % (accuracy_score(y_test, y_pred) * 100))