import numpy as np
import pandas as pd
from keras.layers import *
import keras.backend as K
import tensorflow as tf
from tensorflow.python.keras.models import Model
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder

bank = pd.read_csv('D:/School/School_Project/Bananice/Dataset/bank.csv')
# 对原始数据进行初步处理，复制原始数据集
bank_data = bank.copy()
# 将deposit变量转换为0-1型，转换为变量名为deposit_ cat
# 删除原始变量deposit
bank_data['deposit_cat'] = bank_data['deposit'].map({'yes': 1, 'no': 0})
bank_data.drop('deposit', axis=1, inplace=True)
# 删除month和day变量
bank_data.drop('month', axis=1, inplace=True)
bank_data.drop('day', axis=1, inplace=True)
# pdays取值为-1的样本数，用10000替换
bank_data.loc[bank_data['pdays'] == -1, 'pdays'] = 10000
bank_data.info()

# 类别特征列和连续型特征列
categorical_columns = ["job", "marital", "education",
                       "housing", "loan", "contact",
                       "campaign", "poutcome", "default"]
continuous_columns = ["age", "balance", "duration",
                      "pdays", "previous"]

# 将类别特征做01处理
wide_data = bank_data.copy()
for col in categorical_columns:
    onebot_feats = pd.get_dummies(wide_data[col], prefix=col, prefix_sep='.')
    wide_data.drop([col], axis=1, inplace=True)
    wide_data = pd.concat([wide_data, onebot_feats], axis=1)
data = wide_data.copy()
sparse_feats = ['job.admin.', 'job.blue-collar', 'job.entrepreneur', 'job.housemaid', 'job.management',
                'job.retired', 'job.self-employed', 'job.services', 'job.student', 'job.technician',
                'job.unemployed', 'job.unknown', 'marital.divorced', 'marital.married', 'marital.single',
                'education.primary', 'education.secondary', 'education.tertiary', 'education.unknown',
                'housing.no', 'housing.yes', 'loan.no', 'loan.yes', 'contact.cellular', 'contact.telephone',
                'contact.unknown', 'campaign.1', 'campaign.2', 'campaign.3', 'campaign.4', 'campaign.5',
                'campaign.6', 'campaign.7', 'campaign.8', 'campaign.9', 'campaign.10', 'campaign.11',
                'campaign.12', 'campaign.13', 'campaign.14', 'campaign.15', 'campaign.16', 'campaign.17', 'campaign.18',
                'campaign.19', 'campaign.20', 'campaign.21', 'campaign.22', 'campaign.23', 'campaign.24', 'campaign.25',
                'campaign.26', 'campaign.27', 'campaign.28', 'campaign.29', 'campaign.30', 'campaign.31', 'campaign.32',
                'campaign.33', 'campaign.41', 'campaign.43', 'campaign.63', 'poutcome.failure', 'poutcome.other',
                'poutcome.success', 'poutcome.unknown', 'default.no', 'default.yes']
dense_feats = ["age", "balance", "duration", "pdays", "previous"]


def process_dense_feats(data, feats):
    d = data.copy()
    d = d[feats].fillna(0.0)
    for f in feats:
        d[f] = d[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
    return d


data_dense = process_dense_feats(data, dense_feats)


def process_sparese_feats(data, feats):
    d = data.copy()
    d = d[feats].fillna("-1")
    for f in feats:
        lable_encoder = LabelEncoder()
        d[f] = lable_encoder.fit_transform(d[f])
    return d


data_sparse = process_sparese_feats(data, sparse_feats)

total_data = pd.concat([data_dense, data_sparse], axis=1)
total_data['label'] = data['deposit_cat']

# 构建dense特征的输入
dense_inputs = []
for f in dense_feats:
    _input = Input([1], name=f)
    dense_inputs.append(_input)
# 将输入拼接到一起，方便连接Dense层
concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
# 然后连上输出为1个单元的全连接层，表示对dense变量的加权求和
fst_order_dense_layer = Dense(1)(concat_dense_inputs)
# 单独对每一个sparse特征构造输入，目的是方便后面构造二阶组合特征
sparse_inputs = []
for f in sparse_feats:
    _input = Input([1], name=f)
    sparse_inputs.append(_input)

sparse_1d_embed = []
for i, _input in enumerate(sparse_inputs):
    f = sparse_feats[i]
    voc_size = total_data[f].nunique()
    # 使用12正则化防止过拟合
    reg = tf.keras.regularizers.l2(0.5)
    _embed = Embedding(voc_size, 1, embeddings_regularizer=reg)(_input)
    # 由于Embedding的结果是二维的
    # 因此如果需要在Embedding之后Dense层，则需要先连接上Flatten层
    _embed = Flatten()(_embed)
    sparse_1d_embed.append(_embed)
# sparse特征加权求和
fst_order_sparse_layer = Add()(sparse_1d_embed)
# 合并Linear部分
linear_part = Add()([fst_order_dense_layer, fst_order_sparse_layer])
# 二阶特征
# embedding size
k = 8
# 这里考虑sparse的二阶交叉
sparse_kd_embed = []
for i, _input in enumerate(sparse_inputs):
    f = sparse_feats[i]
    voc_size = total_data[f].nunique()
    reg = tf.keras.regularizers.l2(0.7)
    _embed = Embedding(voc_size, k, embeddings_regularizer=reg)(_input)
    sparse_kd_embed.append(_embed)

# 将sparse的embeddin拼接起来，然后按照FM的特征组合公式计算
concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(concat_sparse_kd_embed)
square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])
sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(sum_kd_embed)
sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])
sub = Lambda(lambda x: x * 0.5)(sub)
snd_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)


# DNN部分
flatten_sparse_embed = Flatten()(concat_sparse_kd_embed)
fc_layer = Dropout(0.5)(Dense(256, activation='relu')(flatten_sparse_embed))
fc_layer = Dropout(0.3)(Dense(128, activation='relu')(fc_layer))
fc_layer = Dropout(0.1)(Dense(64, activation='relu')(fc_layer))
fc_layer_output = Dense(1)(fc_layer)


# 输出结果
output_layer = Add()([linear_part, snd_order_sparse_layer, fc_layer_output])
output_layer = Activation("sigmoid")(output_layer)
model = Model(dense_inputs + sparse_inputs, output_layer)
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

train_data = total_data
train_dense_x = [train_data[f].values for f in dense_feats]
train_sparse_x = [train_data[f].values for f in sparse_feats]
train_label = [train_data['label'].values]
model.fit(train_dense_x + train_sparse_x, train_label, epochs=100, batch_size=256,
          validation_data=(train_dense_x + train_sparse_x, train_label), )
test_loss, test_val_binary_crossentropy, test_accuracy = model.evaluate(train_dense_x + train_sparse_x, train_label)
print("Test accuracy:{}".format(test_accuracy))
