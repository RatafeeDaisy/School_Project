import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from keras.layers import Input, Embedding, Dense, Flatten
from keras.layers import Activation, concatenate
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.models import Model

bank = pd.read_csv('D:/School/School_Project/Bananice/Dataset/bank.csv')

# 对原始数据进行初步处理，复制原始数据集
bank_data = bank.copy()

# 将deposit变量转换为0-1型，转换为变量名为deposit_cat，删除原始变量deposi
bank_data['deposit_cat'] = bank_data['deposit'].map({'yes': 1, 'no': 0})
bank_data.drop('deposit', axis=1, inplace=True)

# 删除month和day变量
bank_data.drop('month', axis=1, inplace=True)
bank_data.drop('day', axis=1, inplace=True)

# pdays取值为-1的样本数，用10000替换
bank_data.loc[bank_data['pdays'] == -1, 'pdays'] = 10000
bank_data.info()

# 类别特征列和连续特征列
categorical_columns = ["job", "marital", "education",
                       "housing", "loan", "contact",
                       "campaign", "poutcome", 'default']
continuous_columns = ["age", "balance", "duration",
                      "pdays", "previous"]

# 将类别特征做0_1处理
wide_data = bank_data.copy()
for col in categorical_columns:
    onehot_feats = pd.get_dummies(wide_data[col], prefix=col, prefix_sep='.')
    wide_data.drop([col], axis=1, inplace=True)
    wide_data = pd.concat([wide_data, onehot_feats], axis=1)
wide_data.info()

# 得到0-1处理后的类别特征
train_cate_features = wide_data.iloc[:, 6:]

# 对类别特征做简单的2阶特征交叉
poly = PolynomialFeatures(degree=2, interaction_only=True)
train_cate_poly = poly.fit_transform(train_cate_features)

# 交叉后的变量特征数量
print(train_cate_poly.shape)
wide_input = Input(shape=(train_cate_poly.shape[1],))

# 将类别特征转换为数值
for col in categorical_columns:
    le = LabelEncoder()
    bank_data[col] = le.fit_transform(bank_data[col])

# 分割出训练的连续型特征和分类型特征
train_conti_features = bank_data[continuous_columns]
train_cate_features = bank_data[categorical_columns]

# 分割出训练和测试标签
y = bank_data.pop('deposit_cat')

# 将连续型特征做归一化处理
scaler = MinMaxScaler()
conti_features = scaler.fit_transform(train_conti_features)

# 为类别数据的每个特征创建Input层和Embedding层
cate_inputs = []
cate_embeds = []
for i in range(len(categorical_columns)):
    input_i = Input(shape=(1,), dtype='int32')
    dim = bank_data[categorical_columns[i]].nunique()
    embed_dim = 8  # 统一设置为8维向量
    embed_i = Embedding(dim, embed_dim, input_length=1)(input_i)
    flatten_i = Flatten()(embed_i)
    cate_inputs.append(input_i)
    cate_embeds.append(flatten_i)

# 连续型特征数据在全连接层统一输入
conti_input = Input(shape=(len(continuous_columns),))
conti_dense = Dense(256, use_bias=False)(conti_input)

# 把全连接层和各Embedding的输出合并在一起
concat_embeds = concatenate([conti_dense] + cate_embeds)
concat_embeds = Activation('relu')(concat_embeds)
bn_concat = BatchNormalization()(concat_embeds)

fc1 = Dense(256, activation='relu')(bn_concat)
bn1 = BatchNormalization()(fc1)
fc2 = Dense(128, activation='relu')(bn1)
bn2 = BatchNormalization()(fc2)
fc3 = Dense(64, activation='relu')(bn2)
deep_input = fc3

# 将Wide、Deep对最后一层的输入做合并
out_layer = concatenate([deep_input, wide_input])

# 定义最终的输入输出
inputs = [conti_input] + cate_inputs + [wide_input]
output = Dense(1, activation='sigmoid')(out_layer)

# 定义模型
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
input_data = [train_conti_features] + [train_cate_features.values[:, i] for i in
                                       range(train_cate_features.shape[1])] + [train_cate_poly]
history = model.fit(input_data, y.values,
                    validation_data=(input_data, y.values),
                    epochs=10,
                    batch_size=128)
test_loss, test_accuracy = model.evaluate(input_data, y)
print("Test accuracy:{}".format(test_accuracy))
