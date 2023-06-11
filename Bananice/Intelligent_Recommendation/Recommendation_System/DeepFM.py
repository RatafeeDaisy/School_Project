import config
import argparse
import tensorflow as tf
from matplotlib import pyplot as plt

# 设置绘图显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 读取训练样本
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value='0',
        num_epochs=1,
        ignore_errors=True
    )
    return dataset


if __name__ == '__main__':
    # 定义模型训练参数
    parser = argparse.ArgumentParser(description='Embedding Based recommendation system')
    parser.add_argument('--train_path', default='trainingSamples.csv', type=str, help='The path to train datset')
    parser.add_argument('--test_path', default='testSamples.csv', type=str, help='The path to test datset')
    args = parser.parse_args()

    # 读取训练数据，测试数据
    train_dataset = get_dataset(args.train_path)
    test_dataset = get_dataset(args.test_path)

    # 定义模型的输入
    inputs = {
        'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
        'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
        'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
        'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
        'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
        'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
        'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

        'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
        'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
        'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),

        'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
        'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
        'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
        'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
        'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
        'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
        'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
        'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),

        'gender': tf.keras.layers.Input(name='gender', shape=(), dtype='string'),
        'age': tf.keras.layers.Input(name='age', shape=(), dtype='int32'),
        'occupation': tf.keras.layers.Input(name='occupation', shape=(), dtype='int32')
    }

    # 模型输入特征定义
    # 电影的Id embedding特征，将整数连续值电影ID转换成One-Hot编码离散值
    movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=4000)

    # 对One-Hot编码之后的电影ID进行计数统计，One-Hot,同一部电影出现多次，计数会超过1
    movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)

    # One-Hot编码之后的电影ID是稀疏离散的，将其降维压缩，得到低维稠密的矩阵
    movie_ind_col = tf.feature_column.indicator_column(movie_col)

    # 电影类型embedding,将字符串类型的离散特征movieGenre1换成One-Hot编码离散值
    item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1",
                                                                               vocabulary_list=config.genre_vocab)
    item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, 10)
    item_genre_ind_col = tf.feature_column.indicator_column(item_genre_col)

    # 用户Id embedding 特征
    user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=6041)
    user_emb_col = tf.feature_column.embedding_column(user_col, 10)
    user_ind_col = tf.feature_column.indicator_column(user_col)

    # 用户历史评论过的电影类型embedding
    user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1",
                                                                               vocabulary_list=config.genre_vocab)
    user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, 10)
    user_genre_ind_col = tf.feature_column.indicator_column(user_genre_col)

    # 用户性别的embedding
    user_gender_col = tf.feature_column.categorical_column_with_vocabulary_list(key='gender',
                                                                                vocabulary_list=config.GENRE_FEATURES[
                                                                                    'gender'])
    user_gender_emb_col = tf.feature_column.embedding_column(user_gender_col, 10)
    user_gender_ind_col = tf.feature_column.indicator_column(user_gender_col)

    # 用户年龄的embedding
    user_age_col = tf.feature_column.categorical_column_with_vocabulary_list(key='age',
                                                                             vocabulary_list=config.GENRE_FEATURES[
                                                                                 'age'],
                                                                             dtype=tf.int32)
    user_age_emb_col = tf.feature_column.embedding_column(user_age_col, 10)
    user_age_ind_col = tf.feature_column.indicator_column(user_age_col)

    # 用户职业的embedding
    user_occupation_col = tf.feature_column.categorical_column_with_vocabulary_list(key='occupation',
                                                                                    vocabulary_list=
                                                                                    config.GENRE_FEATURES['occupation'],
                                                                                    dtype=tf.int32)
    user_occupation_emb_col = tf.feature_column.embedding_column(user_occupation_col, 10)
    user_occupation_ind_col = tf.feature_column.indicator_column(user_occupation_col)

    # FM模型一阶交叉特征，没有embedding,并且是直接级联输出
    fm_first_order_columns = [movie_ind_col, user_ind_col, user_genre_ind_col, item_genre_ind_col,
                              user_age_ind_col, user_gender_ind_col, user_occupation_ind_col]

    # Deep模型输入的特征
    deep_feature_columns = [tf.feature_column.numeric_column('releaseYear'),
                            tf.feature_column.numeric_column('movieRatingCount'),
                            tf.feature_column.numeric_column('movieAvgRating'),
                            tf.feature_column.numeric_column('movieRatingStddev'),
                            tf.feature_column.numeric_column('userRatingCount'),
                            tf.feature_column.numeric_column('userAvgRating'),
                            tf.feature_column.numeric_column('userRatingStddev'),
                            movie_emb_col,
                            user_emb_col]

    # 模型定义
    # 模型Embedding层定义
    item_emb_layer = tf.keras.layers.DenseFeatures([movie_emb_col])(inputs)
    user_emb_layer = tf.keras.layers.DenseFeatures([user_emb_col])(inputs)
    item_genre_emb_layer = tf.keras.layers.DenseFeatures([item_genre_emb_col])(inputs)
    user_genre_emb_layer = tf.keras.layers.DenseFeatures([user_genre_emb_col])(inputs)
    user_age_emb_layer = tf.keras.layers.DenseFeatures([user_age_emb_col])(inputs)
    user_gender_emb_col = tf.keras.layers.DenseFeatures([user_gender_emb_col])(inputs)
    user_occupation_emb_col = tf.keras.layers.DenseFeatures([user_occupation_emb_col])(inputs)

    # FM的一阶交叉特征层
    fm_first_order_layer = tf.keras.layers.DenseFeatures(fm_first_order_columns)(inputs)

    # FM对不同类别特征embedding进行交叉
    product_layer_item_user = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_emb_layer])
    product_layer_user_genre_item = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_genre_emb_layer])
    product_layer_user_age_genre_item = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_age_emb_layer])
    product_layer_user_gender_genre_item = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_gender_emb_col])
    product_layer_user_occupation_genre_item = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_occupation_emb_col])
    product_layer_item_genre_user_genre = tf.keras.layers.Dot(axes=1)([item_genre_emb_layer,
                                                                       user_genre_emb_layer])
    product_layer_item_genre_user = tf.keras.layers.Dot(axes=1)([item_genre_emb_layer, user_emb_layer])
    product_layer_item_genre_user_age = tf.keras.layers.Dot(axes=1)([item_genre_emb_layer, user_age_emb_layer])
    product_layer_item_genre_user_gender = tf.keras.layers.Dot(axes=1)([item_genre_emb_layer, user_gender_emb_col])
    product_layer_item_genre_user_occupation = tf.keras.layers.Dot(axes=1)([item_genre_emb_layer,
                                                                            user_occupation_emb_col])

    # Deep模型，使用所有特征
    deep = tf.keras.layers.DenseFeatures(deep_feature_columns)(inputs)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)

    # 将FM模型和Deep模型最后输出的特征进行级联
    concat_layer = tf.keras.layers.concatenate([fm_first_order_layer,
                                                product_layer_item_genre_user,
                                                product_layer_item_genre_user_genre,
                                                product_layer_item_genre_user_age,
                                                product_layer_item_genre_user_gender,
                                                product_layer_item_genre_user_occupation,
                                                product_layer_item_user,
                                                product_layer_user_genre_item,
                                                product_layer_user_age_genre_item,
                                                product_layer_user_gender_genre_item,
                                                product_layer_user_occupation_genre_item,
                                                deep],
                                               axis=1)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)
    model = tf.keras.Model(inputs, output_layer)

    # 编译模型
    # 设置模型训练使用的损失函数-二值交叉熵(binary_cross entropy)
    loss = tf.keras.losses.binary_crossentropy

    # 设置模型优化函数adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.8, beta_2=0.99)

    # 设置模型评估指标准确度(accuracy)、AUC
    acc = tf.keras.metrics.Accuracy()
    auc_ROC = tf.keras.metrics.AUC(curve='ROC')
    auc_PR = tf.keras.metrics.AUC(curve='PR')
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[acc, auc_ROC, auc_PR])

    # 定义模型保存的路径
    checkpoint_path = "./model"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     verbose=1)

    # 基于Wide&Deep模型的CTR预估、训练模型
    history = model.fit(train_dataset, epochs=5, validation_data=test_dataset, callbacks=[cp_callback])

    # 模型训练结果绘图
    his_dict = history.history
    loss_values = his_dict['loss']
    epochs = range(1, len(loss_values) + 1)
    plt.figure(figsize=(12, 9))
    plt.plot(epochs, loss_values, 'b', label='训练loss')
    plt.title("训练loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/Loss.png', pad_inches=0)
    plt.show()

    # 训练损失
    loss_values = his_dict['loss']
    epochs = range(1, len(loss_values) + 1)
    plt.figure(figsize=(12, 9))
    plt.plot(epochs, loss_values, 'b', label='训练loss')
    plt.title("训练loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/Loss.png', pad_inches=0)
    plt.show()

    # CTR预估的准确率
    acc_values = his_dict['accuracy']
    plt.figure(figsize=(12, 9))
    plt.plot(epochs, acc_values, 'b', label='训练集accuracy')
    plt.title("训练accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.savefig('./result/accuracy.png', pad_inches=0)
    plt.show()

    # CTR预估的AUC
    auc_values = his_dict['auc']
    plt.figure(figsize=(12, 9))
    plt.plot(epochs, auc_values, 'b', label='训练集auc_ROC')
    plt.title("训练集auc_ROC")
    plt.xlabel('Epochs')
    plt.ylabel('auc_ROC')
    plt.savefig('./result/auc_ROC.png', pad_inches=0)
    plt.show()

    # CTR预估的AUC
    auc_values = his_dict['auc_1']
    plt.figure(figsize=(12, 9))
    plt.plot(epochs, auc_values, 'b', label='训练集auc_PR')
    plt.title("训练集auc_PR")
    plt.xlabel('Epochs')
    plt.ylabel('auc_PR')
    plt.savefig('./result/auc_PR.png', pad_inches=0)
    plt.show()

    # 7、模型评估
    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
    print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss,
                                                                                       test_accuracy,
                                                                                       test_roc_auc,
                                                                                       test_pr_auc))
    # 8、输出预测结果
    predictions = model.predict(test_dataset)
    for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
        print("Predicted good rating: {:.2%}".format(prediction[0]),
              " | Actual rating label: ", ("Good Rating" if bool(goodRating) else "Bad Rating"))
