from math import sqrt

# 电影打分数据
datas = {
    '王芳': {'你好，李焕英': 7.8, '唐人街探案3': 8.9, '哥斯拉大战金刚': 8.0,
             '王牌特工源起': 8.8, '刺杀小说家': 5.0, '招魂3': 6.0},

    '张伟': {'你好，李焕英': 6.0, '唐人街探案3': 7.0,
             '哥斯拉大战金刚': 4, '王牌特工源起': 9.2, '招魂3': 6.0, '刺杀小说家': 7.8},

    '李娜': {'刺杀小说家': 6.0, '唐人街探案3': 8.0,
             '王牌特工源起': 8.5, '招魂3': 9.2},

    '王秀英': {'唐人街探案3': 8.1, '哥斯拉大战金刚': 7.2,
               '招魂3': 9.3, '王牌特工源起': 8.1, '刺杀小说家': 5.8},

    '张丽': {'你好，李焕英': 7.6, '唐人街探案3': 8.2, '哥斯拉大战金刚': 4.8,
             '王牌特工源起': 6.2, '招魂3': 6.4, '刺杀小说家': 4.3},

    '李强': {'你好，李焕英': 7.5, '唐人街探案3': 8.2, '招魂3': 6.8,
             '王牌特工源起': 9.8, '刺杀小说家': 7.6},

    '王磊': {'唐人街探案3': 9.2, '刺杀小说家': 3.0}
}


# 要计算基于电影的相似度，需要将上面的datas数据从datas[user][movie]格式转换成mdata[movie][user]
# 也就是将同一用户对不同电影的评分转换成同一电影不同用户的评分，将二维矩阵datas行列互换
def transform_data(udata):
    # 保存电影-用户评分数据的列表
    movie_data = {}
    for user in udata:  # 先获取用户数据
        for movie in udata[user]:  # 再获取当前用户对不同电影的评分数据
            # 将电影-用户数据字典用电影名称初始化
            movie_data.setdefault(movie, {})
            # 将电影名称和用户互调，可以通过直接写字典key值[key]来设置字典的数据
            movie_data[movie][user] = udata[user][movie]

    # 返回转换的数据字典
    return movie_data


print("1.同一电影不同用户评分数据：")
print(transform_data(datas))


# 基于曼哈顿距离计算用户相似度
def sim_manhadun(udata, person1, person2):
    # 先获取两个人之间共同的观影记录
    common_movies = []
    for movie in udata[person1]:  # 从第一个人的评分数据获取观影记录
        if movie in udata[person2]:  # 如果第一个人评分的电影也在第二个人的评分记录中
            # 两个人有共同的观影记录，将电影名称添加到共同观影记录列表中
            common_movies.append(movie)

    # 如果共同观影记录列表为空，说明两个人没有共同的喜好，相似度为0
    if len(common_movies) == 0:
        return 0

    # 有共同观影记录，转换到坐标轴计算两个人之间的曼哈顿式距离
    # |x1-y1|+|x2-y2|

    sim = sum([abs(udata[person1][movie] - udata[person2][movie]) for moive in common_movies])
    return sim


# 基于余弦距离计算电影之间的相似度
def sim_cos(udata, person1, person2):
    # 获取两个人共同观影列表,71行代码功能和45行到49行代码功能是一样的
    common_movies = [movie for movie in udata[person1] if movie in udata[person2]]
    # 获取共同观影列表长度，后面需要进行计算相似度，转换成float
    n = float(len(common_movies))
    if n == 0:
        # 如果两个人的共同观影列表长度为0，说明两个人没有共同喜好，相似度是0
        return 0
    # 计算余弦公式的公共项
    # 1.计算数据集X和数据集Y的乘积，再求和
    sumXY = sum([udata[person1][movie] * udata[person2][movie] for movie in common_movies])

    # 2.计算数据集X的平方和，pow(N,M)求N的M次方
    sumXSq = sum([pow(udata[person1][movie], 2) for movie in common_movies])

    # 3.计算数据集Y的平方和
    sumYSq = sum([pow(udata[person2][movie], 2) for movie in common_movies])

    # 5.计算余弦距离公式的分母
    den = sqrt(sumXSq * sumYSq)

    # 判断分母是否为0，分母为0不能进行除法运算
    if den == 0:
        # 如果分母为0，认为两个用户没有相关性
        return 0
    return sumXY / den


# 寻找数据集中最相似的TopN
def top_match(udata, target, topN=5, sim_fun=sim_cos):
    # 从数据集找到于target目标不一样的同类数据进行相似度的计算
    # 返回：[(相似度值,'otherA'),(相似度值,'otherB')...]
    sim_items = [(sim_fun(udata, target, other), other) for other in udata if other != target]
    # 对相似度进行排序，用value值排序，不是用key值
    sim_items.sort(key=None, reverse=True)

    return sim_items[:topN]


# 基于曼哈顿式距离计算用户相似度，然后进行电影推荐
def get_recommend_items(udata, target_person, sim_fun=sim_manhadun):
    # 推荐总分
    sum_score = {}
    # 用户总相似度
    sum_sim = {}
    for other in udata:  # 遍历每个用户，计算用户相似度
        if other != target_person:  # 目标用户不和自己进行相似度计算
            # 先计算相似度
            sim = sim_fun(udata, target_person, other)

            if sim <= 0:  # 欧式距离忽略相似度为0的情况
                continue

            for movie in udata[other]:  # 从其他用户的观影记录中获取推荐电影名称
                # 推荐的电影是目标用户没有看过的
                if movie not in udata[target_person]:
                    # 先用电影名称初始化推荐总分字典和用户总相似度字典
                    sum_score.setdefault(movie, 0)
                    sum_sim.setdefault(movie, 0)
                    # 将用户相似度*其他用户对电影的评分累计，当作每部电影的推荐总分
                    sum_score[movie] += sim * udata[other][movie]
                    # 总相似度就是用户之间相似度的总和
                    sum_sim[movie] += sim

    # 计算加权评价分
    rankings = [(sum_score[movie] / sum_sim[movie], movie) for movie in sum_sim]

    # 对评分排序,从大到小返回
    rankings.sort(key=None, reverse=True)
    return rankings


# 计算数据相似度矩阵，以电影相似度矩阵为例
def cal_sim_items(udata, num=10, sim_fun=sim_cos):
    # 先将数据转换成电影-不同用户的评分列表
    movies_data = transform_data(udata)
    # 相似度矩阵列表
    item_match = {}

    # 计算所有电影之间的相似度
    for movie in movies_data:
        # 初始化相似度列表，key值是电影名称，value值是空的列表，用来存放与其他电影的相似度
        item_match.setdefault(movie, [])
        # 对于每部电影，计算与他最相似的10部电影的相似度
        item_match[movie] = top_match(movies_data, movie, num, sim_fun)

    return item_match


# 基于余弦距离计算电影相似度，然后进行电影推荐
# 先更新所有电影之间的相似度
def get_sim_items(udata, target, movies_sim):
    # 基于物品的协同过滤不需要考虑用户数量，只需要计算电影的总评分
    # 电影总评分字典
    sum_score = {}
    # 先找到目标用户看过的电影的评分，和基于用户的协同过滤先排除用户已看过的电影相反
    for movie in udata[target]:
        # 获取目标用户已看过的电影评分
        rating = udata[target][movie]
        # 遍历电影相似度矩阵，找到与目标用户已评分电影的其他未观看电影之间的相似度
        # 传进来的movies_sim参数格式是[(电影A与电影B相似度,'电影B')...]
        for (sim_value, other_name) in movies_sim[movie]:
            # 如果其他电影已经被目标用户评分，则说明目标已观看该电影，跳过这部电影
            if other_name in udata[target]:
                continue
            # 用要推荐的电影名称初始化总评分字典，key值是电影名称，value值是预测推荐分，初始化为0
            sum_score.setdefault(other_name, 0)
            # 计算相似度评分，用电影的相似度*原电影评分，求和得到总推荐评分
            sum_score[other_name] += sim_value * rating

    # 将评分从大到小排序
    item_list = sorted(sum_score.items(), key=lambda d: d[1], reverse=True)

    return item_list


# 先计算电影相似度矩阵
all_movies_sim = cal_sim_items(datas)
print("2.给王磊用户推荐的3部电影如下：")
print(get_recommend_items(datas, '王磊')[0][1], get_sim_items(datas, '王磊', all_movies_sim)[1][0],
      get_sim_items(datas, '王磊', all_movies_sim)[2][0])
