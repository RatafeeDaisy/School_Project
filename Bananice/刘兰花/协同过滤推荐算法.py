# encoding = utf-8
# 导入计算开平方的公式

from math import sqrt

# 电影评分数据,同一用户对不同电影的评分
'''datas = {
    'A': {'老炮儿': 3.5, '唐人街探案': 1.0},
    'B': {'老炮儿': 2.5, '唐人街探案': 3.5, '星球大战': 3.0, '寻龙诀': 3.5,
          '神探夏洛克': 2.5, '小门神': 3.0},
    'C': {'老炮儿': 3.0, '唐人街探案': 3.5, '星球大战': 1.5, '寻龙诀': 5.0,
          '神探夏洛克': 3.0, '小门神': 3.5},
    'D': {'老炮儿': 2.5, '唐人街探案': 3.5, '寻龙诀': 3.5, '神探夏洛克': 4.0},
    'E': {'老炮儿': 3.5, '唐人街探案': 2.0, '星球大战': 4.5,
          '神探夏洛克': 3.5, '小门神': 2.0},
    'F': {'老炮儿': 3.0, '唐人街探案': 4.0, '星球大战': 2.0, '寻龙诀': 3.0,
          '神探夏洛克': 3.0, '小门神': 2.0},
    'G': {'老炮儿': 4.5, '唐人街探案': 1.5, '星球大战': 3.0, '寻龙诀': 5.0,
          '神探夏洛克': 3.5}
}'''

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



# 计算电影之间的相似度，需要的数据格式是mdatas[movie][user]
def transform_data(udata):
    # 存放同一部电影不用用户评分的字典
    movie_data = {}
    for user in udata:
        for movie in udata[user]:
            movie_data.setdefault(movie, {})
            # 用电影名称初始化电影-用户字典，用户对电影评分先设置为空值
            # 将用户和电影位置互调
            movie_data[movie][user] = udata[user][movie]
    return movie_data


print('同一电影不同用户的评分列表')
print(transform_data(datas))


# 基于欧式距离计算用户之间的相似度
def sim_ou(udata, person01, person02):
    # 先获取用户共同观影的电影名称列表
    common_movies = []
    # 先获取第一个人的评分电影名称
    for movie in udata[person01]:
        # 如果这部电影也出现在另一个人的评分列表
        if movie in udata[person02]:
            # 两个人都看了同一部电影，将电影放到共同观影列表中
            common_movies.append(movie)
    # 如果共同观影列表是空的，说明两个人没有共同喜好，相似度为0
    if len(common_movies) == 0:
        return 0

    # 基于欧式距离计算两个人在坐标轴上的距离，通过电影评分差距的平方和，再开方得到两点之间距离
    # ((x1-x2)*2+(y1-y2)*2+....)*1/2
    # pow方法可以用于求n次方，pow（3，2）计算3的平方
    sumSq = sum([pow(udata[person01][movie] - udata[person02][movie], 2)
                 for movie in common_movies])
    # 将欧式距离转换成相似度：1/(1+欧式距离)
    sim = 1 / (1 + sqrt(sumSq))
    return sim


# 基于欧式距离计算用户相似度，然后计算电影总分和总相似度，再计算加权平均分
def get_recommend_item(udata, target_person, sim_fun=sim_ou):
    # 电影总分
    sum_score = {}
    # 用户总相似度
    sum_sim = {}
    # 遍历用户评分数据列表，拿到其他用户的观影数据
    for other in udata:
        # 只拿非目标用户数据进行推荐，推荐目标用户没看过的电影
        if other != target_person:
            # 先计算用户相似度
            sim = sim_fun(udata, other, target_person)
            # 如果当前用户与目标用户没共同喜好，跳过分析下一个
            if sim <= 0:
                continue
            for movie in udata[other]:
                # 当前用户的电影目标用户没看过，计算相似分
                if movie not in udata[target_person]:
                    # 用电影名称初始化总分字典和总相似度字典
                    sum_score.setdefault(movie, 0)
                    sum_sim.setdefault(movie, 0)
                    # 相似度*原用户评分，再累计
                    sum_score[movie] += sim * udata[other][movie]
                    # 计算总相似度
                    sum_sim[movie] += sim
    # 为了去掉观影人数对总分的影响，需要进行加权平均分计算
    # 返回[（电影加权平均分，电影名称），...]
    rankings = [(sum_score[movie] / sum_sim[movie], movie) for movie in sum_sim]
    # 安装电影加权平均分从高到低进行排序，不能按照key排序，只能按照value排序
    rankings.sort(key=None, reverse=True)
    return rankings


print("基于欧式距离的用户协同过滤推荐算法，给A用户推荐的电影如下:")
print(get_recommend_item(datas, 'A'))


# 基于pearson距离计算相似度，以用户为例计算相似度
def sim_pearson(udata, person1, person2):
    # 先寻找共同数据，如果是用户，那就找共同观影名称，如果电影，那就找共同评分的用户
    # 105代码功能和45到49代码功能是一样的
    common_movies = [movie for movie in udata[person1] if movie in udata[person2]]

    # 计算pearson相似度是带有小数点的，先提前转换成float
    n = float(len(common_movies))
    if n == 0:
        return 0
    # 没有共同数据，说明两个人（电影）之间没有关联，默认返回相似度为0
    # pearson公式计算，先计算共同项集
    # 1、第一个数据集X，共同数据集common_movies的第一个人或者第一部电影所有评分的总和
    sumX = sum([udata[person1][movie] for movie in common_movies])

    # 2、第二个数据集Y，共同数据集common_movies的第二个人或者第二部电影所有评分的总和
    sumY = sum([udata[person2][movie] for movie in common_movies])

    # 3、求pearson公式第一项，X1*Y1+X2*Y2+...+Xn*Yn
    sumXY = sum([udata[person1][movie] * udata[person2][movie] for movie in common_movies])

    # 4、求pearson公式分母第三项，X1*X1+X2*X2+...Xn*Xn,也就是计算平方和
    # pow()可以计算N次方，比如pow（m，n）可以计算m的n次方
    sumXSq = sum([pow(udata[person1][movie], 2) for movie in common_movies])

    # 5、求pearson公式分母第五项，Y1*Y1+Y2*Y2+...Yn*Yn,也就是计算平方和
    # pow()可以计算N次方，比如pow（m，n）可以计算m的n次方
    sumYSq = sum([pow(udata[person2][movie], 2) for movie in common_movies])

    # 6、求pearson公式的分子
    num = sumXY - sumX * sumY / n

    # 7、求pearson公式的分母
    den = sqrt((sumXSq - pow(sumX, 2) / n) * (sumYSq - pow(sumY, 2) / n))

    # 除法分母不能为0
    if den == 0:
        # 如果分母等于0，没法进行除法计算，默认相似度为0
        return 0

    # pearson距离相似度值
    sim = num / den
    return sim


# 求用户或者电影最相似的几个同类
def top_match(udata, target, topN=5, sim_fun=sim_pearson):
    # 计算与当前目标不相等的所有同类之间的相似度
    # 返回目标与其他同类之间的相似度列表[(相似度，同类其他名称(A))，(相似度，同类其他名称(B))...]
    sim_items = [(sim_fun(udata, target, other), other) for other in udata if other != target]

    # 将相似度从大到小排序
    sim_items.sort(key=None, reverse=True)
    return sim_items[:topN]


print('基于欧式距离计算相似度，与A最相似的3个用户为：')
print(top_match(datas, 'A', 3, sim_ou))

# 计算电影相似度，需要传入同一部电影不同用户的评分
movies_data = transform_data(datas)
print('基于pearson距离计算相似度，与唐人街探案最相似的5部电影为：')
print(top_match(movies_data, '唐人街探案'))


# 要基于物品进行协同过滤，需要计算两两物品之间的相似度
def cal_sim_items(udata, num=10, sim_fun=sim_pearson):
    # 以电影为中心，计算两部电影的相似度，需要的数据是电影-用户评分
    m_data = transform_data(udata)

    # 所有电影相似度字典
    items_sim = {}

    # 遍历电影-用户评分列表，拿到电影名称，再计算其他电影和当前电影的相似度
    for movie in m_data:
        # 用当前电影名称作为key值初始化电影相似度字典，字典的value值是其他电影和它的相似度，先初始化为空列表
        items_sim.setdefault(movie, [])

        # 对于每部电影，计算最匹配的n部电影，将这些匹配数据放到对应电影的相似度列表，也就是字典的value中
        items_sim[movie] = top_match(m_data, movie, num, sim_fun)
    return items_sim


print('列出所有电影之间的相似度：')
print(cal_sim_items(datas))


# 基于物品的协同过滤进行电影推荐
# 电影A评分=电影（A，B）相似度*电影B的评分+电影（A，C）相似度*电影C的评分
# 电影B，电影C是目标用户已经看过的电影，电影A是想要推荐的电影
# 需要用到所有电影之间的相似度来进行计算
def get_sim_item(udata, target, movie_sim):
    # 基于物品的协同过滤不需要考虑用户数量，只需要计算电影总评分
    # 电影总评分字典
    sum_score = {}

    # 先找到用户已经看过的电影的评分，和基于用户协同过滤先排除用户已经看过的电影是相反的
    for movie in udata[target]:

        # 获取目标用户已看过的电影评分
        rating = udata[target][movie]

        # 遍历电影相似度列表，获取与目标用户已评分的电影和所有其他电影之间的相似度
        # 传进来的movie_sim电影相似度列表数据格式是【电影A与电影B的相似度，电影B的名称】
        for (sim_value, movie_name) in movie_sim[movie]:

            # 如果相似度列表里面的其他电影目标用户已经看过，跳过
            if movie_name in udata[target]:
                continue

            # 用其他未看过的电影名称作为key值初始化总评分字典,value值是电影预测推荐分数，默认初始化为0
            sum_score.setdefault(movie_name, 0)

            # 计算推荐电影总分，电影相似度*原电影评分，求和
            sum_score[movie_name] += sim_value * rating

    # 将评分从大到小排序
    item_list = sorted(sum_score.items(), key=lambda d: d[1], reverse=True)
    return item_list


# 先获取所有电影相似度列表
all_movies_sim = cal_sim_items(datas)

print('基于pearson距离的物品协同过滤推荐算法，给A用户推荐的电影如下：')
print(get_sim_item(datas, 'A', all_movies_sim))
