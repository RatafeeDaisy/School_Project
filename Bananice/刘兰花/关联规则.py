# encoding=utf-8
# 要想使用apriori关联规则推荐算法，需要先安装efficient_apriori包
# 推荐使用清华源进行安装
# pip install efficient_apriori -i https://pypi.tuna.tsinghua.edu.cn/simple
# 导入关联规则算法apriori

from efficient_apriori import apriori

datas = [
    ['牛奶', '面包'],
    ['面包', '尿布', '啤酒', '鸡蛋'],
    ['牛奶', '尿布', '啤酒', '可乐'],
    ['面包', '牛奶', '尿布', '啤酒'],
    ['面包', '牛奶', '尿布', '可乐']
]

# 使用apriori算法对交易数据进行分析挖掘
# 传入交易数据，支持度大于等于0.6，置信度大于等于0.75
# 返回频繁项列表和满足支持度和置信度的强关联规则
items, rules = apriori(datas, min_support=0.6, min_confidence=0.75)
print('频繁项集列表：', items)
print('强关联规则：', rules)
