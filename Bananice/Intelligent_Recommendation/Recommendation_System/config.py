# -*- coding: utf-8 -*-
# 电影类型
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance',
               'War', 'Comedy', 'Western', 'Documentary', 'Sci-Fi',
               'Drama', 'Thriller', 'Crime', 'Fantasy', 'Animation',
               'Mystery', "Children's", 'Musical', '0']

# 电影类型变量的取值
GENRE_FEATURES = {
    'gender': ['M', 'F'],
    'age': [18, 25, 35, 45, 50, 56],
    'occupation': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'userGenre1': genre_vocab,
    'userGenre2': genre_vocab,
    'userGenre3': genre_vocab,
    'userGenre4': genre_vocab,
    'userGenre5': genre_vocab,
    'movieGenre1': genre_vocab,
    'movieGenre2': genre_vocab,
    'movieGenre3': genre_vocab
}
