#-*- coding: utf-8 -*-
'''
@author tianyu,cleartear
'''
#-*- coding: utf-8 -*-
import sys
import random
import pandas as pd
import numpy as np

class Reload(object):
    '''This is the class to reload the movies'''

    def __init__(self, to_select, movie_list):
        self.genre = {}
        self.reload_table = {}
        self.to_select = to_select
        self.recommended = {}
        self.movie_list = movie_list


    def loadfile(self, filename1, filename2):
        self.genre = pd.read_excel(filename1)
        self.reload_table = pd.read_csv(filename2)
        print("Loadfile and to_select data succ.")

    def cosine_similarity(self, a):
        sum_similarity = 0
        n = len(self.movie_list)
        for movieIndex in self.movie_list:
            similarity = sum(self.genre.loc[self.genre['movieId'] == int(movieIndex)].iloc[0, 4:22].values * self.genre.loc[self.genre['movieId'] == a].iloc[0, 4:22].values) / np.sqrt(sum(self.genre.loc[self.genre['movieId'] == int(movieIndex)].iloc[0, 4:22].values) * sum(self.genre.loc[self.genre['movieId'] == a].iloc[0, 4:22].values))
            sum_similarity += similarity
        return sum_similarity / n

    def reload(self):
        similarity = []
        for m in self.to_select:
            similarity.append(Reload.cosine_similarity(self, int(m[0])))
            to_select = list()
        print("Begin to calculate similarity.", file=sys.stderr)
        for tuple1 in self.to_select:
            to_select.append(tuple1[0])
        print("Calculate similarity succ.", file=sys.stderr)
        dict_rec = {'movieId': to_select, 'similarity': similarity}
        rec = pd.DataFrame(dict_rec)
        print("Begin to reload the recommend movies...", file=sys.stderr)
        for mId in rec['movieId']:
            rec.loc[rec['movieId'] == mId, 'score'] = 0 - rec.loc[rec['movieId'] == mId, 'similarity'].values[0] + \
                                                      self.reload_table.loc[
                                                          self.reload_table['movieId'] == int(mId), 'wilson_lb'].values[0]
        rec = rec.sort_values('score', ascending=False)
        # recommended 是最后降序输出的推荐电影列表
        self.recommended = rec.head(5)['movieId'].tolist()
        return self.recommended












#
# genre = pd.read_excel('movie_genres.xls')
# reload_table = pd.read_csv('reload.csv')
# # movie_list是推荐系统的被推荐用户看过的电影
#
# def cosine_similarity(a):
#     sum_similarity = 0
#     n = len(movie_list)
#     for movieIndex in movie_list:
#         similarity = sum(genre.loc[genre['movieId'] == movieIndex].iloc[0, 4:22].values*genre.loc[genre['movieId'] == a].iloc[0, 4:22].values)/np.sqrt(sum(genre.loc[genre['movieId'] == movieIndex].iloc[0, 4:22].values)*sum(genre.loc[genre['movieId'] == a].iloc[0, 4:22].values))
#         sum_similarity += similarity
#     return sum_similarity/n
# # to_select是之前多路召回生成的电影列表
# similarity = []
# for m in to_select:
#     similarity.append(cosine_similarity(m))
# dict_rec = {'movieId':to_select.values, 'similarity':similarity}
# rec = pd.DataFrame(dict_rec)
# for mId in rec['movieId']:
#     rec.loc[rec['movieId'] == mId, 'score'] = 0 - rec.loc[rec['movieId'] == mId, 'similarity'].values[0] + reload_table.loc[reload_table['movieId'] == mId, 'wilson_lb'].values[0]
# rec = rec.sort_values('score', ascending = False)
# # recommanded 是最后降序输出的推荐电影列表
# recommanded = rec.head(5)['movieId'].tolist()
