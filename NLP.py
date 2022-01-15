import pandas as pd
import numpy as np
import re
import ast
from gensim.models import doc2vec



class NLP(object):
    # clean title for the import name

    def __init__(self):
        self.watched_movies = {}
        self.rec_movies = {}


    def mv_title_clean(self,title):
        # search title for (year) pattern
        s = re.search('\(([^)]+)', title)

        # if pattern exists, remove from string
        if s:
            title = title[:s.span()[0]].strip()

        # if ', The' or ', A' is a the end of the string, move it to the front
        # e.g. change "Illusionist, The" to "The Illusionist"
        if title[-5:] == ', The':
            title = 'The ' + title[:-5]
        elif title[-4:] == ', An':
            title = 'An ' + title[:-4]
        elif title[-3:] == ', A':
            title = 'A ' + title[:-3]

        return title

    def recommend(self, watched_movies):

        model1 = doc2vec.Doc2VecTrainables.load('src/corpus_for_rec')  # 加载模型

        self.watched_movies = watched_movies
        #print(user_movies1)
        for i in range(len(self.watched_movies )):
            self.watched_movies[i] = self.mv_title_clean(self.watched_movies[i])

        #print(user_movies1)
        mv_tags_list1 = pd.read_csv('mv_tags_list.csv')#读取对应的文档，可以直接计算

        mv_tags_vectors1 = model1.dv.vectors

        # compute user vector as an average of movie vectors seen by that user


        user_movie_vector1 = np.zeros(shape = mv_tags_vectors1.shape[1])
        for mv in self.watched_movies:
          mv_index = mv_tags_list1[mv_tags_list1["Title"] == mv].index.values[0]
          user_movie_vector1 += mv_tags_vectors1[mv_index]

        user_movie_vector1 /= len(user_movies1)
        print('User\'s movie vector is:\n'+str(user_movie_vector1))

        print('Movie Recommendations:')

        sims1 = model1.dv.most_similar(positive = [user_movie_vector1], topn = 30)
        for i, j in sims1:
          movie_sim = mv_tags_list1.loc[int(i), "Title"].strip()
          if movie_sim not in self.watched_movies:
            self.rec_movies.append(movie_sim)
            print(movie_sim)

        return self.rec_movies
    #rec_movies 存放的就是最后想要的推荐电影，如果你把他改成函数接口的话，return这个就OK的