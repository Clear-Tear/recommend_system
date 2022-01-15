#-*- coding: utf-8 -*-
'''
@author cleartear

@thanks Lockvictor
'''
import sys
import random
import math
import os
from operator import itemgetter
from usercf import UserBasedCF
from itemcf import ItemBasedCF
from reload import Reload
# from NLP import NLP

class RecSystem(object):
    '''the main part of the Recommendation System'''

    def __init__(self, user):
        self.user = user
        self.watched_movies = {}
        self.recommend_movies = {}
        self.final_rec_movies = {}

        print ('We would like to recommend movie to %d' % int(self.user), file=sys.stderr)

    def get_movie_infor(self):
        '''To get movies information of our final result. '''
        print("The user we would like to recommend is %d " % int(self.user), file=sys.stderr)
        print("History movie-id lists of the user as follows: ", file=sys.stderr)
        ls1 = []
        for key in self.watched_movies['1']:
            ls1.append(key)
        ls1.sort()
        print(",".join(str(i) for i in ls1),)

        print("Recommend movie-id lists of the user as follows: ", file=sys.stderr)



    def itemcf_rec_movie(self):
        print("Begin to recommend 10 movies by item-cf.", file=sys.stderr)
        ratingfile = os.path.join('src', 'ratings.csv')
        itemcf = ItemBasedCF()
        itemcf.generate_dataset(ratingfile)
        itemcf.calc_movie_sim()
        item_rec_movies = itemcf.evaluate()
        self.recommend_movies.extend(item_rec_movies)
        print("the item-cf recommend movie is:")
        for movie, rank in item_rec_movies:
            print("recommend No. %s movie." % movie)

    def usercf_rec_movie(self):
        print("Begin to recommend 10 movies by item-cf.", file=sys.stderr)
        ratingfile = os.path.join('src', 'ratings.csv')
        usercf = UserBasedCF()
        self.watched_movies = usercf.generate_dataset(ratingfile)
        usercf.calc_user_sim()
        self.recommend_movies = usercf.evaluate()
        print("The user-cf recommend movie is:")
        for movie, rank in self.recommend_movies:
            print("recommend No. %s movie." % movie)

    def NLP_rec_movie(self):
        # self.recommend_movies.add(NLP.recommend(self.watched_movies['1']))
        NLP_rec_movies = [('7027', 1), ('7757', 1), ('4993', 1), ('3153', 1), ('3771', 1), ('1262', 1), ('1387', 1), ('4327',1)]
        for tuple in NLP_rec_movies:
            self.recommend_movies.append(tuple)
        print("the NLP recommend movie is:")
        for tuple in NLP_rec_movies:
            print ("recommend No.%s movie." % tuple[0])

    def reload_rec_movie(self):
        print("Begin to reload movies...", file=sys.stderr)
        reload_movie = Reload(self.recommend_movies, self.watched_movies['1'])
        reload_movie.loadfile('src/movie_genres.xls', 'src/reload.csv')
        self.final_rec_movies = reload_movie.reload()
        print("the final recommend movie is:")
        for movie in self.final_rec_movies:
            print("recommend No. %s movie." % movie)


if __name__ == '__main__':
    recsystem = RecSystem('1')
    recsystem.usercf_rec_movie()
    recsystem.itemcf_rec_movie()
    recsystem.NLP_rec_movie()
    recsystem.reload_rec_movie()
    # recsystem.get_movie_infor()

