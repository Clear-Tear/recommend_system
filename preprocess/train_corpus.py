import warnings
import re
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')#忽略warning
absolute_path = 'D:/My code/Python/DataMing/src/ml-20m/'
# movie titles
mv_genres = pd.read_csv(absolute_path+'movies.csv')

# movie tags
mv_tags = pd.read_csv(absolute_path+'genome-scores.csv')

# movie tag descriptions
mv_tags_desc = pd.read_csv(absolute_path+'genome-tags.csv')

# smaller dataset we use(created for our situation)
movielist = pd.read_csv('D:/My code/Python/DataMing/src/links.csv')

def movie_title_clean(title):
    # search title for (year) pattern
    s = re.search('\(([^)]+)', title)

    # if pattern exists, remove from string
    year = 9999
    if s:

        title = title[:s.span()[0]].strip()
        year = s.group(1)

        # check if year is actual year
        if str(year).isdigit():
            year = int(year)
        else:
            year = 9999

            # if ', The' or ', A' is a the end of the string, move it to the front
    # e.g. change "Illusionist, The" to "The Illusionist"
    if title[-5:] == ', The':
        title = 'The ' + title[:-5]
    elif title[-4:] == ', An':
        title = 'An ' + title[:-4]
    elif title[-3:] == ', A':
        title = 'A ' + title[:-3]

    return title, year

# clean title and extract release year
mv_genres['title'] = mv_genres['title'].str.strip()
mv_genres['Title_Year'] = mv_genres['title'].map(movie_title_clean)
mv_genres['Title'] = mv_genres['Title_Year'].apply(lambda x: x[0])
mv_genres['Release Year'] = mv_genres['Title_Year'].apply(lambda x: x[1])

mv_genres_stack = mv_genres[mv_genres['genres'] != '(no genres listed)'].set_index('movieId').genres.str.split('|', expand = True).stack()
mv_genres_explode = pd.get_dummies(mv_genres_stack, prefix = 'g').groupby(level = 0).sum().reset_index()
del mv_genres_stack

# genre vector (binary string)
mv_genres_explode['genre_vector'] = mv_genres_explode.iloc[:,1:].values.tolist()

# append genre vector
mv_genres = mv_genres.merge(mv_genres_explode[['movieId','genre_vector']], on = 'movieId', how = 'left')
# check out genre dataframe with genre vector (which you can realise in the jupyter notebook, but not other environments)
#mv_genres.head()

# join dataframes to get tag description and movie title name all in one table
mv_tags_denorm = mv_tags.merge(mv_tags_desc, on = 'tagId').merge(mv_genres, on = 'movieId')

# for each movie, compute the relevance rank of tags so we can eventually rank order tags for each movie
mv_tags_denorm['relevance_rank'] = mv_tags_denorm.groupby("movieId")["relevance"].rank(method = "first", ascending = False).astype('int64')

# check out an example of top tags for a movie
mv_tags_denorm[mv_tags_denorm.Title == 'Remember the Titans'][['movieId','Title','tag','relevance','relevance_rank']].sort_values(by = 'relevance', ascending = False)
#.head(10) you can check this in jupyter notebook by adding the .head(10) to the above line

# compute median relevance score for each relevance rank
mv_tags_rank_agg = mv_tags_denorm.groupby('relevance_rank')['relevance'].median().reset_index(name = 'relevance_median').head(100)

# compute percent change of median relevance score as we go down in rank
mv_tags_rank_agg['relevance_median_pct_chg'] = mv_tags_rank_agg['relevance_median'].pct_change()

# flatten tags table to get a list of top 100 tags for each movie
mv_tags_list = mv_tags_denorm[mv_tags_denorm.relevance_rank <= 100].groupby(['movieId','Title'])['tag'].apply(lambda x: ','.join(x)).reset_index()
mv_tags_list['tag_list'] = mv_tags_list.tag.map(lambda x: x.split(','))

# merge original dataset to the dataset we need to avoid non-existing movies in the smaller dataset
temp = movielist.merge(mv_tags_list,on = 'movieId')
temp.drop(['imdbId','tmdbId'],axis=1,inplace = True)
mv_tags_list = temp
# 调整窗宽，如果不显示也可以不要这一句
pd.set_option('display.max_colwidth', -1)

# compute Jaccard Index to get most similar movies to target movie

pd.reset_option('display.max_colwidth')

# corpus of movie tags
mv_tags_corpus = mv_tags_list.tag.values


stop_words = stopwords.words('english')


# tokenize document and clean
def word_tokenize_clean(doc):
    # split into lower case word tokens
    tokens = word_tokenize(doc.lower())

    # remove tokens that are not alphabetic (including punctuation) and not a stop word
    tokens = [word for word in tokens if word.isalpha() and not word in stop_words]

    return tokens

# preprocess corpus of movie tags before feeding it into Doc2Vec model
mv_tags_doc = [TaggedDocument(words=word_tokenize_clean(D), tags=[str(i)]) for i, D in enumerate(mv_tags_corpus)]

# instantiate Doc2Vec model

max_epochs = 50
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=0)  # paragraph vector distributed bag-of-words (PV-DBOW)

model.build_vocab(mv_tags_doc)

# train Doc2Vec model
# stochastic (random initialization), so each run will be different unless you specify seed

print('Epoch', end = ': ')
for epoch in range(max_epochs):
  print(epoch, end = ' ')
  model.train(mv_tags_doc,
              total_examples=model.corpus_count,
              epochs=model.epochs)
  # decrease the learning rate
  model.alpha -= 0.0002
  # fix the learning rate, no decay
  model.min_alpha = model.alpha

  # mv_tags_vectors = model.docvecs.vectors_docs


model.save('src/corpus_for_rec')
mv_tags_list.to_csv("src/mv_tags_list.csv",index = False)
