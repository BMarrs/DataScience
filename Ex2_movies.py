import pandas as pd
from matplotlib import pyplot as plot
# set the field names

datapath = 'C:/pydata-book-master/ch02/movielens/'

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(datapath + 'users.dat', sep='::', header=None, names=unames, engine='python')
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(datapath + 'ratings.dat', sep='::', header=None, names=rnames, engine='python')
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(datapath + 'movies.dat', sep='::', header=None, names=mnames, engine='python')

# Merge the data sets together based on keys
data = pd.merge(pd.merge(ratings, users), movies)
# Aggregate the data with a pivot
mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
mean_ratings_age = data.pivot_table('rating', index='title', columns='age', aggfunc='mean')
# Let's filter out data that didn't receive at least 250 ratings.
ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title > 250]
mean_ratings = mean_ratings.ix[active_titles]
# See what the top ratings are among women
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
# See what the difference in mean ratings between men and women are by adding in a new field.
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']

