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


def rating_check(row):
    """Categorical check on significance"""
    if row['diff'] >= 0.2:
        delta = 'Significant_M'
    elif 0 < row['diff'] < 0.2:
        delta = 'Small_M'
    elif -0.2 < row['diff'] <= 0:
        delta = 'Small_F'
    elif row['diff'] <= -0.2:
        delta = 'Significant_F'
    else:
        delta = 'Unresolved'
    return delta

mean_ratings['rating_check'] = mean_ratings.apply(rating_check, axis=1)
# get the top disagreement films...
sorted_by_diff = mean_ratings.sort_values(by='diff', ascending=True)

print(sorted_by_diff.describe())

print(sorted_by_diff[:10])
print(sorted_by_diff[::-1][:10])
# now let's get the movies that are most varied in their ratings:
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.ix[active_titles]
print(rating_std_by_title.sort_values(ascending=False)[:10])




