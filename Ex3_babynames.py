import pandas as pd
import numpy as np
from matplotlib import pyplot as plot

col_names = ['name', 'sex', 'births']
# input the data for 1880
names1880 = pd.read_csv('C:/pydata-book-master/ch02/names/yob1880.txt', names=col_names)

print(names1880[:20])

name_group = names1880.groupby(by='name')['births'].sum()
name_group = name_group.sort_values(ascending=False)
sex_group = names1880.groupby(by='sex')['births'].sum()


print(name_group[:20])
print(sex_group)

# now let's concatenate the entire data set
years = range(1880, 2010)
pieces = []

for year in years:
    path = 'C:/pydata-book-master/ch02/names/yob%s.txt' % year
    frame = pd.read_csv(path, names=col_names)
    frame['year'] = year
    pieces.append(frame)

names = pd.concat(pieces, ignore_index=True)

print(names[:20])

# aggregate the data set
total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
print(total_births.tail())

total_births.plot(title='Total births by sex and year')
plot.show()


# add a function to determine the likelihood of a name
def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group

names = names.groupby(['year', 'sex']).apply(add_prop)
# sanity check for summing to 1
# print(np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1))


parts = []
for year, group in names.groupby(['year', 'sex']):
    parts.append(group.sort_values(by='births', ascending=False)[:1000])
top1000 = pd.concat(parts, ignore_index=True)
print(top1000)
