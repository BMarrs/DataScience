import json
from collections import defaultdict
from collections import Counter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plot
path = 'C:/pydata-book-master/ch02/usagov_bitly_data2012-03-16-1331923249.txt'

# Open the file with the encoding specified, and ignore errors with typing.
filepath = open(path, encoding="ascii", errors="surrogateescape")
# place the json into a dict
records = [json.loads(line, encoding="utf-8") for line in filepath]
# get the entries for time zones, ignoring cases where timezone doesn't exist.
time_zones = [rec['tz'] for rec in records if 'tz' in rec]

print(time_zones[:10])


# hard way of counting entries
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts


# alternate standard lib way:
def get_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts


counts = get_counts(time_zones)
counts2 = get_counts2(time_zones)
print(counts['America/New_York'])
print(counts2['America/Sao_Paulo'])


# getting the top 10 counts
def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]


print(top_counts(counts))

counts3 = Counter(time_zones)
print("\n\n\n")
print(counts3)

# Now let's do it with pandas
frame = pd.DataFrame(records)
tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])
tz_counts[:10].plot(kind='barh', rot=0)
# show the histogram
# plot.show()

# Parse some information...
print(frame['a'][1])
print(frame['a'][50])
print(frame['a'][51])
# print(frame['a'].value_counts())
results = pd.Series([x.split()[0] for x in frame.a.dropna()])
print(results[:5])
moz_res = pd.Series([x.split()[0] for x in frame.a.dropna() if 'Mozilla' in x])
print(moz_res[:10])
moz_res_cnts = moz_res.value_counts()
print(moz_res_cnts)
print(results.value_counts()[:8])
moz_res_cnts[:20].plot(kind='barh', rot=1)
# plot.show()

# do some checks on whether the user is on Windows or not.
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
print(operating_system[:10])
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts[:10])
indexer = agg_counts.sum(1).argsort()
print(indexer[:10])
count_subset = agg_counts.take(indexer)[-10:]
print(count_subset)
count_subset.plot(kind='barh', stacked=True)
# plot.show()
# normalize the plots to get the percentages of use
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)
plot.show()