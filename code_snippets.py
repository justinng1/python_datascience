# standard libraries.
import numpy as np
import pandas as pd

# useful for plotting on Jupyter notebooks.
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12,8)

# sklearn.
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing

# joining two datasets together. (https://pandas.pydata.org/pandas-docs/stable/merging.html)
train_features = pd.read_csv('../data/train_features.csv')
train_target = pd.read_csv('../data/train_target.csv')
train = pd.merge(train_features, train_target, on='join_var', how='inner') # verify if it joined correctly!

del train_features
del train_target

# preview the data.
train.shape
train.head()
train.columns
dtrain.dtypes

# get counts of the levels of a particular column.
train['var1'].value_counts()

# histogram plots.
plt.hist(df_train['var2'], bins=100)
plt.show()

# get unique values of columns
test_company = test['var1'].unique()
train_company = train['var1'].unique()

# see if the same values are in both datasets.
np.in1d(test_company, train_company)
np.in1d(train_company, test_company)

# group by functions.
gb = train.groupby(['var1'])
comp_mean = gb['target'].mean()
plt.hist(comp_mean, bins=63)
plt.show()

# another group by example.
compjob_count = train.groupby(['var1', 'var2'])['salary'].count()
compjob_count
compjob_count.loc['LEVEL of var1']

# selecting rows of a specific level.
job_ceo = train[train.var2 == 'LEVEL']
job_ceo['target'].value_counts()
plt.hist(job_ceo.target, bins=300)
plt.show()

# displaying variance.
train[train.var2 == 'JUNIOR'].salary.var()

# summary with multiple criteria.
train[(train.var2 == 'CFO') & (train.var1 == 'COMP1')].target.describe()

# dummy variables.
train = pd.get_dummies(train, columns=['var1', 'var2', 'var3', 'var4', 'var5'])
train.columns

# drop columns.
train = train.drop(['var11', 'var12', 'var56', 'var45', 'var98', 'var123'], axis=1)

# setting up to input into sklearn models.
X = train.drop(['target', 'var0'], axis=1)
y = train.target
X_scaled = preprocessing.scale(X) # use this if you want to scale features.

# different regressions with kfoldcv = 5.
clf1 = linear_model.LinearRegression()
scores = cross_val_score(clf1, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf2 = tree.DecisionTreeRegressor()
scores = cross_val_score(clf2, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf3 = ensemble.RandomForestRegressor(n_estimators=100)
scores = cross_val_score(clf3, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf4 = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform')
scores = cross_val_score(clf4, X_scaled, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# fit individual model.
clf1.fit(X,y)

# plotting histograms based on different levels.
major_lvls = train['major'].value_counts() # get levels
for i in major_lvls.index:  # for each level plot a histogram.
    plt.hist(train[train.major == i].target, alpha=0.5, label=i, bins=300)
plt.legend(loc='upper right')
plt.show()

# riser encoding for a specific column.
d = {}

col = 'major'
gb = train.groupby([col])
mean = gb.target.mean()
riser_enc = {}
for l in mean.index:
    riser_enc[l] = mean[l]
d[col] = riser_enc

train.replace(d, inplace=True)
train.head()

