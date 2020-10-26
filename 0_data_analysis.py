import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

####################################### Import data #######################################
#dataset = pd.read_csv('./data/data.csv')
dataset = pd.read_csv('./data/data_undersampled_manually.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

################################################################################################################################

print('Dropping User ID and encoding Gender...')
dataset = dataset.drop(columns=['User ID'])
dataset['Gender'].replace({"Male": 1, "Female": 0}, inplace=True)

print('\nDATA EXPLORATION')
print('\nSHAPE')
print(dataset.shape)
print('\nINFO')
dataset.info()
print('\nDESCRIPTION')
print(dataset.describe())
n_rows_head = 10
print('\nFIRST ' + str(n_rows_head) + ' ENTRIES')
print(dataset.head(n_rows_head))
print('\nMINIMUM VALUES')
print(dataset.min())
print('\nMAXIMUM VALUES')
print(dataset.max())
print('\nMEAN VALUES')
print(dataset.mean())

################################################################################################################################

#Heatmap
plt.subplots(figsize=(13,10))
heat = dataset.corr()
sns.heatmap(heat)
sns.heatmap(heat, annot = True)
plt.yticks(rotation=0)
plt.savefig('data_heatmap.png', dpi = 250)
plt.clf()

################################################################################################################################

print('\nNUMBER OF ENTRIES PER LABEL')
print(dataset.groupby('Purchased').size())
print('\n')

kde_style = {"color": "darkcyan", "lw": 2, "label": "KDE", "alpha": 0.7}
hist_style = {"histtype": "stepfilled", "linewidth": 3, "color":"darkturquoise", "alpha": 0.25}

# histogram of y values
sns.distplot(dataset.Purchased, kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Purchase data Histogram')
plt.xlabel('Purchase Yes/No')
plt.ylabel('Frequency')
plt.axvline(dataset.Purchased.mean(), color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=2)
plt.savefig('data_histogram.png', dpi = 250)
plt.clf()

################################################################################################################################
plt.close()

# create scatter plot
plt.xlabel('Age')
plt.ylabel('Salary')
#plt.title('')
for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1],
                    c = [matplotlib.colors.ListedColormap(('darkred', 'darkgreen'))(i)], label = j)
plt.legend(('0', '1'),loc='upper right', bbox_to_anchor=(1.05, 1.15))
plt.tight_layout()
plt.savefig('data_scatterplot.png', dpi = 250)
plt.clf()

################################################################################################################################
plt.close()

###################################### OVERSAMPLING
# oversample minority class
#oversample = SMOTE(sampling_strategy=0.75)
#X, y = oversample.fit_resample(X, y)

###################################### UNDERSAMPLING
# undersample with tomek links
#undersample = TomekLinks()
#X, y = undersample.fit_resample(X, y)

###################################### IN BOTH CASES
# make 2-D array of target variables (needed to create new dataframe)
#df_y = [[int(target)] for target in y]

# create scatter plot after resampling
#plt.xlabel('Age')
#plt.ylabel('Salary')
#plt.title('')
#for i, j in enumerate(np.unique(y)):
#        plt.scatter(X[y == j, 0], X[y == j, 1],
#                    c = [matplotlib.colors.ListedColormap(('darkred', 'darkgreen'))(i)], label = j)
#plt.legend(('0', '1'),loc='upper right', bbox_to_anchor=(1.05, 1.15))
#plt.tight_layout()
#plt.savefig('data_scatterplot_after.png', dpi = 250)
#plt.clf()

###################################### OVERSAMPLING
# create new dataframe with oversampled class 1
#new_data_over = np.concatenate((X, df_y), axis=1)
#dataset_over = pd.DataFrame(data=new_data_over, columns=['Age','Salary','Purchased'])
#print('\nNUMBER OF ENTRIES PER LABEL AFTER OVERSAMPLING THE MINORITY CLASS')
#print(dataset_over.groupby('Purchased').size())
#print('\n')
#dataset_over.to_csv('./Social_Network_Ads_OVERSAMPLED_1.csv', index=False)

###################################### UNDERSAMPLING
# create new dataframe with undersampled tomek links classes
#new_data_under = np.concatenate((X, df_y), axis=1)
#dataset_under = pd.DataFrame(data=new_data_under, columns=['Age','Salary','Purchased'])
#print('\nNUMBER OF ENTRIES PER LABEL AFTER UNDERSAMPLING WITH TOMEK LINKS')
#print(dataset_under.groupby('Purchased').size())
#print('\n')
#dataset_under.to_csv('./Social_Network_Ads_UNDERSAMPLED_TL.csv', index=False)

################################################################################################################################
plt.close()

############################################################################# new plots after oversampling/undersampling

#Heatmap
#plt.subplots(figsize=(13,10))
######################################
#heat = dataset_over.corr()
#heat = dataset_under.corr()
######################################
#sns.heatmap(heat)
#sns.heatmap(heat, annot = True)
#plt.yticks(rotation=0)
######################################
#plt.savefig('data_heatmap_OVERSAMPLED_1.png', dpi = 250)
#plt.savefig('data_heatmap_UNDERSAMPLED_TL.png', dpi = 250)
######################################
#plt.clf()

# histogram of y values
######################################
#sns.distplot(dataset_over.Purchased, kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
#sns.distplot(dataset_under.Purchased, kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
######################################
#plt.title('Purchase data Histogram')
#plt.xlabel('Purchase Yes/No')
#plt.ylabel('Frequency')
######################################
#plt.axvline(dataset_over.Purchased.mean(), color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=2)
#plt.axvline(dataset_under.Purchased.mean(), color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=2)
#plt.savefig('data_histogram_OVERSAMPLED_1.png', dpi = 250)
#plt.savefig('data_histogram_UNDERSAMPLED_TL.png', dpi = 250)
######################################
#plt.clf()
