# Try different Classifiers

parent_dir = ''

this_dir = './'
plot_dir = 'learning-curves/'
path_to_plot_dir = parent_dir+this_dir+plot_dir
from pathlib import Path
Path(path_to_plot_dir).mkdir(parents=True, exist_ok=True)

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from evaluate import *
import operator

####################################### Import data #######################################

# Importing the dataset (manually undersampled outliers of class 0)
dataset = pd.read_csv('./data/data_undersampled_manually.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Importing the dataset (undersampled with tomek links)
#dataset = pd.read_csv('Social_Network_Ads_UNDERSAMPLED_TL.csv')
#X = dataset.iloc[:, [0, 1]].values
#y = dataset.iloc[:, 2].values

# Importing the dataset (oversampled class 1)
#dataset = pd.read_csv('Social_Network_Ads_OVERSAMPLED_1.csv')
#X = dataset.iloc[:, [0, 1]].values
#y = dataset.iloc[:, 2].values

# Importing the dataset (normal)
#dataset = pd.read_csv('./data/data.csv')
#X = dataset.iloc[:, [2, 3]].values
#y = dataset.iloc[:, 4].values

################################################################################################################################
print('DATA EXPLORATION')
print('SHAPE')
print(dataset.shape)
print('INFO')
dataset.info()
print('DESCRIPTION')
print(dataset.describe())
n_rows_head = 5
print('FIRST ' + str(n_rows_head) + ' ENTRIES')
print(dataset.head(n_rows_head))
print('\n')
################################################################################################################################

logit = LogisticRegression(class_weight='balanced')
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
ada = AdaBoostClassifier()
rf = RandomForestClassifier(class_weight='balanced')
nb = GaussianNB()
svm = SVC(probability=True, class_weight='balanced')
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

models = {'logistic regression':logit, 'k-nearest neighbors':knn, 'adaboost':ada,
           'random forest':rf, 'naive bayes':nb, 'support vector machines':svm,
           'linear discriminant analysis':lda, 'quadratic discriminant analysis':qda}

metrics_functions = {
'accuracy':get_accuracy,
'f1':get_f1,
'f05':get_f05,
'precision':get_precision,
'recall':get_recall,
'auc roc':get_roc_auc,
'auc pr':get_pr_auc,
'hamming loss':get_hamming_loss
}

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

print('Using Repeated Stratified K-Fold Cross Validation.\n')
rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=36851234)

for name, model in sorted(models.items()):
    for metrics_name, metrics_func in sorted(metrics_functions.items()):
        train_sizes, train_scores, test_scores = calc_learning_curve(cv=rskf, X=X, y=y, model=model, metrics_func=metrics_func)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        # make plot
        plt.title(name+" learning curve")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training "+metrics_name)
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="RSKF CV "+metrics_name)
        plt.legend(loc="best")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.ylabel(metrics_name)
        plt.xlabel("Training Set Size (# examples)")
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.savefig(path_to_plot_dir+name+'_'+metrics_name+'_learning_curve.png', dpi=250)
        plt.clf()
        plt.close()
