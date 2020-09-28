# Try different Classifiers

parent_dir = ''

this_dir = './'
plot_dir = 'plots/'
path_to_plot_dir = parent_dir+this_dir+plot_dir
from pathlib import Path
Path(path_to_plot_dir).mkdir(parents=True, exist_ok=True)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import plot_metrics
from evaluate import *
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import sys

model_names = ('logit', 'knn', 'adaboost', 'rf', 'naive_bayes', 'svm', 'LDA', 'QDA')

try:
    model_name = sys.argv[1]
except:
    print("You must specify a classifier method, using naive bayes as default one.\n")
    model_name = 'naive_bayes'

# Importing the dataset
dataset = pd.read_csv('./data/data.csv')
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
n_rows_head = 5
print('\nFIRST ' + str(n_rows_head) + ' ENTRIES')
print(dataset.head(n_rows_head))
print('\nMINIMUM VALUES')
print(dataset.min())
print('\nMAXIMUM VALUES')
print(dataset.max())
print('\nMEAN VALUES')
print(dataset.mean())
################################################################################################################################

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting model to the Training set
if model_name == model_names[0]:
    print('Using Logistic Regression.')
    classifier = LogisticRegression(class_weight='balanced')
elif model_name == model_names[1]:
    print('Using K-Nearest Neighbors Classifier.')
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
elif model_name == model_names[2]:
    print('Using AdaBoost Classifier.')
    classifier = AdaBoostClassifier()
elif model_name == model_names[3]:
    print('Using Random Forest Classifier.')
    classifier = RandomForestClassifier(class_weight='balanced')
elif model_name == model_names[4]:
    print('Using Naive Bayes Classifier.')
    classifier = GaussianNB()
elif model_name == model_names[5]:
    print('Using Support Vector Machines Classifier.')
    classifier = SVC(probability=True, class_weight='balanced')
elif model_name == model_names[6]:
    print('Using Linear Discriminant Analysis.')
    classifier = LinearDiscriminantAnalysis()
elif model_name == model_names[7]:
    print('Using Quadratic Discriminant Analysis.')
    classifier = QuadraticDiscriminantAnalysis()
else:
    print('Unknown option for classifier, using naive bayes as default one!')
    model_name = 'naive_bayes'
    classifier = GaussianNB()

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# just to calculate also training Confusion Matrix
y_pred_train = classifier.predict(X_train)

# define sets
set_label_list = ['Training', 'Testing']

# Plot the Confusion Matrix
classes = [0,1]
title_cm = 'Ads Purchase Confusion Matrix ('+model_name.upper()+')'
y_lists = [[y_train, y_pred_train],[y_test, y_pred]]
initial_path = path_to_plot_dir+model_name+'-'+'confusion-matrix-'

for ((true_y, pred_y), label) in zip(y_lists, set_label_list):
    plot_metrics.plot_confusion_matrix(y_true=true_y, y_pred=pred_y, classes=classes,
                          title=title_cm+' ('+label+' set)',
                          path=initial_path+label.upper()+'.png')

# Predict probabilities #######################################################
y_score = classifier.predict_proba(X_test)
probabilities_1 = y_score[:,1] # take only positive probabilities
# Plot also ROC curve
title_roc = 'Ads Purchase Receiver Operating Characteristic curve ('+model_name.upper()+')'
initial_path_roc = path_to_plot_dir+model_name+'-'+'roc-curve-'
path_roc = initial_path_roc+set_label_list[1].upper()+'.png'
# Call function for plotting ROC curve
plot_metrics.plot_roc(y_test, probabilities_1, title_roc, path_roc)
# Plot also PR curve
title_pr = 'Ads Purchase Precision Recall curve ('+model_name.upper()+')'
initial_path_pr = path_to_plot_dir+model_name+'-'+'pr-curve-'
path_pr = initial_path_pr+set_label_list[1].upper()+'.png'
# Call function for plotting PR curve
plot_metrics.plot_pr(y_test, probabilities_1, title_pr, path_pr)

# Visualising the results for both training and testing set
set_list = [[X_train, y_train], [X_test, y_test]]

for ((X_set, y_set), label) in zip(set_list, set_label_list) :
    plt.figure(figsize=(10,10))
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = [ListedColormap(('darkred', 'darkgreen'))(i)], label = j)
    plt.title(model_name.upper() + ' (' + label + ' set)')
    plt.xlabel('Standardized Age')
    plt.ylabel('Standardized Estimated Salary')
    plt.legend()
    plt.savefig(path_to_plot_dir+model_name+'-results-' + label.upper() + '.png', dpi=250)
    plt.clf()
    plt.close()


print(model_name.upper() + ' PERFORMANCE:')
print_all_metrics(y_pred, probabilities_1, y_test)
