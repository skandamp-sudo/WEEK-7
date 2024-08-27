#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd  
from pandas import Series, DataFrame 
 
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


iris = pd.read_csv("Iris.csv") 


# In[3]:


iris.head() 


# In[4]:


iris.info()


# # Removing Unneed column

# In[5]:


iris.drop("Id", axis=1, inplace = True) 


# # Some EDA with iris

# In[7]:


fig, ax = plt.subplots(figsize=(10, 7))
iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='red', label='Setosa', ax=ax)
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='green', label='Versicolor', ax=ax)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Virginica', ax=ax)

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_title('Sepal Length vs Sepal Width')
ax.legend()
plt.show()


# In[10]:


sns.FacetGrid(iris, hue='Species', height=5) \
   .map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm') \
   .add_legend()

plt.show()


# In[11]:


fig, ax = plt.subplots(figsize=(10, 7))
iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red', label='Setosa', ax=ax)
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Versicolor', ax=ax)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Virginica', ax=ax)
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_title('Petal Length vs Petal Width')
ax.legend()
plt.show()


# In[12]:


iris.hist(edgecolor='black', linewidth=1.2) 
fig = plt.gcf() 
fig.set_size_inches(12,6) 
plt.show() 


# In[13]:


plt.figure(figsize=(15,10)) 
plt.subplot(2,2,1) 
sns.violinplot(x='Species', y = 'SepalLengthCm', data=iris) 
plt.subplot(2,2,2) 
sns.violinplot(x='Species', y = 'SepalWidthCm', data=iris)  
plt.subplot(2,2,3)
sns.violinplot(x='Species', y = 'PetalLengthCm', data=iris) 
plt.subplot(2,2,4) 
sns.violinplot(x='Species', y = 'PetalWidthCm', data=iris) 


# # Now the given problem is a classification problem..
#  Thus we will be using the classification algorithms
#  to build a model.
#  Classification: Samples belong to two or more classes and we
#  want to learn from already labeled data how to predict the
#  class of unlabeled data
#  Regression: If the desired output consists of one or more
#  continuous variables, then the task is called regression. An
#  example of a regression problem would be the prediction of
#  the length of a salmon as a function of its age and weight.
#  Before we start, we need to clear some ML notations.
#  attributes-->An attribute is a property of an instance that may be used to
#  determine its classification. In the following dataset, the attributes are the
#  petal and sepal length and width. It is also known as Features.
#  Target variable, in the machine learning context is the variable that is or
#  should be the output. Here the target variables are the 3 flower species.

# In[16]:


from sklearn.linear_model import LogisticRegression  # for Logistic Regression Algorithm
from sklearn.model_selection import train_test_split  # to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier
from sklearn import svm  # for Support Vector Machine algorithm
from sklearn import metrics # for checking the model accuracy 
from sklearn.tree import DecisionTreeClassifier # for using DTA


# In[17]:


iris.shape


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
iris_numeric = iris.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(4, 4))
sns.heatmap(iris_numeric.corr(), annot=True, cmap='cubehelix_r')
plt.title('Correlation Heatmap of Iris Numeric Features')
plt.show()


# In[25]:


train, test = train_test_split(iris, test_size=0.3)
print(train.shape) 
print(test.shape)


# In[26]:


train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]  
train_y = train.Species 
 
test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] 
test_y = test.Species 


# In[27]:


train_X.head() 


# In[29]:


test_X.head()


# In[30]:


train_y.head()


# # Support Vector Machine SVM

# In[35]:


model = svm.SVC() # select the svm algorithm 
 
# we train the algorithm with training data and training output 
model.fit(train_X, train_y) 
 
# we pass the testing data to the stored algorithm to predict the outcome 
prediction = model.predict(test_X) 
print('The accuracy of the SVM is: ', metrics.accuracy_score(prediction, test_y)) #
 #we pass the predicted output by the model and the actual output


# # Logistic Regression

# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming train_X, train_y, test_X, test_y are already defined

# Create and train the model
model = LogisticRegression()
model.fit(train_X, train_y)

# Make predictions on the test set
prediction = model.predict(test_X)

# Calculate and print the accuracy
accuracy = accuracy_score(test_y, prediction)
print('The accuracy of Logistic Regression is:', accuracy)


# # Decision Trees

# In[37]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Assuming train_X, train_y, test_X, test_y are already defined

# Create and train the model
model = DecisionTreeClassifier()
model.fit(train_X, train_y)

# Make predictions on the test set
prediction = model.predict(test_X)

# Calculate and print the accuracy
accuracy = accuracy_score(test_y, prediction)
print('The accuracy of Decision Tree is:', accuracy)



# # K nearest Neighbour 

# In[38]:


model = KNeighborsClassifier(n_neighbors=3) 
model.fit(train_X, train_y) 
prediction = model.predict(test_X) 
print('The accuracy of KNN is: ', metrics.accuracy_score(prediction, test_y)) 


# # Let's check the accuracy for various values of n for K-Nearest Neighbors 

# In[51]:


a_index = list(range(1, 11))
accuracy_scores = []
for i in a_index:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    
    accuracy_scores.append(accuracy_score(test_y, prediction))
a = pd.Series(accuracy_scores)
plt.plot(a_index, a, marker='o')
plt.xticks(a_index)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for different values of k')
plt.show()


# In[47]:


train_X.head(5)


# In[48]:


train_y.head(5)


# # Creatig Petals and sepals training data

# In[52]:


petal = iris[['PetalLengthCm','PetalWidthCm','Species']] 
sepal = iris[['SepalLengthCm','SepalWidthCm','Species']]


# # For IRIS petals

# In[53]:


train_p,test_p = train_test_split(petal, test_size=0.3, random_state=0) #petals 
train_x_p = train_p[['PetalWidthCm','PetalLengthCm']] 
train_y_p = train_p.Species 
test_x_p = test_p[['PetalWidthCm','PetalLengthCm']] 
test_y_p = test_p.Species 


# # For IRIS sepals

# In[54]:


train_s,test_s = train_test_split(sepal, test_size=0.3, random_state=0) #sepals 
train_x_s = train_s[['SepalWidthCm','SepalLengthCm']] 
train_y_s = train_s.Species 
test_x_s = test_s[['SepalWidthCm','SepalLengthCm']] 
test_y_s = test_s.Species 


# # SVM Algorithm 

# In[55]:


model = svm.SVC()
model.fit(train_x_p, train_y_p)
prediction_p = model.predict(test_x_p)
print('The accuracy of the SVM using Petals is:', accuracy_score(test_y_p, prediction_p))
model = svm.SVC()
model.fit(train_x_s, train_y_s)
prediction_s = model.predict(test_x_s)
print('The accuracy of the SVM using Sepals is:', accuracy_score(test_y_s, prediction_s))


# # Logistic Regression

# In[56]:


model = LogisticRegression()
model.fit(train_x_p, train_y_p)
prediction_p = model.predict(test_x_p)
print('The accuracy of the Logistic Regression using Petals is:', accuracy_score(test_y_p, prediction_p))
model = LogisticRegression()
model.fit(train_x_s, train_y_s)
prediction_s = model.predict(test_x_s)
print('The accuracy of the Logistic Regression using Sepals is:', accuracy_score(test_y_s, prediction_s))


# # Decision Tree

# In[57]:


model = DecisionTreeClassifier()
model.fit(train_x_p, train_y_p)
prediction_p = model.predict(test_x_p)
print('The accuracy of the Decision Tree using Petals is:', accuracy_score(test_y_p, prediction_p))
model.fit(train_x_s, train_y_s)
prediction_s = model.predict(test_x_s)
print('The accuracy of the Decision Tree using Sepals is:', accuracy_score(test_y_s, prediction_s))


# # K Nearest Neighbor 

# In[58]:


model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_x_p, train_y_p)
prediction_p = model.predict(test_x_p)
print('The accuracy of the KNN using Petals is:', accuracy_score(test_y_p, prediction_p))
model.fit(train_x_s, train_y_s)
prediction_s = model.predict(test_x_s)
print('The accuracy of the KNN using Sepals is:', accuracy_score(test_y_s, prediction_s))


# In[59]:


import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('There are {} samples in the training set and {} samples in the test set'.format(
X_train.shape[0], X_test.shape[0]))


# In[61]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# In[64]:


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
    c=cmap(idx), marker=markers[idx], label=cl)


# In[65]:


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidth=1, marker='o',
                    s=55, label="test set")


# In[67]:


from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=1.0)
svm.fit(X_train_std, y_train)
print('The accuracy of the SVM classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))
print('The accuracy of the SVM classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))


# In[69]:


plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105,15))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


# In[70]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
print('The accuracy of the KNN classifier on training data is {:.2f} out of 1'.format(knn.score(X_train_std, y_train)))
print('The accuracy of the KNN classifier on test data is {:.2f} out of 1'.format(knn.score(X_test_std, y_test)))


# In[72]:


plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn, test_idx=range(105,15))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


# In[6]:


from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Get prediction scores
y_score = model.decision_function(X_test)

# Compute ROC curve
roc_display = RocCurveDisplay.from_predictions(y_test, y_score)

# Show plot
plt.show()


# In[7]:


data = load_breast_cancer()
data


# In[8]:


data.keys()


# In[9]:


data.DESCR


# In[10]:


df = pd.DataFrame(data.data, 
                  columns = data.feature_names)
 # Add the target columns, and fill it with the target data
df["target"] = data.target
 # Show the dataframe
df


# In[11]:


df.info()


# In[12]:


df.isna().sum()


# In[13]:


df["target"].value_counts()


# In[14]:


df["target"].value_counts().plot(kind="bar", color=["peru", "darkmagenta"]);


# In[15]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 7))
ax = sns.heatmap(corr_matrix)


# In[16]:


data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Apples and Oranges CSV

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[2]:


import pandas as pd
df = pd.read_csv('apples_and_oranges.csv')
print(df.head())


# In[7]:


data=pd.DataFrame(data)
data.head()


# In[8]:


import seaborn as sns
sns.scatterplot(x="Weight", y="Size", hue="Class", data=data)


# In[13]:


from sklearn.model_selection import train_test_split
#training_set, test_set = train_test_split(data, test_size=0.2, random_state = 1)
training_set,test_set = train_test_split(data,test_size=0.2,random_state=1)
print("train:",training_set)
print("test:",test_set)


# In[14]:


x_train = training_set.iloc[:,0:2].values  # data
y_train = training_set.iloc[:,2].values  # target
x_test = test_set.iloc[:,0:2].values  # data
y_test = test_set.iloc[:,2].values  # target
print(x_train,y_train)
print(x_test,y_test)


# In[18]:


from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=1,C=1,gamma='auto')
classifier.fit(x_train,y_train)


# In[19]:


y_pred = classifier.predict(x_test)
print(y_pred)


# In[21]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy = float(cm.diagonal().sum())/len(y_test)
print('model accuracy is:',accuracy*100,'%')


# In[22]:


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)


# In[ ]:




