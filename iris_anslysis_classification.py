#This code should be done in google colaboratory 

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()
iris_data = sns.load_dataset('iris')
#In case of any other complier we must read csv file

iris_data.head()

iris_data.tail()

iris_data.describe()

sns.pairplot(iris_data, hue="species")
plt.show()

data_values = iris_data.values
X = data_values[:,0:4]
Y = data_values[:,4]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)

accuracy = knn.score(X_test, Y_test)
print("Accuracy:", accuracy*100)

input_data = (5.8,2.7,5.1,1.9)
#changing the input_data into numpyarray
input_data_np = np.asarray(input_data)
#reshaping the array as we are predicting for one instance 
input_data_reshape = input_data_np.reshape(1,-1)
prediction = knn.predict(input_data_reshape)
print(prediction)
