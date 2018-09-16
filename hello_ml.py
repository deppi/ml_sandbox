from sklearn import datasets
from sklearn.cross_validation import train_test_split # deprecated method that splits your 
													  # test and train data into two sets to run model testws
from sklearn import tree
from sklearn.metrics import accuracy_score

# sample toy dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# splits the data set into half train data and half test data to verify model
# there is some level of random ordering here, so training and results will vary run by run.
X_train, X_test, y_train, y_test = 	train_test_split(X, y, test_size=0.5)

tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)

tree_predictions = tree_classifier.predict(X_test)

# gives how accurate the model. this uses the jaccard similarity method of calculating similarity
# i.e. values in both (ordered) sets (y_test vs tree_predictions) divided by values in all sets
# in other terms, the (# times tree_predictions[i] = y_test[i]) / len (tree_predictions)
print (accuracy_score(y_test, tree_predictions))