import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def populate_genres_map(csv):
	genres_to_int_map = {}
	# slightly confusing line here, but just grabs unique genres in the Genre1 column.
	for i, genre in enumerate(pd.Series(csv['Genre1']).drop_duplicates()):
		genres_to_int_map[genre] = i
	return genres_to_int_map

# raw csv data, will need a bit of wrangling before we can make predictions
raw_data = pd.read_csv('IMDB.csv', encoding = "ISO-8859-1")
# kill rows that miss features we are using for classification.
raw_data = raw_data.dropna(subset=['Rating', 'Genre1', 'Budget', 'Runtime'])

# map genres to an integer to work with our classifiers.
GENRES_TO_INT_MAP = populate_genres_map(raw_data)

# Answer result set
y = np.array(raw_data['Rating'].tolist())

# X will need a bit more work...
# x1 is genre id, x2 is runtime in minutes, x3 is budget in dollars, 
# but some are in pounds but we assume all are dollars for simplicity :)
X = np.array([ [GENRES_TO_INT_MAP[x1], int(x2.split(' ')[0]), int(x3.replace(' ', '').replace(',', '').strip('$£'))] 
		for x1, x2, x3 
		in zip(raw_data['Genre1'], raw_data['Runtime'], raw_data['Budget']) 
	])

X_train, X_test, y_train, y_test = 	train_test_split(X, y, test_size=0.5)

lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)

tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)

tree_predictions = tree_classifier.predict(X_test)

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

knn_predictions = knn_classifier.predict(X_test)

# gives how accurate the model. this uses the jaccard similarity method of calculating similarity
# i.e. values in both (ordered) sets (y_test vs tree_predictions) divided by values in all sets
# in other terms, the (# times tree_predictions[i] = y_test[i]) / len (tree_predictions)
print ("Decision Tree Accuracy: %{}\nKNN Accuracy: %{}\n"
	.format(accuracy_score(y_test, tree_predictions), accuracy_score(y_test, knn_predictions)))