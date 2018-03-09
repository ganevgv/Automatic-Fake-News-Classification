from classes.import_data import import_sklearn_data
from sklearn.metrics import accuracy_score
import numpy as np

# np.random.seed(18)

X_train, y_train, X_valid, y_valid, X_test, y_test = import_sklearn_data(binary_data=True)

labels = list(set(y_train))

y_train_pred = np.random.choice(labels, size=len(y_train))
y_valid_pred = np.random.choice(labels, size=len(y_valid))
y_test_pred = np.random.choice(labels, size=len(y_test))

print(' Train accuracy: {:.5f}'.format(accuracy_score(y_train, y_train_pred)))
print(' Valid accuracy: {:.5f}'.format(accuracy_score(y_valid, y_valid_pred)))
print(' Test accuracy: {:.5f}'.format(accuracy_score(y_test, y_test_pred)))
