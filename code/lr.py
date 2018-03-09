from classes.import_data import import_sklearn_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X_train, y_train, X_valid, y_valid, X_test, y_test = import_sklearn_data(binary_data=True)
# 	0.03      Valid accuracy: 0.255452		   Test accuracy: 0.243959

lr = LogisticRegression(C=0.03)
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_valid_pred = lr.predict(X_valid)
y_test_pred = lr.predict(X_test)

print(' Train accuracy: {:.5f}'.format(accuracy_score(y_train, y_train_pred)))
print(' Valid accuracy: {:.5f}'.format(accuracy_score(y_valid, y_valid_pred)))
print(' Test accuracy: {:.5f}'.format(accuracy_score(y_test, y_test_pred)))
