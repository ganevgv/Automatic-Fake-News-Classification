from classes.import_data import import_sklearn_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


X_train, y_train, X_valid, y_valid, X_test, y_test = import_sklearn_data(binary_data=True)
#   2.0		  Valid accuracy: 0.245327		   Test accuracy: 0.222915

nb = MultinomialNB(alpha=2.0)
nb.fit(X_train, y_train)

y_train_pred = nb.predict(X_train)
y_valid_pred = nb.predict(X_valid)
y_test_pred = nb.predict(X_test)

print(' Train accuracy: {:.5f}'.format(accuracy_score(y_train, y_train_pred)))
print(' Valid accuracy: {:.5f}'.format(accuracy_score(y_valid, y_valid_pred)))
print(' Test accuracy: {:.5f}'.format(accuracy_score(y_test, y_test_pred)))
