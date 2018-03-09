from classes.import_data import import_sklearn_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


X_train, y_train, X_valid, y_valid, X_test, y_test = import_sklearn_data(binary_data=True)
#   'linear'  C: 0.4		 			            Valid accuracy: 0.253894	 Test accuracy: 0.232268
#   'rbf'     C: 2.0   gamma: 0.05                  Valid accuracy: 0.267134     Test accuracy: 0.254871
#   'poly'    C: 1.0   gamma: 0.1    degree: 2      Valid accuracy: 0.267134     Test accuracy: 0.252533


svm = SVC(kernel='linear', C=0.4)
svm.fit(X_train, y_train)

y_train_pred = svm.predict(X_train)
y_valid_pred = svm.predict(X_valid)
y_test_pred = svm.predict(X_test)

print(' Train accuracy: {:.5f}'.format(accuracy_score(y_train, y_train_pred)))
print(' Valid accuracy: {:.5f}'.format(accuracy_score(y_valid, y_valid_pred)))
print(' Test accuracy: {:.5f}'.format(accuracy_score(y_test, y_test_pred)))
