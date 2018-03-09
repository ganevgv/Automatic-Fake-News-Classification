from classes.import_data import import_majority_data
from classes.majority import majority
from sklearn.metrics import accuracy_score


X_train, y_train, X_valid, y_valid, X_test, y_test = import_majority_data(binary_data=False)
#   'false'   Valid accuracy: 0.253894		   Test accuracy: 0.231489

mj = majority()
mj.fit(X_train)

y_train_pred = mj.predict(X_train)
y_valid_pred = mj.predict(X_valid)
y_test_pred = mj.predict(X_test)

print(' Train accuracy: {:.5f}'.format(accuracy_score(y_train, y_train_pred)))
print(' Valid accuracy: {:.5f}'.format(accuracy_score(y_valid, y_valid_pred)))
print(' Test accuracy: {:.5f}'.format(accuracy_score(y_test, y_test_pred)))

# from sklearn.metrics import precision_recall_fscore_support
# print(' Train f: {}'.format(precision_recall_fscore_support(y_train, y_train_pred, average='micro')))
# print(' Valid f: {}'.format(precision_recall_fscore_support(y_valid, y_valid_pred, average='micro')))
# print(' Test f: {}'.format(precision_recall_fscore_support(y_test, y_test_pred, average='micro')))
