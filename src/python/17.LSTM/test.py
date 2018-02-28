from sklearn.externals import joblib
from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)  
joblib.dump(clf, "train_model.m")
# clf = joblib.load("train_model.m")
# clf.predit(X) # 此处test_X为特征集