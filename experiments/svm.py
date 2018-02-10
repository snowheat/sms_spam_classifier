from sklearn import svm
import numpy as np


class Svm:
    def __init__(self):
        self.__model = type('test', (object,), {})()
        pass

    def train(self, X_training_data):
        self.__model = svm.SVC(kernel="linear")
        self.__model = self.__model.fit(X_training_data['data_tfidf'], X_training_data['labels'])

        predicted_y = self.__model.predict(X_training_data['data_tfidf'])
        # print(np.mean(predicted_y == X_training_data['labels']))
        pass

    def test(self, X_test_data):
        # test_data = np.dot(np.dot(X_test_data['data_tfidf'], self.__u), np.linalg.inv(np.diag(self.__s)))
        predicted_y = self.__model.predict(X_test_data['data_tfidf'])
        # predicted_y = self.__model.predict(test_data)
        print(np.mean(predicted_y == X_test_data['labels']))

        misclassified = np.where(predicted_y != X_test_data['labels'])
        feat = X_test_data['data_tfidf_features']
        print(feat)
        print(len(predicted_y[misclassified]), misclassified)
        for i in np.nditer(misclassified):
            print(i + 1, predicted_y[i])  # spam, dll
            row = X_test_data['data_tfidf'][i]
            index = np.where(row > 0)
            for b in np.nditer(index):
                print(feat[b], row[b])
        pass
