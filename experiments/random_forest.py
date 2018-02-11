from sklearn import ensemble
from scipy.sparse.linalg import svds
import numpy as np


class RandomForest:
    def __init__(self):
        self.__model = type('test', (object,), {})()
        pass

    def train(self, X_training_data):
        self.__model = ensemble.RandomForestClassifier(random_state=1)


        self.__model = self.__model.fit(X_training_data['data_tfidf'], X_training_data['labels'])

        """
        self.__u, self.__s, self.__vt = svds(X_training_data['data_tfidf'].transpose(),
                                             X_training_data['svd_max_features'])

        self.__model = self.__model.fit(self.__vt.transpose(), X_training_data['labels'])
        """

        # self.__model = self.__model.fit(X_training_data['data_tfidf'], X_training_data['labels'])

        # predicted_y = self.__model.predict(X_training_data['data_tfidf'])
        # print(np.mean(predicted_y == X_training_data['labels']))
        pass

    def test(self, X_test_data):
        total_data = 0
        total_predicted_true = 0

        total_ham_data = 0
        total_ham_predicted_true = 0

        total_spam_data = 0
        total_spam_predicted_true = 0

        # test_data = np.dot(np.dot(X_test_data['data_tfidf'], self.__u), np.linalg.inv(np.diag(self.__s)))
        predicted_y = self.__model.predict(X_test_data['data_tfidf'])

        # predicted_y = self.__model.predict(test_data)

        for row in range(len(X_test_data['labels'])):

            if X_test_data['labels'][row] == 'spam':
                if X_test_data['labels'][row] == predicted_y[row]:
                    total_spam_predicted_true += 1
                    total_predicted_true += 1
                else:
                    print("[[ SPAM SALAH > ", row, " : ", X_test_data['data'][row])
                total_spam_data += 1
            else:
                if X_test_data['labels'][row] == predicted_y[row]:
                    total_ham_predicted_true += 1
                    total_predicted_true += 1
                else:
                    print("[[ HAM SALAH > ", row, " : ", X_test_data['data'][row])
                total_ham_data += 1

            total_data += 1

        print("[[[ akurasi total ", total_predicted_true / total_data)
        print("[[[ akurasi total ham ", total_ham_predicted_true / total_ham_data)
        print("[[[ akurasi total spam ", total_spam_predicted_true / total_spam_data)
        print("[[[ total data", len(X_test_data['data_tfidf']))
        print("[[[ total ham data ", total_ham_data)
        print("[[[ total spam data ", total_spam_data)
        print(len(X_test_data['labels']), len(predicted_y))
        #predicted_y = self.__model.predict(X_test_data['data_tfidf'])

        print(np.mean(predicted_y == X_test_data['labels']))
        pass
