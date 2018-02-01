import operator


class Svm:

    def __init__(self):
        pass

    def train(self, pre_processed_train_data):
        a = {}



        for row in pre_processed_train_data:
            for term in row:
                if term in a:
                    a[term] += 1
                else:
                    a[term] = 1

        sorted_a = sorted(a.items(), key=operator.itemgetter(1))

        print(len(sorted_a))

        b = []

        for term, value in sorted_a :
            if value > 5:
                b.append(term)

        print(len(b))

        pass

    def test(self, pre_processed_test_data):
        pass