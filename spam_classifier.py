
# coding: utf-8

import re, pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from experiments.knn import Knn
from experiments.sgd import SGD
from experiments.rule_based import RuleBased
from experiments.svm import Svm
from experiments.neural_net import NeuralNet
from experiments.random_forest import RandomForest
from experiments.decision_tree import DecisionTree
from experiments.naive_bayes import NaiveBayes
from experiments.vector_space_model import VectorSpaceModel

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class SpamClassifier:

    def __init__(self, combination, algorithm='decision_tree', stemming=False, lemma=True, zero=True, stopwords=True, normalization=True, tfidf_max_features=None, svd_max_features=1000):

        print("=============================", algorithm, " - Combination ", combination, "=============================")

        # https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
        # https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
        # http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py
        # https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models
        # https://stackoverflow.com/questions/40731271/test-and-train-dataset-has-different-number-of-features

        # Options
        self.__options = {
            'stemming': stemming,
            'lemma' : lemma,
            'zero' : zero,
            'stopwords' : stopwords,
            'normalization' : normalization,
            'tfidf_max_features' : tfidf_max_features,
            'svd_max_features' : svd_max_features,
        }

        if self.__options['tfidf_max_features'] is not None:
            if self.__options['svd_max_features'] >= self.__options['tfidf_max_features']:
                self.__options['svd_max_features'] = self.__options['tfidf_max_features'] - int(0.25*self.__options['tfidf_max_features'])

        print("lihat kemari ", self.__options['svd_max_features'])
        # Vectorizer
        self.__vectorizer = CountVectorizer(max_features=self.__options['tfidf_max_features'])

        # TF-IDF Transformer
        self.__tfidf_transformer = TfidfTransformer()

        # Get pre processed train data
        self.__pre_processed_train_data = self.__get_pre_processed_data("spam_train.txt")

        # Get document-term matrix (X) train data
        self.__X_train_data = self.__get_X_train_data()

        print("> Train data loaded & pre processed.")


        # Get pre processed test data
        self.__pre_processed_test_data = self.__get_pre_processed_data("spam_test.txt")

        # Get document-term matrix (X) test data
        self.__X_test_data = self.__get_X_test_data()

        print("> Test data loaded & pre processed.")

        print("> Experiment : " + algorithm)
        self.__experiment = {
            'rule_based'        : RuleBased(), # insan
            'decision_tree'     : DecisionTree(), # sigit
            'naive_bayes'       : NaiveBayes(),
            'random_forest'     : RandomForest(),
            'svm'               : Svm(), # insan
            'neural_network'    : NeuralNet(),
            'vector_space_model': VectorSpaceModel(),
            'knn'               : Knn(),
            'sgd'               : SGD(),
        }[algorithm]

    def __get_pre_processed_data(self, file):

        # load data
        filepath = file

        # build data array
        with open(filepath) as fp:

            all_text = fp.read().lower()

            # remove all non alphanumeric character excluding newline, dot, tab and @
            pattern = re.compile('[^a-zA-Z0-9\\n\'\.\\t\@]')
            all_text = pattern.sub(' ', all_text)

            # replace tab and dot with single space
            all_text = all_text.replace(".", " ")
            all_text = all_text.replace("\t", " ")

            # word normalization
            if self.__options['normalization']:
                all_text = all_text.replace("'d", " would")
                all_text = all_text.replace("'m", " am")
                all_text = all_text.replace("'ve", " have")
                all_text = all_text.replace("'ll", " will")
                all_text = all_text.replace("'s", " is")
                all_text = all_text.replace("'re", " are")
                all_text = all_text.replace("n't", " not")

            # replace repeatable character : 0

            # replace 0 with <space>0<space>
            if self.__options['zero']:
                all_text = all_text.replace("0", " 0 ")

            # replace more than 1 spaces with space
            all_text = re.sub(' +', ' ', all_text)

            raw_data = [s.strip() for s in all_text.splitlines()]

            lmtzr = WordNetLemmatizer()
            ps = PorterStemmer()
            stopWords = set(stopwords.words('english'))

            pre_processed_data = {'labels' : [], 'data' : [], 'spam_words' : []}

            for row in raw_data:
                row_split = row.split()

                # get label
                pre_processed_data['labels'].append(row_split[0])

                term_array = []
                for term in row_split[1:]:

                    # stemming
                    if self.__options['stemming']:
                        term = ps.stem(term)
                    else:
                        term = term

                    # lemmatization
                    if self.__options['lemma']:
                        lemmatized_term = lmtzr.lemmatize(term, pos='v')
                    else:
                        lemmatized_term = term



                    # remove stop words
                    if self.__options['stopwords']:
                        if term not in stopWords:

                            term_array.append(lemmatized_term)

                            if row_split[0] == "spam":
                                if lemmatized_term not in pre_processed_data['spam_words']:
                                    pre_processed_data['spam_words'].append(lemmatized_term)
                    else:
                        term_array.append(lemmatized_term)

                        if row_split[0] == "spam":
                            if lemmatized_term not in pre_processed_data['spam_words']:
                                pre_processed_data['spam_words'].append(lemmatized_term)

                # get data
                pre_processed_data['data'].append(" ".join(term_array))

        return pre_processed_data

    # get document-term matrix from training data
    def __get_X_train_data(self):

        X = {'labels': [], 'data': [], 'data_tfidf': [], 'data_tfidf_features': [], 'shape': [], 'vocabulary_': [], 'spam_words': [], 'svd_max_features': []}

        X['labels'] = self.__pre_processed_train_data['labels'];
        X['svd_max_features'] = self.__options['svd_max_features']

        X['spam_words'] = self.__pre_processed_train_data['spam_words'];

        X['data'] = self.__pre_processed_train_data['data']


        # encode to document term matrix - frequency
        X_frequency = self.__vectorizer.fit_transform(self.__pre_processed_train_data['data'])

        # encode to document term matrix - tfidf
        X_tfidf = self.__tfidf_transformer.fit_transform(X_frequency)

        X['data_tfidf'] = X_tfidf.toarray()
        X['data_tfidf_features'] = self.__vectorizer.get_feature_names()

        X['shape'] = X_frequency.shape
        X['vocabulary_'] = self.__vectorizer.vocabulary_

        """
        print(X['labels'])
        print(len(X['labels']))
        print(X['shape'])
        print(X['vocabulary_'])
        print(X_tfidf) # tfidf yg compact ternormalisasi
        print(X['data_tfidf_features']) # tfidf bentuk matrix penuh
        print("==================================================")
        """

        return X

        # get document-term matrix from training data

    def __get_X_test_data(self):

        X = {'labels': [], 'data': [], 'data_tfidf': [], 'data_tfidf_features': [], 'shape': [], 'vocabulary_': [], 'svd_max_features': []}

        X['labels'] = self.__pre_processed_test_data['labels'];
        X['svd_max_features'] = self.__options['svd_max_features']

        X['data'] = self.__pre_processed_test_data['data']

        # encode to document term matrix - frequency
        X_frequency = self.__vectorizer.transform(self.__pre_processed_test_data['data'])

        # encode to document term matrix - tfidf
        X_tfidf = self.__tfidf_transformer.transform(X_frequency)

        X['data_tfidf'] = X_tfidf.toarray()
        X['data_tfidf_features'] = self.__vectorizer.get_feature_names()

        X['shape'] = X_frequency.shape
        X['vocabulary_'] = self.__vectorizer.vocabulary_

        """
        print(X['labels'])
        print(len(X['labels']))
        print(X['shape'])
        print(X['vocabulary_'])
        print(X_tfidf) # tfidf yg compact ternormalisasi
        print(X['data_tfidf_features']) # tfidf bentuk matrix penuh
        """

        return X

    def train(self):
        self.__experiment.train(self.__X_train_data)

    def test(self):
        self.__experiment.test(self.__X_test_data)

    def classify(self):
        pass