
# coding: utf-8

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from experiments.knn import Knn
from experiments.svm import Svm
from experiments.naive_bayes import NaiveBayes
from experiments.vector_space_model import VectorSpaceModel

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class SpamClassifier:

    def __init__(self, algorithm='decision_tree'):

        # Get pre processed train data
        self.__pre_processed_train_data = self.__get_pre_processed_data("spam_train.txt")
        print("> Train data loaded & pre processed.")

        # Get pre processed test data
        self.__pre_processed_test_data = self.__get_pre_processed_data("spam_test.txt")
        print("> Test data loaded & pre processed.")

        print("> Experiment : " + algorithm)
        self.__experiment = {
            'lexical'           : '0', # insan
            'decision_tree'     : '1', # sigit
            'naive_bayes'       : NaiveBayes(),
            'random_forest'     : '3',
            'svm'               : Svm(), # insan
            'neural_network'    : '5',
            'sgd'               : '6',
            'vector_space_model': VectorSpaceModel(),
            'knn'               : Knn(),
        }[algorithm]

    def __get_pre_processed_data(self, file):

        pre_processed_data = []

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

            # replace words

            # replace 0 with <space>0<space>
            all_text = all_text.replace("0", " 0 ")

            # replace more than 1 spaces with space
            all_text = re.sub(' +', ' ', all_text)

            raw_data = [s.strip() for s in all_text.splitlines()]

            lmtzr = WordNetLemmatizer()

            stopWords = set(stopwords.words('english'))

            for data in raw_data:

                term_array = []
                for term in data.split():

                    # remove stop words
                    if term not in stopWords:

                        # lemmatization
                        term_array.append(lmtzr.lemmatize(term, pos='v'))

                pre_processed_data.append(term_array)

        return pre_processed_data

    def train(self):
        self.__model = self.__experiment.train(self.__pre_processed_train_data)

    def test(self):
        self.__experiment.test(self.__pre_processed_test_data)

    def classify(self):
        pass