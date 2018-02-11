from spam_classifier import SpamClassifier
import timeit
start = timeit.default_timer()

#neural network
#sgd
algorithm = 'sgd'

#algorithm = input("Choose algorithm: ")



"""
spamClassifier = SpamClassifier(combination='A', algorithm=algorithm, stemming=False, lemma=True, zero=False, stopwords=True, normalization=False)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='B', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False)
spamClassifier.train()
spamClassifier.test()


spamClassifier = SpamClassifier(combination='C', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=False)
spamClassifier.train()
spamClassifier.test()
"""

"""
spamClassifier = SpamClassifier(combination='E', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True)
spamClassifier.train()
spamClassifier.test()
"""



spamClassifier = SpamClassifier(combination='B', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='B2', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False,
                                tfidf_max_features=4000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='B3', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False,
                                tfidf_max_features=2000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='B4', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False,
                                tfidf_max_features=1000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='B5', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False,
                                tfidf_max_features=500)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='B6', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False,
                                tfidf_max_features=4000, svd_max_features=2000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='B7', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False,
                                tfidf_max_features=2000, svd_max_features=1000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='B8', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False,
                                tfidf_max_features=1000, svd_max_features=500)
spamClassifier.train()
spamClassifier.test()


# ========================================================================================================================

spamClassifier = SpamClassifier(combination='D', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=True)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='D2', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=True,
                                tfidf_max_features=4000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='D3', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=True,
                                tfidf_max_features=2000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='D4', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=True,
                                tfidf_max_features=1000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='D5', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=True,
                                tfidf_max_features=500)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='D6', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=True,
                                tfidf_max_features=4000, svd_max_features=2000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='D7', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=True,
                                tfidf_max_features=2000, svd_max_features=1000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='D8', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=True,
                                tfidf_max_features=1000, svd_max_features=500)
spamClassifier.train()
spamClassifier.test()

# ========================================================================================================================

spamClassifier = SpamClassifier(combination='F', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='F2', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True,
                                tfidf_max_features=4000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='F3', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True,
                                tfidf_max_features=2000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='F4', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True,
                                tfidf_max_features=1000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='F5', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True,
                                tfidf_max_features=500)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='F6', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True,
                                tfidf_max_features=4000, svd_max_features=2000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='F7', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True,
                                tfidf_max_features=2000, svd_max_features=1000)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='F8', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True,
                                tfidf_max_features=1000, svd_max_features=500)
spamClassifier.train()
spamClassifier.test()




stop = timeit.default_timer()
print("time : ", stop - start)