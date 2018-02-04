from spam_classifier import SpamClassifier
import timeit
start = timeit.default_timer()

algorithm = input("Choose algorithm: ")

spamClassifier = SpamClassifier(combination='A', algorithm=algorithm, stemming=False, lemma=True, zero=False, stopwords=True, normalization=False)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='B', algorithm=algorithm, stemming=True, lemma=False, zero=True, stopwords=True, normalization=False)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='C', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=False)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='D', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=True, normalization=True)
spamClassifier.train()
spamClassifier.test()

spamClassifier = SpamClassifier(combination='E', algorithm=algorithm, stemming=False, lemma=True, zero=True, stopwords=False, normalization=True)
spamClassifier.train()
spamClassifier.test()

stop = timeit.default_timer()
print("time : ", stop - start)