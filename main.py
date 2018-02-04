from spam_classifier import SpamClassifier

spamClassifier = SpamClassifier(algorithm='knn')
spamClassifier.train()
spamClassifier.test()