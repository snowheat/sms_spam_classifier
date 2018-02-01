from spam_classifier import SpamClassifier

spamClassifier = SpamClassifier(algorithm='svm')
spamClassifier.train()
spamClassifier.test()