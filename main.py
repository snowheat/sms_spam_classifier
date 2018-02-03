from spam_classifier import SpamClassifier

spamClassifier = SpamClassifier(algorithm='sgd')
spamClassifier.train()
spamClassifier.test()