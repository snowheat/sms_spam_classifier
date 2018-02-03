from spam_classifier import SpamClassifier

spamClassifier = SpamClassifier(algorithm='decision_tree')
spamClassifier.train()
spamClassifier.test()