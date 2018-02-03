from spam_classifier import SpamClassifier

spamClassifier = SpamClassifier(algorithm='rule_based')
spamClassifier.train()
spamClassifier.test()