from nltk.corpus import movie_reviews
import nltk
import random


documents = [(list(movie_reviews.words(file_id)), category) for category in movie_reviews.categories() for file_id in
             movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist([w.lower() for w in movie_reviews.words()])
#print(all_words.keys())
word_features = list(all_words.keys())[:2000]

def document_features(document):
    document_words = set(document)
    features = dict()
    for word in word_features:
        features['contains' + word] = word in document_words
    return features


print(document_features(movie_reviews.words('pos/cv957_8737.txt')))

feauture_sets = [(document_features(document), category) for document, category in documents]

train_set = feauture_sets[100:]
test_set = feauture_sets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

print(classifier.show_most_informative_features())
