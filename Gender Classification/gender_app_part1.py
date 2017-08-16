
# To build a small application which will take input a name of a person and tell if the name is female or
# male
# To analyse which algorithm will work


# A very simple function which returns the last letter of the name which is provided as input
def gender_features_part1(word):
    word = str(word).lower()
    return {'last_letter': word[-1:]}

#print(gender_features_part1('Sam'))


# Now to get a sample of names using the nltk built-in module

from nltk.corpus import names as names_sample
import nltk, random



names = [(name, 'male') for name in names_sample.words('male.txt')] + [(name, 'female') for name in
                                                                       names_sample.words('female.txt')]
random.shuffle(names)


# let's print the features for each name




# Let's make a feature set for all the names

feature_sets = [(gender_features_part1(name.lower()), gender) for name, gender in names]


# Making a testing data set and training data set


train_set = feature_sets[3000:]


test_set = feature_sets[:3000]

# Now we use the Naive Buyes classifier and train it using the train set

classifier = nltk.NaiveBayesClassifier.train(train_set)


# Now we test it against names


#print(classifier.classify(gender_features_part1('Samy')))


# To tes the accuracy of the classifier

#print(nltk.classify.accuracy(classifier, test_set)*100)

# nltk.classify.accuracy(ML Classifier, testing data set)



# Seeing what this classifier has learned from the training set

# show_most_informative_features function

# no of features we want to see - default value of 10

print(classifier.show_most_informative_features())