import nltk, random

# We will add error analysis now to our application to see how we can better improve our features function so the
# algorithm is better able to decide the correct gender


class GenderApp(object):
    def __init__(self):
        names_sample = nltk.corpus.names
        self.names = [(name.lower(), 'male') for name in names_sample.words('male.txt')] + [(name.lower(), 'female')
                                                                         for name in names_sample.words('female.txt')]
        random.shuffle(self.names)
        self.feature_sets = [(GenderApp.gender_features_part2(name), gender) for name, gender in self.names]
        self.train_set = self.feature_sets[:4000]
        self.test_set = self.feature_sets[4000:]
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)

    def error_analysis(self):
        # We are going to re-train our classifier and introduce a new set for error_analysis
        self.train_set = self.feature_sets[2000:]
        self.dev_set = [(name, gender) for name, gender in self.names[:1000]]
        self.test_set = self.feature_sets[1000:2000]
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)
        self.errors = []
        for name, tag in self.dev_set:
            guess = self.classifier.classify(GenderApp.gender_features_part2(name))
            if guess != tag:
                self.errors.append((name, tag, guess))
        print('Total Errors :', len(self.errors))
        print('Accuracy :', 100 - (len(self.errors)/len(self.dev_set) * 100))
        for name, tag, guess in self.errors:
            print('Name :', name, '- Tag :', tag, '- Guess:', guess)

    @staticmethod
    def gender_features_part2(word):
        name = word.lower()  # let's normalise our input
        features = dict()
        features['first_letter'] = name[0]
        features['last_letter'] = name[1]
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features['count' + letter] = name.count(letter)
            features['has' + letter] = letter in name
        return features

    def check_gender(self, name):
        name = name.lower()
        print('Gender for ' + name + ' : ' + self.classifier.classify(GenderApp.gender_features_part2(name)))

    def check_accuracy_of_the_classifier(self):
        print('Accuracy of classifier is : ', nltk.classify.accuracy(self.classifier, self.test_set) * 100)

    def see_learned_information_of_classifier(self, n=10):
        self.classifier.show_most_informative_features(n)

if __name__ == '__main__':
    app = GenderApp()
    app.error_analysis()