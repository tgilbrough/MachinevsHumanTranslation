from collections import Counter
from math import log

# Basic implementation of Interpolation Model
class InterpolationModel: 

    def __init__(self, n):
        self.n = n

    # Train interpolation model given list of training lines
    def train(self, lines):
        # Create counters for all n-grams of words
        self.ngrams = [Counter() for i in range(self.n)]
        self.number_of_words = 0

        # Count up n-grams in lines
        for line in lines:
            # First pad line with start and stop symbols
            padded_line = list(line)
            for i in range(self.n - 1):
                padded_line.insert(0, '<start>')
            padded_line.append('</stop>')

            # Corner cases so counting at the beginning of the sentence does not break
            for i in range(1, self.n):
                self.ngrams[i - 1][' '.join(['<start>'] * i)] += 1

            # Count n-grams
            for i in range(self.n - 1, len(padded_line)):
                for j in range(self.n):
                    words = padded_line[i - j : i + 1]

                    ngram = ' '.join(words)
                    self.ngrams[j][ngram] += 1

        self.number_of_words = sum([self.ngrams[0][i] for i in self.ngrams[0]])
        self.vocab_size = len(self.ngrams[0].keys()) + 1

    # Given testing lines and lambda and k hyperparameters, return an array of log probabilities
    # for each of the testing lines as a list of floats. k hyperparameter is used for
    # the add k smoothing, while lambda hyperparameters are used as coefficients to probabilty of ngrams,
    # further defined in writeup
    def evaluate(self, lines, lambdas, k):
        assert abs(sum(lambdas) - 1.0) < 1e-8
        assert len(lambdas) == self.n
        
        self.k = k
        self.lambdas = lambdas

        log_probs = []

        # Calculate probabilities for each line
        for line in lines:          
            log_prob = 0.0

            # First pad line with start and stop symbols
            padded_line = list(line)
            for i in range(self.n - 1):
                padded_line.insert(0, '<start>')
            padded_line.append('</stop>')

            # Evaluate each word
            for i in range(self.n - 1, len(padded_line)):
                words = padded_line[i - self.n + 1: i + 1]
                ngram = ' '.join(words)

                # Piece wise function               
                log_prob += log(self.prob(words), 2)

            #print(log_prob, '--', ' '.join(line))  
            # Append line's log probability to list
            log_probs.append(log_prob)

        return log_probs

    # Interpolation of n-gram models 
    def prob(self, words):
        return sum([self.lambdas[i] * self.discounted_pml(words[i:]) for i in range(self.n)])

    # Smoothed probability estimate given words
    def discounted_pml(self, words):
        if len(words) == 1:
            return (self.ngrams[0][words[0]] + self.k) / (self.number_of_words + self.k * self.vocab_size)
        else:
            return (self.ngrams[len(words) - 1][' '.join(words)] + self.k) / (self.ngrams[len(words) - 2][' '.join(words[:-1])] + self.k * self.vocab_size)
