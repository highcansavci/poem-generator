import numpy as np

from config import GENERATED_LINE_COUNT, GENERATED_POEM_PATH, END_WORD
from tokenizer.tokenizer import Tokenizer


class GenerativeModel:

    def __init__(self, reader):
        self.reader = reader
        self.vector_model = {}
        self.first_order_markov_model = {}
        self.second_order_markov_model = {}

    def fit(self, train_x):
        self.compute(train_x)

    def compute(self, train_x):
        self.encode_line(train_x)

    def generate(self):
        return self.generate_lines()

    def generate_init_word(self):
        found_word = ""
        while not found_word:
            prob = np.random.random()
            cumulative = 0
            for word, pb in self.vector_model.items():
                cumulative += pb
                if prob < cumulative and self.first_order_markov_model.get(word):
                    found_word = word
        return found_word

    def generate_second_word(self, first_word):
        found_word = ""
        while not found_word:
            prob = np.random.random()
            cumulative = 0
            for word, pb in self.first_order_markov_model[first_word].items():
                cumulative += pb
                if prob < cumulative and self.second_order_markov_model[first_word].get(word):
                    found_word = word
        return found_word

    def generate_remaining_words_until_end(self, first_word, second_word, f):
        fst_word = first_word
        sword = second_word
        is_ended = False
        while not is_ended:
            prob = np.random.random()
            cumulative = 0
            for word, pb in self.second_order_markov_model[fst_word][sword].items():
                cumulative += pb
                if prob < cumulative and self.second_order_markov_model.get(sword) and self.second_order_markov_model[sword].get(word):
                    f.write(word + " ")
                    fst_word = sword
                    sword = word
                if word == END_WORD:
                    is_ended = True
                    f.write("\n")

    def generate_lines(self):
        with open(GENERATED_POEM_PATH, 'w') as f:
            for i in range(GENERATED_LINE_COUNT):
                word = self.generate_init_word()
                print(f"{i + 1}. sentence first word is generated")
                f.write(word + " ")
                second_word = self.generate_second_word(word)
                print(f"{i + 1}. sentence second word is generated")
                f.write(second_word + " ")
                self.generate_remaining_words_until_end(word, second_word, f)
                print(f"{i + 1}. sentence remaining words are generated")

    def encode_line(self, data):
        # build vector, first_markov and second_markov models
        for line in data:
            split_string = Tokenizer.tokenize(line)
            if len(split_string) == 0:
                continue
            split_string.append(END_WORD)
            for i in range(len(split_string) - 1):
                word = split_string[i]
                if i == 0:
                    self.vector_model[word] = self.vector_model.get(word, 0) + 1
                elif i == 1 or i == len(split_string) - 2:
                    self.first_order_markov_model[word] = self.first_order_markov_model.get(word, {})
                    next_word = split_string[i + 1]
                    self.first_order_markov_model[word][next_word] = self.first_order_markov_model[word].get(next_word, 0) + 1
                else:
                    self.second_order_markov_model[word] = self.second_order_markov_model.get(word, {})
                    next_word = split_string[i + 1]
                    self.second_order_markov_model[word][next_word] = self.second_order_markov_model[word].get(next_word, {})
                    preceding_word = split_string[i + 2]
                    self.second_order_markov_model[word][next_word][preceding_word] \
                        = self.second_order_markov_model[word][next_word].get(preceding_word, 0) + 1

        # normalization of vector_model
        total_vector_model_size = 0
        for _, value in self.vector_model.items():
            total_vector_model_size += value
        for key, _ in self.vector_model.items():
            self.vector_model[key] /= total_vector_model_size

        # normalization of the first order markov model
        for _, value in self.first_order_markov_model.items():
            total_first_markov_model_size = 0
            for _, value_fo in value.items():
                total_first_markov_model_size += value_fo
            for key_fo, _ in value.items():
                value[key_fo] /= total_first_markov_model_size

        # normalization of the second order markov model
        for _, value in self.second_order_markov_model.items():
            for _, value_fo in value.items():
                total_second_order_markov_model_size = 0
                for _, value_so in value_fo.items():
                    total_second_order_markov_model_size += value_so
                for key_so, _ in value_fo.items():
                    value_fo[key_so] /= total_second_order_markov_model_size



