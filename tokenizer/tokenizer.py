import string


class Tokenizer:
    @classmethod
    def tokenize(cls, line):
        line = line.rstrip().lower()
        if line:
            line = line.translate(str.maketrans('', '', string.punctuation)).split()
        return line


