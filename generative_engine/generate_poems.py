class GeneratePoems:

    def __init__(self, generative_model):
        self.generative_model = generative_model

    def fit(self, train_x):
        self.generative_model.fit(train_x)

    def generate(self):
        self.generative_model.generate()

