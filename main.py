# Generate Poems Using Markov Model
# @author Can Savcı
import time

from config import ROBERT_FROST_URL
from generative_engine.generate_poems import GeneratePoems
from generative_engine.generative_model import GenerativeModel
from reader.reader import Reader
from tokenizer.tokenizer import Tokenizer

if __name__ == '__main__':
    reader = Reader(ROBERT_FROST_URL)
    generative_model = GenerativeModel(reader)
    generate_poems = GeneratePoems(generative_model)

    print("*************  Read Data  *******************")
    start_time = time.time()
    train_x = reader.readRobertFrostTxt()
    start_time = time.time() - start_time
    print(f"Read data completed in {start_time}")

    print("*************  Train Model  ******************")
    start_time = time.time()
    generate_poems.fit(train_x)
    start_time = time.time() - start_time
    print(f"Train completed in {start_time}")

    print("*************  Generate a Poem  **************")
    start_time = time.time()
    generate_poems.generate()
    start_time = time.time() - start_time
    print(f"Poem generated in {start_time}")

    print("********  Prompt the Generated Poem  *********")
    start_time = time.time()
    with open("generated_lines.txt", 'r') as f:
        for line in f:
            print(line)
    start_time = time.time() - start_time
    print(f"Poem generated in {start_time}")

    print("******************************************")
