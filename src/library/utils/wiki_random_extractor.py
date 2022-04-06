# Extract wikipedia sentence randomly
import os
import glob
import random
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_wikipedia_sentence(N: int,
                               data_dir: str = "../data",
                               L: int = 1500,
                               sentence_length: int = -1,
                               test: bool = False):
    """Extract wikipedia sentence randomly.

    Args:
        N (int): Number of samples to extract.
        data_dir (str, optional): Path to wikipedia data. Defaults to "./data".
        L (int, optional): Minimum number of character of sentence to extract, by default 1500. Defaults to 1500.
        minimum_sentence_length (int, optional): The minimum number of sentences at each sample.
        test (bool, optional): If you haven't download Wikipedia corpus and want to test other codes please set True. Defaults to False.

    Returns:
        List[str]: Randomly extracted sentences.
    """
    if test:
        return ['This is a test sentence to check the code is working.']
    txt_file_paths = glob.glob(os.path.join(data_dir, "wiki/parts/*/wiki_*"))
    random_paths = []
    for _ in range(2 * N):
        random_path = random.choice(txt_file_paths)
        random_paths.append(random_path)
    sentences = []
    cnt = 0
    for random_path in random_paths:
        with open(random_path, "rb") as f:
            lines = f.readlines()
        random.shuffle(lines)
        find_long_line = False
        while not find_long_line and len(lines) > 0:
            line = lines.pop(0)
            line = line.decode('utf-8')
            if len(line) > L and line[0] != "<" and sentence_length == -1:
                sentences.append(line)
                find_long_line = True
                cnt += 1
            elif sentence_length > 0 and len(line) > L and line[0] != "<":
                doc = nlp(line)
                sents = list(doc.sents)
                separated_line = None
                if len(sents) >= sentence_length:
                    separated_line = [
                        str(sents[i]) for i in range(sentence_length)
                    ]
                if separated_line is not None and len(
                        "".join(separated_line)) <= 2 * L:
                    sentences.append(separated_line)
                    find_long_line = True
                    cnt += 1
        if cnt == N:
            break
    return sentences
