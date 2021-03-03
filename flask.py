import sys
import time

from gensim.models import KeyedVectors
import fasttext
from Dialogue.preprocessor import clean, filter_content

from Dialogue.intention.business import Intention
from Dialogue.retrieval.hnsw_faiss import HNSW,wam


sys.path.append('..')
import Dialogue.config as config
import pandas as pd
import logging
import faiss

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


class Response:
    def __init__(self):

        self.fast = fasttext.load_model(config.ft_path)
        self.hnsw = HNSW(config.w2v_path,
                    config.ef_construction,
                    config.M,
                    config.hnsw_path,
                    config.train_path)

    def get_intention(self, text):
        question = text.strip()
        label, score = self.fast.predict(clean(filter_content(question)))
        print(label[0])
        res = pd.DataFrame()


        if len(question) > 1 and label[0] == '__label__1':

            search = self.hnsw.search(question, 4)
            print(search)
            res = res.append(
                pd.DataFrame(
                    {'query': [question] * 4,
                     'similar': search['custom'],
                     'response': search['assistance']}))
            print(res)
            res.to_csv("test.csv")


if __name__ == '__main__':
    model = Response()
    model.get_intention("我买的数据线充不进去电")
