import preprocess.preprocess as prep

from itertools import product
import pyterrier as pt
import pandas as pd
import os

if __name__ == "__main__":

    pdes, train, att = prep.get_dataframe()

    train_product_info = prep.preprocess(train, pdes, att)
    print(train_product_info.columns)
    triples = prep.get_triples_data(train_product_info)
    data = pd.read_csv('triples.tsv', sep='\t')
    data = data[['query', 'positive', 'negative']]
    print("here: \n", data.columns)
    data.to_csv('triples.tsv', sep='\t', index=False)
