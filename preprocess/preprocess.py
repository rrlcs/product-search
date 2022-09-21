from itertools import product
import pyterrier as pt
import pandas as pd
import os


def get_dataframe():
    pdes = pd.read_csv("dataset/product_descriptions.csv",
                       on_bad_lines="skip", engine="python")

    train = pd.read_csv("dataset/train.csv", on_bad_lines="skip",
                        encoding="unicode_escape", engine="python")

    att = pd.read_csv('dataset/attributes.csv')

    return pdes, train, att


def preprocess(train, pdes, att):
    # Convert relevance scores into binary
    train['relevance'] = (train['relevance'] >= 2).astype(int)

    # Add column query id (qid)
    train['qid'] = train.groupby('search_term').ngroup()

    train_with_pdes = (
        pd.merge(train, pdes, on="product_uid", how="left")).drop_duplicates()

    att = att[['product_uid', 'value']]
    att = att.groupby('product_uid').agg(lambda x: x.tolist())
    att['value'] = att['value'].str.join('#')

    train_product_info = (pd.merge(train_with_pdes, att,
                          on='product_uid', how='left')).fillna('')

    train_product_info['product_info'] = ((train_product_info['product_title']) + "$" +
                                          (train_product_info['product_description']) +
                                          "$" + (train_product_info['value'])).replace('\n', '')

    # train_product_info.to_csv('df_preprocessed.csv')
    train_product_info['product_uid'] = train_product_info['product_uid'].apply(
        lambda x: str(x))
    train_product_info['product_info'] = train_product_info['product_info'].apply(
        lambda x: str(x))

    return train_product_info


def train_test_split(train_product_info):
    df_train = train_product_info.sample(frac=0.8, random_state=1)
    # df_train['product_uid'] = df_train['product_uid'].apply(lambda x: str(x))
    # df_train['product_info'] = df_train['product_info'].apply(lambda x: str(x))

    df_test = train_product_info.drop(df_train.index)
    # df_test['product_uid'] = df_test['product_uid'].apply(lambda x: str(x))
    # df_test['product_info'] = df_test['product_info'].apply(lambda x: str(x))

    return df_train, df_test


def init_pyterrier():
    if not pt.started():
        pt.init()


def create_index(df_train_formatted):
    pd_indexer = pt.DFIndexer("./pd_index")
    indexref = pd_indexer.index(
        df_train_formatted["text"], df_train_formatted["docno"])

    return indexref


def print_stats(indexref):
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())


def search_query(indexref, q):
    print("Search Results: ")
    print(type(pt.BatchRetrieve(indexref).search(q)))
    bm25 = pt.BatchRetrieve(indexref).search(q)
    # p = bm25 % 10
    print(bm25)


def get_query(df_train):
    df_query = df_train[['qid', 'search_term']]
    df_query = df_query.rename(columns={'search_term': 'query'})
    query = df_query['query'].to_list()
    query = [q.replace('/', '') for q in query]
    query = [q.replace('\'', '') for q in query]
    df_query['query'] = query

    return df_query


def retrieve_results(df_query, indexref):
    retr = pt.BatchRetrieve(indexref, wmodel="BM25")
    res = retr(df_query)

    return res


def get_qrels_data(df_train):
    qrels = df_train[['qid', 'product_uid', 'relevance']]
    qrels = qrels.assign(iter=0)
    qrels = qrels[['qid', 'iter', 'product_uid', 'relevance']]
    qrels = qrels.rename(columns={'product_uid': 'doc_id'})
    qrels = qrels.astype(str)
    qrels['relevance'] = qrels['relevance'].astype(int)

    return qrels


def evaluate(res, df_query, qrels):
    eval = pt.Utils.evaluate(res, qrels)
    res = res[['qid', 'docno', 'score', 'rank']]

    mrr_eval = pt.Experiment(
        [res],
        df_query,
        qrels,
        eval_metrics=["recip_rank"]
    )

    return eval, mrr_eval


def format_df(df):

    uid = df['product_uid'].to_list()
    text = df['product_info'].to_list()
    df = pd.DataFrame({
        'docno':
        uid,
        'text':
        text
    })
    return df


def get_triples_data(df):
    df = df[['search_term', 'relevance', 'product_info']]
    df_relevant = df[df['relevance'] == 1]
    df_non_relevant = df[df['relevance'] == 0]
    # print("\n", df_relevant.head)
    # print("\n", df_non_relevant.head)
    triples = pd.merge(df_relevant, df_non_relevant,
                       how='inner', on='search_term')
    # print(triples.head)
    triples = triples.rename(columns={
                             'search_term': 'query', 'product_info_x': 'positive', 'product_info_y': 'negative'})
    triples = triples[['query', 'positive', 'negative']]

    triples = triples[['query', 'positive', 'negative']]
    triples['query'] = triples['query'].apply(
        lambda x: str(x))
    triples['positive'] = triples['positive'].apply(
        lambda x: str(x))
    triples['negative'] = triples['negative'].apply(
        lambda x: str(x))
    triples = triples.reset_index(drop=True)
    query = triples['query'].to_list()
    query = [str(q) for q in query]
    positive = triples['positive'].to_list()
    positive = [str(p) for p in positive]
    negative = triples['negative'].to_list()
    negative = [str(n) for n in negative]
    # print(query)
    triples = pd.DataFrame({
        'query':
        query,
        'positive':
        positive,
        'negative':
        negative
    })
    triples = triples.sample(frac=0.001, random_state=1)
    triples.to_csv('triples.tsv', sep='\t')
    # print(triples.columns, triples.dtypes)
    return


# if __name__ == "__main__":
#     pdes, train, att = get_dataframe()

#     train_product_info = preprocess(train, pdes, att)
#     print(train_product_info.columns)
#     triples = get_triples_data(train_product_info)
#     data = pd.read_csv('triples.tsv', sep='\t')
#     data = data[['query', 'positive', 'negative']]
#     print("here: \n", data.columns)
#     data.to_csv('triples.tsv', sep='\t', index=False)
    # exit()

    df_train, df_test = train_test_split(train_product_info)

    df_train_formatted = format_df(df_train)
    df_test_formatted = format_df(df_test)
    data_formatted = format_df(train_product_info)

    init_pyterrier()

    exists = os.path.isdir('pd_index')
    if not exists:
        indexref = create_index(data_formatted)
    else:
        indexref = pt.IndexRef.of("./pd_index/data.properties")
    print_stats(indexref)

    search_query(indexref, q="metal l brackets")

    exit()

    df_query = get_query(df_train)

    res = retrieve_results(df_query, indexref)

    qrels = get_qrels_data(df_train)

    # Evaluate
    eval, mrr_eval = evaluate(res, df_query, qrels)
    print("eval: ", eval)
    print("mrr eval: ", mrr_eval)
