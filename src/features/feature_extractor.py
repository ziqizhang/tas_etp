'''
Given a particular spending record, build a feature representation for it
'''

'''
- supplier name is encoded using BERT sentence transformer
- spending is scaled by max

a vector is created as a concat of the supplier name embedding (bert sent transformer) and the spending amount
'''
import numpy, pandas
from util import logging_util as lu
from features import text_encoder as te

def features_supplier_n_spending(dataframe, col_supplier_name, col_spending, language_model):
    col_one_list = dataframe[col_spending].tolist()
    max_spending=numpy.max(col_one_list)
    unique_suppliers = sorted(list(dataframe[col_supplier_name].unique()))
    lu.log.info("Embedding texts, total={}".format(len(unique_suppliers)))
    language_model = te.load_model(language_model)
    embeddings = te.embed_sentences(language_model, unique_suppliers)
    mapping = {}
    for i in range(0, len(unique_suppliers)):
        mapping[unique_suppliers[i]] = embeddings[i]

    feature_vectors=[]
    lu.log.info("Building feature representations, total={}".format(len(dataframe)))
    for index, row in dataframe.iterrows():
        supplier =row[col_supplier_name]
        spending = row[col_spending]
        scaled_spending=spending/max_spending
        emb = list(mapping[supplier])
        #emb.append(scaled_spending)
        feature_vectors.append(emb)
    return pandas.DataFrame(feature_vectors)
