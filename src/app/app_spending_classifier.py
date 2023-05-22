'''
Takes the spending records in a format like this:https://docs.google.com/spreadsheets/d/16LGr599JSDFzRVPf2oaE82JEp8kT_g0cRS2v0YcEDFM/edit#gid=594154261

Uses only two features: supplier name, spending

And runs n-fold validation
'''
import sys, pandas,csv, os
from util import logging_util as lu
from features import feature_extractor
from classifier import textclassifier as tc
from data import supplier_similarity as ss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
'''
This is a utility method to deal with multi-labels for a spending item

If a spending item has multiple labels, in the input data file we expect a format like: cat1|cat2|cat3
Ideally these should be mapped manually. but in case this is not, we 'flatten' these records by
- creating X duplicate records
- divide the total spending equally
'''
def flatten(dataframe, col_index_label, col_index_spending):
    index_mapping = {}
    count=0
    rows=[]
    for index, row in dataframe.iterrows():
        label = row.iloc[col_index_label]
        if type(label) is not str:
            continue
        labels = set()
        if '|' in label:
            lbs = label.split("|")
            for l in lbs:
                l = l.strip()
                if len(l)>1:
                    labels.add(l)
        else:
            labels.add(label)

        spending = row.iloc[col_index_spending]
        avg = spending/len(labels)
        for l in labels:
            newrow = list(row.copy(deep=True))
            newrow[col_index_spending] = avg
            newrow[col_index_label] = l
            rows.append(newrow)
            index_mapping[count] = index
            count += 1

    return pandas.DataFrame(rows), index_mapping

'''
The input csv has many misspelled supplier names, this method if called, will load a GS csv file containing
supplier name mappings, and use the mapping to 'normalise' the data in the input csv

the mapping csv file looks like this https://docs.google.com/spreadsheets/d/1LDaQw4nv8g4eKsrwSKtP1sf8wZaoIxKcL02VN1c8oo8/edit?usp=share_link
'''
def map_supplier_names(csv_mapping_file, col_supplier1, col_supplier2, col_label, dataframe, col_supplier):
    mappings =ss.supplier_mapping_loader(csv_mapping_file, col_supplier1, col_supplier2, col_label)
    for index, row in dataframe.iterrows():
        sup = row[col_supplier]
        if sup in mappings.keys():
            standardised = mappings[sup][0]
            df.at[index, col_supplier] = standardised
    return dataframe

if __name__ == "__main__":
    '''
    /home/zz/Data/tas_etp/initial_coding_september-last-sheet_converted.csv
all-MiniLM-L12-v2
TopCategory
Supplier
Total
/home/zz/Data/tas_etp/classification
/home/zz/Data/tas_etp/supplier_name_similarity.csv
    
    '''
    data_file=sys.argv[1]
    transformer_model=sys.argv[2]
    col_label = sys.argv[3]
    col_supplier=sys.argv[4]
    col_spending=sys.argv[5]
    outfolder=sys.argv[6]

    df = pandas.read_csv(data_file, header=0, sep=',', quoting=csv.QUOTE_ALL, quotechar='"',encoding="utf-8")

    if len(sys.argv)>7:
        df=map_supplier_names(sys.argv[7], 'Supplier 1','Supplier 2','Yes/No (1 for yes, 0 for no)', df, col_supplier)

    label_vector = df.loc[:,col_label]
    lu.log.info("Encoding features...")
    feature_df=feature_extractor.features_supplier_n_spending(df, col_supplier, col_spending, transformer_model)

    feature_df.insert(loc=0, column=None, value=label_vector)
    lu.log.info("Flattening feature matrix...")
    feature_df, rowidx_to_originalrowidx= flatten(feature_df, 0, len(feature_df.columns)-1)

    lu.log.info("Classifier training...")
    tc.nfold_validation(feature_df,
                        rowidx_to_originalrowidx,
                        outfolder, alg="svm",nfold=10)

    #party_spend_category_percent(in_file, 'Party', 'TopCategory','Total',out_folder)
    #party_spend_category_percent(in_file, 'Party', 'SubCategory', 'Total', out_folder)

    #category_spend_party_percent(in_file, 'Party', 'TopCategory', 'Total', out_folder)
    #category_spend_party_percent(in_file, 'Party', 'SubCategory', 'Total', out_folder)
    #party_spending_per_supplier(in_file, 'Party', 'Supplier', 'Total', out_folder)