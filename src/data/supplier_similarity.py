'''
loads convereted data created by coding_converter.py (e.g. initial_coding_september-last-sheet-convered.csv)
find unique set of suppliers, compute pair wide similarity using levenstein and output a csv
'''
import pandas as pd
from Levenshtein import ratio
import csv,sys

'''
given the supplier name mappings in a csv file create clusters of the same supplier and a mapping.
An example of such csv file is https://docs.google.com/spreadsheets/d/1LDaQw4nv8g4eKsrwSKtP1sf8wZaoIxKcL02VN1c8oo8/edit?usp=share_link

IMPORTANT: the supplier name pairs in the csv file must be ranked by similarity as the method only reads up to the last line that
has an annotation. If data is not ordered in such a way some will be missed and the mapping may be incomplete
'''
def supplier_mapping_loader(in_file, col_supplier1, col_supplier2, col_label):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    name_clusters={}
    for index, row in df.iterrows():
        label = row[col_label]

        if label==1:
            supplier1 = row[col_supplier1]
            supplier2 = row[col_supplier2]
            if supplier1 in name_clusters.keys():
                values=name_clusters[supplier1]
            elif supplier2 in name_clusters.keys():
                values=name_clusters[supplier2]
            else:
                values=set()
            values.add(supplier1)
            values.add(supplier2)
            name_clusters[supplier1]=values
            name_clusters[supplier2]=values
        elif label==0:
            continue
        else:
            break
    name_cluster_valuesorted={}
    for k, v in name_clusters.items():
        name_cluster_valuesorted[k] = sorted(list(v))
    return name_cluster_valuesorted

def supplier_similarity(in_file, outfile, col_supplier):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    suppliers = sorted(list(df[col_supplier].unique()))
    print("total suppliers {}".format(len(suppliers)))

    pairs=[]
    for i in range(0, len(suppliers)):
        for j in range(i+1, len(suppliers)):
            pairs.append([suppliers[i], suppliers[j]])

    scores={}
    count=0
    for p in pairs:
        count+=1
        sim=ratio(p[0], p[1])
        scores[(p[0],p[1])] = sim
        if count%1000 ==0:
            print("{}/{}".format(count, len(pairs)))

    sorted_scores={k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    with open(outfile, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(['Supplier 1', 'Supplier 2', 'NameSimilarity', 'Yes/No (1 for yes, 0 for no)'])
        for k, v in sorted_scores.items():
            writer.writerow([k[0],k[1], v])

if __name__ == "__main__":
    supplier_similarity(sys.argv[1], sys.argv[2], 'Supplier')