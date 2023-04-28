'''
loads convereted data created by coding_converter.py (e.g. initial_coding_september-last-sheet-convered.csv)
find unique set of suppliers, compute pair wide similarity using levenstein and output a csv
'''
import pandas as pd
from Levenshtein import ratio
import csv,sys

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