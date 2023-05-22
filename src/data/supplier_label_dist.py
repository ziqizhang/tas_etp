'''
Look at how many labels each supplier has been assigned
'''
import pandas as pd
import csv,sys

def analyse_label_dist(in_file, col_supplier, col_top_category, col_sub_category, out_file):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    suppliers = sorted(list(df[col_supplier].unique()))

    rows=[]
    for s in suppliers:
        subset = df[(df[col_supplier] == s)]
        topcategories = (list(subset[col_top_category].unique()))
        subcategories = (list(subset[col_sub_category].unique()))

        topcategories_flattened=set()
        for t in topcategories:
            if type(t) is not str:
                continue
            if '|' in t:
                parts= t.split('|')
                for p in parts:
                    topcategories_flattened.add(p.strip())
            else:
                topcategories_flattened.add(t.strip())
        subcategories_flattened=set()
        for sub in subcategories:
            if type(t) is not str:
                continue
            if '|' in sub:
                parts= sub.split('|')
                for p in parts:
                    subcategories_flattened.add(p.strip())
            else:
                subcategories_flattened.add(sub.strip())
        rows.append([s, len(topcategories_flattened), len(subcategories_flattened)])

    with open(out_file,'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Supplier","TopCategories","SubCategories"])
        for r in rows:
            writer.writerow(r)

if __name__ == "__main__":
    analyse_label_dist(sys.argv[1],'Supplier','TopCategory','SubCategory',sys.argv[2])