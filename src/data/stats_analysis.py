'''
loads convereted data created by coding_converter.py (e.g. initial_coding_september-last-sheet-convered.csv)
and apply analyses:
- overlap of suppliers between pairs of parties
- per party % spend per major and sub category
- per major and category party spend as %
- per party %spend each supplier
'''
import pandas as pd
import matplotlib.pyplot as plt
import sys,re
plt.style.use('ggplot')

'''
each party has one stacked bar
'''
def party_spend_category_percent(in_file, col_party, col_category, col_spend, out_folder):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    df=flatten(df,col_category, col_spend)
    parties = sorted(list(df[col_party].unique()))
    categories = sorted(list(df[col_category].unique()))

    party_category_percent_data={}
    party_label_data={}

    print("total parties {}".format(len(parties)))

    for p in parties: #for each party, we calc its % spend for each categry and insert a value into the data holder
        subset = df[(df[col_party] == p)]
        total_spend=round(subset[col_spend].sum(),0)

        party_label=p+"="+str(total_spend)
        party_label_data[p]=party_label
        cat_percent_data=[]

        for c in categories:
            sub_subset=subset[(subset[col_category] == c)]
            if len(sub_subset)==0:
                cat_percent_data.append(0)
            else:
                total_spend_by_cat=sub_subset[col_spend].sum()
                percent_total_spend_by_cat =total_spend_by_cat/total_spend
                cat_percent_data.append(percent_total_spend_by_cat)
        party_category_percent_data[p]=cat_percent_data

    #drawing the graph
    print("graph")
    cols = ["Party"]
    cols.extend(categories)
    rows=[]
    for p, percentdata in party_category_percent_data.items():
        row=[party_label_data[p]]
        row.extend(list(percentdata))
        rows.append(row)

    df = pd.DataFrame(rows,
                      columns=cols)
    df.plot(x='Party', kind='bar', stacked=True,colormap='Paired',
            title='Party spend by top category (%)',linewidth=2)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=5, title_fontsize=10)
    plt.xlabel("Parties and Their Total Spending")
    plt.ylabel("Percentage by Category")
    plt.title("Party spending by {} (in %)".format(col_category))
    plt.tight_layout()
    plt.savefig(out_folder + "/party_spend_by_{}.png".format(col_category), format='png', dpi=300, bbox_inches='tight')
    plt.clf()
   # plt.show()


'''
each category has one stacked bar (of party)
'''
def category_spend_party_percent(in_file, col_party, col_category, col_spend, out_folder):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    df=flatten(df,col_category, col_spend)
    parties = sorted(list(df[col_party].unique()))
    categories = sorted(list(df[col_category].unique()))

    category_party_percent_data={}
    category_label_data={}

    print("total categories {}".format(len(categories)))

    for c in categories: #for each party, we calc its % spend for each categry and insert a value into the data holder
        subset = df[(df[col_category] == c)]
        total_spend=round(subset[col_spend].sum(),0)

        category_label=c+"="+str(total_spend)
        category_label_data[c]=category_label
        party_percent_data=[]

        for p in parties:
            sub_subset=subset[(subset[col_party] == p)]
            if len(sub_subset)==0:
                party_percent_data.append(0)
            else:
                total_spend_by_party=sub_subset[col_spend].sum()
                percent_total_spend_by_party =total_spend_by_party/total_spend
                party_percent_data.append(percent_total_spend_by_party)
        category_party_percent_data[c]=party_percent_data

    #drawing the graph
    print("graph")
    cols = ["Category"]
    cols.extend(parties)
    rows=[]
    for c, percentdata in category_party_percent_data.items():
        row=[category_label_data[c]]
        row.extend(list(percentdata))
        rows.append(row)

    df = pd.DataFrame(rows,
                      columns=cols)
    df.plot(x='Category', kind='bar', stacked=True,colormap='Paired',
            title='{} spend by party (%)'.format(col_category),linewidth=2)
    plt.legend(bbox_to_anchor=(-0.1, -1), loc='upper left', fontsize=4, title_fontsize=8)
    plt.xlabel("Categories and Total Spending")
    plt.ylabel("Percentage by Party")
    plt.tight_layout()
    plt.savefig(out_folder + "/{}_spend_by_party.png".format(col_category), format='png', dpi=300, bbox_inches='tight')
    plt.clf()
   # plt.show()

'''
there are entries with multiple categories and we need to duplicate those records and split the value evenly
'''
def flatten(dataframe, col_category, col_spend):
    columns = list(dataframe.head())
    datarows=[]
    for index, row in dataframe.iterrows():
        cat = row[col_category]
        if type(cat) is not str:
            continue
        if "|" in cat:
            cats = cat.split("|")
            total=float(row[col_spend])
            average = total/len(cats)
            for c in cats:
                newrow = row.copy(deep=True)
                newrow[col_category] = c
                newrow[col_spend]=average
                datarows.append(list(newrow))
        else:
            datarows.append(list(row))
    new_df = pd.DataFrame(datarows, columns=columns)
    return new_df

def party_supplier_overlap(in_file, col_party, col_supplier, out_folder):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    parties = sorted(list(df[col_party].unique()))
    print("total parties {}".format(len(parties)))
    party_pairs=[]
    for i in range(0, len(parties)):
        for j in range(i+1, len(parties)):
            party_pairs.append((parties[i], parties[j]))

    dice_scores={}
    for pp in party_pairs:
        p1 = pp[0]
        p2 = pp[1]
        #p1 suppliers
        subset = df[(df[col_party] == p1)]
        p1_suppliers = sorted(list(subset[col_supplier].unique()))

        # p2 suppliers
        subset = df[(df[col_party] == p2)]
        p2_suppliers = sorted(list(subset[col_supplier].unique()))

        dice=calculate_dice(set(p1_suppliers), set(p2_suppliers))
        if dice>0:
            dice_scores[p1 +" - "+ p2]=dice

    #create bar chart
    sorted_scores = {k: v for k, v in sorted(dice_scores.items(), key=lambda item: item[1], reverse=True)}

    x = list(sorted_scores.keys())
    dicescores = list(sorted_scores.values())

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, dicescores, color='green')
    plt.ylabel("Dice Score")
    plt.xlabel("Party Pairs")
    plt.title("Supplier overlap between Party pairs")

    plt.xticks(x_pos, x)
    plt.xticks(rotation=90, ha='right')
    #plt.canvas.draw()
    plt.tight_layout()
    plt.savefig(out_folder + "/party_supplier_overlap.png", format='png', dpi=300, bbox_inches='tight')
    plt.clf()

def party_spending_per_supplier(in_file, col_party, col_supplier, col_spend, out_folder, max=20):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    parties = sorted(list(df[col_party].unique()))
    print("total parties {}".format(len(parties)))
    for p in parties:
        subset = df[(df[col_party] == p)]
        suppliers = sorted(list(subset[col_supplier].unique()))
        total_spend = round(subset[col_spend].sum(), 0)
        supplier_percent={}
        for s in suppliers:
            sub_subset=subset[(subset[col_supplier] == s)]
            total_by_supplier=sub_subset[col_spend].sum()
            percent =total_by_supplier/total_spend
            supplier_percent[s]=percent

        # create bar chart
        sorted_scores = {k: v for k, v in sorted(supplier_percent.items(), key=lambda item: item[1], reverse=True)}

        x = list(sorted_scores.keys())
        x=x[0:max]
        supplier_percent = list(sorted_scores.values())
        supplier_percent=supplier_percent[0:max]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, supplier_percent, color='green')
        plt.ylabel("Percentage of Party Total")
        plt.xlabel("Suppliers")
        plt.title("% of spend per supplier")

        plt.xticks(x_pos, x)
        plt.xticks(rotation=90, ha='right')
        # plt.canvas.draw()
        plt.tight_layout()
        plt.savefig(out_folder + "/party_{}_spend_by_supplier.png".format(re.sub('[^0-9a-zA-Z]+', '_', p)), format='png', dpi=300, bbox_inches='tight')
        plt.clf()

def calculate_dice(set1, set2):
    inter=set1.intersection(set2)
    union=set1.union(set2)
    return 2*len(inter)/(len(set1)+len(set2))

if __name__ == "__main__":
    in_file="/home/zz/Data/tas_etp/initial_coding_september-last-sheet-converted.csv"
    out_folder="/home/zz/Data/tas_etp"
    party_supplier_overlap(in_file, 'Party', 'Supplier',out_folder)

    #party_spend_category_percent(in_file, 'Party', 'TopCategory','Total',out_folder)
    #party_spend_category_percent(in_file, 'Party', 'SubCategory', 'Total', out_folder)

    #category_spend_party_percent(in_file, 'Party', 'TopCategory', 'Total', out_folder)
    #category_spend_party_percent(in_file, 'Party', 'SubCategory', 'Total', out_folder)
    #party_spending_per_supplier(in_file, 'Party', 'Supplier', 'Total', out_folder)