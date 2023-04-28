'''
Reads the initial coding (last sheet of https://docs.google.com/spreadsheets/d/1wZQlF2g8wsFUtt4E6wkvRdIlnkMdiHojI5VAXPDfnzo/edit#gid=1956064770,
or saved locally at /Data/tas_etp/initial_coding_september-last-sheet.csv)

parses the csv into another format with these columns:

suplier, party, total spend, expense category, expense sub category, example, notes

'''
import pandas as pd
import sys, numpy
def convert(in_file, index_first_category_in_header,
            out_folder,
            col_supplier, col_party, col_total, col_more_than_1, col_notes,
            col_missing):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    #work out the taxonomy tree
    topcat, subcat, top_to_sub, sub_to_top=calculate_taxonomy(df, index_first_category_in_header)
    #print taxonomy
    for k, v in top_to_sub.items():
        print("+ {}".format(k))
        for sub in v:
            print("\t\t - {}".format(sub))

    datarows=[]
    #process each row and convert its content format
    for index, row in df.iterrows():
        if index==0:
            continue
        supplier = row[col_supplier]
        party=row[col_party]
        total=row[col_total]
        if type(total) is str and not total[0].isdigit():
            total=total[1:].replace(",","")
        more_than_1=nan_to_str(row[col_more_than_1])
        notes=nan_to_str(row[col_notes])
        missing=nan_to_str(row[col_missing])

        #find which subcategory
        assigned_subcat=set()
        allocated_example=set()
        for sub_i, sub_name in subcat.items():
            v = row[sub_i]
            if type(v) is str and (v.lower().startswith("1") or v.lower().startswith('y')):
                example = row[sub_i+1]
                assigned_subcat.add(sub_name)
                if type(example) is not str:
                    example=""
                allocated_example.add(example)
            elif type(v) is str and not (v.lower().startswith("1") or v.lower().startswith("y")):
                if sub_name.lower().startswith("completely unclear"):
                    example = row[sub_i + 1]
                    assigned_subcat.add(sub_name)
                    if type(example) is not str:
                        example = ""
                    allocated_example.add(example)
                else:
                    print("unexpected value, row={}, subcat={}, value={}".format(index+2, sub_name, v))
                    notes+=" | "+str(v)

        #flatten values
        subcat_str=""
        example_str=""
        topcat_str=""
        for sc in assigned_subcat:
            subcat_str+=sc+"|"
            topcat_str+=sub_to_top[sc]+"|"
        for e in allocated_example:
            example_str+=e+"|"

        #create new row
        if subcat_str.endswith("|"):
            subcat_str=subcat_str[0:len(subcat_str)-1]
        if example_str.endswith("|"):
            example_str=example_str[0:len(example_str)-1]
        if topcat_str.endswith("|"):
            topcat_str=topcat_str[0:len(topcat_str)-1]

        if subcat_str=="" or topcat_str=="":
            print("empty cat, row={}, money={}".format(index+2, total))

        newrow=[supplier, party, total, topcat_str, subcat_str, example_str, more_than_1, missing,notes]
        datarows.append(newrow)

    new_df= pd.DataFrame(datarows, columns=['Supplier',
                                            'Party',
                                            'Total',
                                            'TopCategory',
                                            'SubCategory',
                                            'Example',
                                            'MultipleItems',
                                            'MissingInvoice',
                                            'Notes'])
    new_df.to_csv(out_folder+"/converted_data.csv", sep=',', encoding='utf-8')

def calculate_taxonomy(dataframe, index_first_cateogry):
    top_column_headers= list(dataframe.head())
    top_category={}
    #find the starting index of each category
    for i in range(index_first_cateogry, len(top_column_headers)):
        h = top_column_headers[i]
        if not h.startswith("Unnamed"):
            top_category[i]=h

    #calculate the subcategory indeces
    firstrow=list(dataframe.iloc[[0]].values.flatten())
    top_category_children={}
    sub_category_parent={}
    sub_category={}

    top_category_indeces=sorted(list(top_category.keys()))
    for x in range(0, len(top_category_indeces)):
        sub_cats=[]
        top_cat = top_category[top_category_indeces[x]]
        start=top_category_indeces[x]
        if x == len(top_category_indeces)-1:
            end = len(firstrow)
        else:
            end=top_category_indeces[x+1]

        #now we know the start and end indeces of the columns corresponding to the subcategories of this topc category (start)
        #lets find the sub categories
        for j in range(start, end):
            value = firstrow[j]
            if type(value) is str:
                sub_category[j]=value
                sub_cats.append(value)
                sub_category_parent[value]=top_cat
        top_category_children[top_cat]=sorted(sub_cats)

    return top_category, sub_category, top_category_children, sub_category_parent

def nan_to_str(value):
    if type(value) is not str:
        if numpy.isnan(value):
            return ""
    else:
        return value




if __name__ == "__main__":
    convert(sys.argv[1], int(sys.argv[2]),
            sys.argv[3],
            col_supplier=0,
            col_party=1,
            col_total=2,
            col_more_than_1=5,
            col_notes=6,
            col_missing=7)