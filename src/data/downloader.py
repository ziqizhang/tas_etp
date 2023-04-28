'''
This file downloads invoices given the exported invoice spreadsheet
(link https://search.electoralcommission.org.uk/Search/Spending?currentPage=1&rows=20&sort=TotalExpenditure&order=desc&tab=1&open=filter&et=pp&includeOutsideSection75=true&evt=ukparliament&ev=3696&optCols=ExpenseCategoryName&optCols=AmountInEngland&optCols=AmountInScotland&optCols=AmountInWales&optCols=AmountInNorthernIreland&optCols=DatePaid)
(example exported file: /home/zz/Data/tas_etp/results.csv)
'''

import numpy as np
import os,sys
import pandas as pd
import urllib.request


list_path = sys.argv[1]
export_path = sys.argv[2]

'''
Obtain 'invoice_list.csv from
http://search.electoralcommission.org.uk/Search/Spending?currentPage=1&rows=20&sort=ECRef&order=desc&tab=1&open=filter&et=pp&includeOutsideSection75=true&evt=ukparliament&ev=3696&optCols=ExpenseCategoryName&optCols=AmountInEngland&optCols=AmountInScotland&optCols=AmountInWales&optCols=AmountInNorthernIreland&optCols=DatePaid
Bottom right of result list - Export Results as csv
'''
df = pd.read_csv(list_path)

os.makedirs(export_path,exist_ok=True)
for ID in df['RedactedSupportingInvoiceId']:
    if not np.isnan(ID):
        ID = str(int(ID))
        tar = os.path.join(export_path, ID+'.pdf')
        urllib.request.urlretrieve('http://search.electoralcommission.org.uk/Api/Spending/Invoices/'+ID, tar)
        print("Downloaded",tar)
print("Download finished.")