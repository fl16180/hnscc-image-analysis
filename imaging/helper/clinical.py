import numpy as np
import pandas as pd
import re
import glob
import os
import processing


DATA_DIR = 'C:/Users/fredl/Documents/datasets/EACRI HNSCC/'

def slides_to_patient_id():

    pattern1='_\d+-\d+_';
    pattern2='_ES1\d \d+_';

    slides = processing.get_list_of_samples(DATA_DIR + 'processed/', '*].npy')

    ids = []
    for s in slides:
        m1 = re.search(pattern1, s)
        m2 = re.search(pattern2, s)

        if m1:
            id = s[m1.start() + 1:m1.end() - 4]
        elif m2:
            id = s[m2.start() + 6:m2.end() - 1]
        else:
            print "No match for", id

        ids.append(id)

    return pd.DataFrame({'slide':slides, 'id':ids})


def load_clinical_table(loc=None):
    if loc is None:
        loc = DATA_DIR + '/clinical/clinical data HNSCC.xlsx'

    dat = pd.read_excel(loc)

    # rename columns using the first word of each column
    new_cols = [x.split(" ")[0] for x in dat.columns.values]
    new_cols[new_cols.index('overall')] = 'survival'
    dat.columns = new_cols

    names = dat['HistoNr.']
    clpattern='\d\d\d+';

    processed_names = []
    for s in names:
        m = re.search(clpattern, s)

        if m:
            id = s[m.start():m.end()]
        else:
            print "No match for", id

        processed_names.append(id)

    dat['id'] = processed_names
    return dat



def clinical_lookup_table():
    df1 = slides_to_patient_id()
    df2 = load_clinical_table()
    return pd.merge(df1, df2)


if __name__ == '__main__':
    df = clinical_lookup_table()
    df.head()
