import pandas as pd
import numpy as np
import gzip
import os
import glob
import csv
from sklearn.model_selection import train_test_split


os.chdir(os.getcwd() + '/data/')

# convert to csv if it has not already been converted
all_gz_filenames = [i for i in glob.glob('*.{}'.format('gz'))]
all_csv_filenames = [i for i in glob.glob('*.{}'.format('csv'))]

for gz_filename in all_gz_filenames:
    with gzip.open(gz_filename) as f:
        filename, ext = os.path.splitext(gz_filename)
        if filename not in all_csv_filenames:
            df_csv = pd.read_csv(f)
            # eliminate everything but Charged Off and Fully Paid in loan_status
            ls_df = df_csv[df_csv['loan_status'] == 'Charged Off'].append(df_csv[df_csv['loan_status'] == 'Fully Paid'])
            # eliminate 60 month loans, going to focus on 36 months because loan term matters for tied up money
            # would create a dual model in a real life scenario
            lean_df = ls_df[ls_df['term'] == ' 36 months']
            lean_df.to_csv(filename, index=False, encoding='utf-8-sig')

all_csv_filenames = [i for i in glob.glob('*.{}'.format('csv'))]
# merge into one csv if it doesn't exist
if "merged_files.csv" not in all_csv_filenames:
    df = pd.concat([pd.read_csv(f) for f in all_csv_filenames])

    # drop date columns, emp_title, and zip_code
    df.drop(columns=['issue_d',
                     'earliest_cr_line',
                     'last_credit_pull_d',
                     'sec_app_earliest_cr_line',
                     'emp_title',
                     'zip_code'],
            inplace=True)

    # drop % signs to convert columns to float
    df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '')
    df['revol_util'] = df['revol_util'].astype(str).str.replace('%', '')

    df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce') / 100
    df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce') / 100

    # fill blank int/floats with 0
    df_float = df.select_dtypes(include=['float64'])
    df_float.fillna(value=0, inplace=True)

    df_int = df.select_dtypes(include=['int64'])
    df_int.fillna(value=0, inplace=True)

    # fill blank categorical data with UNKNOWN
    df_object = df.select_dtypes(include=['object'])
    df_object.fillna(value='Unknown', inplace=True)

    # convert to categorical to factors
    df_object = pd.get_dummies(df_object)

    # merge split tables back together
    df = pd.concat([df_object, df_float, df_int], axis=1, sort=False)

    # calculate percentage gained over life of loan, will calculate in R
    df['percent_return_over_investment'] = ((36 * df['installment'] / df['funded_amnt']) * df['loan_status_Fully Paid'])

    # drop negative return values, doesn't make sense if loan was paid off
    df = df[df['percent_return_over_investment'] >= 0]
    df.drop(columns=['percent_return_over_investment'], inplace=True)

    # drop unnecessary column
    df.drop(columns=['loan_status_Charged Off'], inplace=True)

    # replace all spaces with underscores
    df.columns = df.columns.str.replace(' ', '_')

    # Cleaned and organized data into one file, only 36 mo loans
    df.to_csv("merged_files.csv", index=False)

    # split train and test dfs
    train, test = train_test_split(df, test_size=0.30, random_state=0)

    # change cwd and save to R file
    os.chdir("..")
    os.chdir(os.path.abspath(os.curdir) + '/R_files/')

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)


# original data statistics
all_csv_filenames = [i for i in glob.glob('*.{}'.format('csv'))]
all_csv_filenames.remove('merged_files.csv')
df = pd.concat([pd.read_csv(f) for f in all_csv_filenames])
print(df.describe)
# count of NaNs
print(df.isnull().sum().sum())
