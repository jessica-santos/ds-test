import pandas as pd
from fancyimpute import MICE
import time


categ_variables = ['f25', 'f54', 'f55', 'f58', 'f161', 'f477']

def clean_empty_columns(df):
    n_rows = len(df)
    for c in df.columns:
        n_na = sum(df[c].isnull())
        if n_na == n_rows:
            del df[c]
            print('deletado ' + str(c))
    return df

def clean_columns_uniquevalue(df):
    for c in df.columns:
        if df[c].nunique(dropna=True) == 1:
            del df[c]
            print('deletado ' + str(c))
    return df

def treat_missing_valuesMICE(X):
    X_filled = MICE(init_fill_method='median', n_imputations=10, n_burn_in=5).complete(X)
    return X_filled

def treat_variables(df):
    df = clean_empty_columns(df)
    df = clean_columns_uniquevalue(df)
    df['f58'] = df['f58'].str.replace('/', '')
    df['f58'] = df['f58'].str.replace('.', '')
    df['f58'] = df['f58'].str.replace(',', '')

    df = pd.get_dummies(df, categ_variables)

    df['time'] = df.apply(lambda x: time.strftime('%H', time.localtime(x['epoch'] / 1000)), axis=1)
    df['day'] = df.apply(lambda x: time.strftime('%d', time.localtime(x['epoch'] / 1000)), axis=1)
    df['month'] = df.apply(lambda x: time.strftime('%m', time.localtime(x['epoch'] / 1000)), axis=1)
    df['weekday'] = df.apply(lambda x: time.strftime('%w', time.localtime(x['epoch'] / 1000)), axis=1)

    del df['epoch']

    columns = df.columns
    # Treat missing values for numerical columns
    X = treat_missing_valuesMICE(df[columns[0:495]].as_matrix())

    X = pd.DataFrame(X, columns=columns[0:495])
    df = pd.concat([X, df[columns[495:]]], axis=1)

    return df