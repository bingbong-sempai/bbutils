# Updated 2023-08-27


from datetime import datetime, timedelta
from IPython.display import display
import json
import numpy as np
import pandas as pd
from pathlib import Path
import re


# Dev functions
def read_json(fn):
    'Load a json file.'
    with open(fn) as file:
        return json.load(file)

def read_json_lines(fn):
    'Read json objects stored in lines of a text file.'
    data = []
    with open(fn) as file:
        for line in file:
            data.append(json.loads(line))
    return data

def to_json(py_object, fn):
    'Save a python object as a json file.'
    with open(fn, 'w') as file:
        json.dump(py_object, file)
        
def get_next(sequence, steps=1):
    'Get the next element of an sequence.'
    iterator = iter(sequence)
    for _ in range(steps):
        val = next(iterator)
    return val

def camel_to_snake(name):
    'Convert a camel case name to snake case.'
    words = re.findall(
        r'[A-Z]?[a-z_]+' # Normal words
        r'|[A-Z]{2,}(?=[A-Z][a-z]|[^A-Za-z]|$)' # Capitalized words
        r'|\d+' # Numbers
        r'|[A-Z]', # Single capitals
        name
    )
    return '_'.join(words).lower()
    
def make_path(dir):
    'Create a path if it does not exist yet.'
    path = Path(dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Datetimes
utc_to_ph = lambda x: x + timedelta(hours=8)
today_ph = lambda: utc_to_ph(datetime.today())
today = today_ph().strftime('%y%m%d')


# Data normalization
def notna(x):
    'Check if value is np.nan or None.'
    return x == x and x is not None

def force_numeric(num, error_value=np.nan):
    'Force a text into a number.'
    if isinstance(num, Number) and notna(num):
        return num
    elif isinstance(num, str):
        num = re.sub(r'[^0-9.\-]', '', num)
        try:
            return float(num)
        except ValueError:
            return error_value
    else:
        return error_value

def normalize_text_col(col: pd.Series, split_chars='.,-/()'):
    'Clean texts to be used as identifiers.'
    split_re = r'[{}\s]+'.format(''.join(f'\{c}' for c in split_chars))
    normalized = (
        col
        # Normalize accents
        .str.normalize('NFKD')
        .str.encode('ascii', errors='ignore')
        .str.decode('utf-8')
        .str.lower()
        # Normalize splitting characters
        .str.replace(split_re, ' ', regex=True)
        .str.replace(r'[^a-z0-9 ]', '', regex=True)
        .str.strip()
    )
    return normalized



# Pandas customization
def explode_dict(series: pd.Series) -> pd.DataFrame:
    'Expand a series of dicts into a DataFrame, maintaining the index.'
    series.values[pd.isna(series.values)] = {}
    df = pd.DataFrame.from_records(series.values, series.index)
    return df

def safe_select(df, cols):
    'Filter a DataFrame to selected columns, creating the columns if necessary.'
    selected = (
        df.assign(**{col: np.nan for col in cols if col not in df})
        [cols]
    )
    return selected

def show(df: pd.DataFrame):
    'Fully display a Pandas DataFrame.'
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.max_colwidth', -1
    ):
        display(df)

pd.options.display.float_format = lambda x: f'{x:,.2f}'
pd.options.mode.copy_on_write = True
pd.core.generic.NDFrame.show = lambda self: show(self)
pd.DataFrame.glimpse = lambda df, n=3: df.head(min(n, len(df))).T.pipe(show)
pd.DataFrame.safe_select = lambda df, cols: safe_select(df, cols)
pd.Series.explode_dict = lambda ser: explode_dict(ser)
pd.Series.flip = lambda ser: pd.Series(ser.index, ser)


# Pandas data processing
def profile_df(df, sample_index=0):
    'Profile the columns of a DataFrame.'
    profile = pd.DataFrame({
        'sample': df.iloc[sample_index],
        'dtype': df.dtypes,
        'valid_fraction': df.notna().sum() / len(df),
        'unique_count': df.nunique(),
    })
    return profile

def pythonize_names(names: pd.Series, split_chars='.,-/()|') -> pd.Series:
    'Clean names to conform to Python naming conventions.'
    split_re = r'[{}\s]+'.format(''.join(f'\{c}' for c in split_chars))
    cleaned = (
        names
        .str.lower()
        .str.strip()
        .str.replace(split_re, '_', regex=True)
        .str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
        .str.replace(r'$^', '_', regex=True)
    )
    return cleaned

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    'Modify the columns of a DataFrame to conform to Python name rules.'
    colnames = pythonize_names(df.columns.astype(str))
    df.columns = [
        '_' + name if name[0].isnumeric() else name for name in colnames
    ]
    dupe_mask = df.columns.duplicated()
    if dupe_mask.any():
        dupes = df.columns[dupe_mask].unique()
        for dupe in dupes:
            dupe_matches = (df.columns == dupe) & dupe_mask
            df.columns.values[dupe_matches] = [
                f'{dupe}{i}' for i in range(1, dupe_matches.sum()+1)]
    return df

def snakify_colnames(df):
    'Convert camel case colnames to snake case.'
    df.columns = [camel_to_snake(col) for col in df.columns]
    return df

def robust_any(col):
    'Detect if column contains True-like values, with True-like exceptions.'
    try:
        return col.any()
    except ValueError:
        return True

def drop_empty_cols(df):
    'Drop columns without observed values.'
    return df.loc[:, lambda df: df.apply(robust_any)]
    
def concat_series_values(series: pd.Series, sep='|'):
    'Concatenate the values of a Series with a sep.'
    null_mask = series.isna()
    if null_mask.all():
        return None
    else:
        return series[~null_mask].astype(str).str.cat(sep=sep)


# Analysis
def cdf(series, **kwargs):
    'Generate the cumulative distribution function of a Series.'
    cdf = (
        series
        .value_counts(sort=False, **kwargs)
        .sort_index()
        .cumsum()
    )
    return cdf