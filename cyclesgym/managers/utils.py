import pandas as pd
from datetime import datetime as dt


__all__ = ['date_to_ydoy', 'ydoy_to_date', 'num_lines']


# TODO: Position=1 by default relies on the drop afterwards, update it. Don't do things in place
def date_to_ydoy(df, old_col_name, new_col_name, position=1):
    df.insert(position, f'DOY{new_col_name}', pd.to_datetime(df[old_col_name]).dt.dayofyear)
    df.insert(position, f'YEAR{new_col_name}', pd.to_datetime(df[old_col_name]).dt.year)
    df.drop(columns=old_col_name, inplace=True)
    return df


def ydoy_to_date(df, old_col_name, new_col_name, position=1):
    dates = [dt.strftime(dt.strptime(f'{year}-{doy}', '%Y-%j'), format='%Y-%m-%d') for year, doy in zip(df[f'YEAR{old_col_name}'], df[f'DOY{old_col_name}'])]
    new_df = df.copy()
    new_df.insert(position, f'DATE{new_col_name}', dates)
    new_df.drop(columns=[f'DOY{old_col_name}', f'YEAR{old_col_name}'], inplace=True)
    # df.insert(position, f'DATE{new_col_name}',
    #           pd.to_datetime(df[old_col_name]).dt.dayofyear)
    return new_df


def num_lines(file):
    """
    Count number of lines in a file.
    Parameters
    ----------
    file: str of pathlib.Path
    """
    with open(file, 'r') as f:
        for i, l in enumerate(f, 1):
            pass
        return i
