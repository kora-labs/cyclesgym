import pandas as pd
from datetime import datetime as dt


__all__ = ['date_to_ydoy', 'ydoy_to_date', 'num_lines']


def date_to_ydoy(df, old_col_name='DATE', new_col_names=['YEAR', 'DOY'], inplace=False):
    new_df = df.copy(deep=True) if not inplace else df
    position = new_df.columns.get_loc(old_col_name)
    new_df.insert(position, new_col_names[1], pd.to_datetime(new_df[old_col_name]).dt.dayofyear)
    new_df.insert(position, new_col_names[0], pd.to_datetime(new_df[old_col_name]).dt.year)
    new_df.drop(columns=old_col_name, inplace=True)
    return new_df
    # # position = new.columns.get_loc(old_col_name)


def ydoy_to_date(df, old_col_names=['YEAR', 'DOY'], new_col_name='DATE', inplace=False):
    dates = [dt.strftime(dt.strptime(f'{year}-{doy}', '%Y-%j'), format='%Y-%m-%d') for year, doy in zip(df[old_col_names[0]], df[old_col_names[1]])]
    new_df = df.copy(deep=True) if not inplace else df
    position = new_df.columns.get_loc(old_col_names[0])
    new_df.insert(position, new_col_name, dates)
    new_df.drop(columns=old_col_names, inplace=True)
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
