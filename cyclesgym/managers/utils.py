import pandas as pd
from datetime import datetime as dt


__all__ = ['date_to_ydoy', 'ydoy_to_date', 'num_lines']


# TODO: Position=1 by default relies on the drop afterwards, update it. Don't do things in place
def date_to_ydoy(df, old_col_name='DATE', new_col_names=['YEAR', 'DOY'], position=1, inplace=False):
    df.insert(position, new_col_names[1], pd.to_datetime(df[old_col_name]).dt.dayofyear)
    df.insert(position, new_col_names[0], pd.to_datetime(df[old_col_name]).dt.year)
    df.drop(columns=old_col_name, inplace=True)
    return df
    # new_df = df.copy(deep=True) if not inplace else df
    # # position = new.columns.get_loc(old_col_name)
    # new_df.insert(position, f'DOY{new_col_name}', pd.to_datetime(new_df[old_col_name]).dt.dayofyear)
    # new_df.insert(position, f'YEAR{new_col_name}', pd.to_datetime(new_df[old_col_name]).dt.year)
    # new_df.drop(columns=old_col_name, inplace=True)
    # return new_df


def ydoy_to_date(df, old_col_names=['YEAR', 'DOY'], new_col_name='DATE', position=1, inplace=False):
    dates = [dt.strftime(dt.strptime(f'{year}-{doy}', '%Y-%j'), format='%Y-%m-%d') for year, doy in zip(df[old_col_names[0]], df[old_col_names[1]])]
    new_df = df.copy()
    # new_df = df.copy(deep=True) if not inplace else df
    new_df.insert(position, new_col_name, dates)
    new_df.drop(columns=old_col_names, inplace=True)
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
