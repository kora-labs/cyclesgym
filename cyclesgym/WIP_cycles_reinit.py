import pathlib
import stat
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil

from cyclesgym.paths import CYCLES_PATH

def call_cycles(control_file, doy=None):
    CYCLES_PATH.joinpath('Cycles').chmod(stat.S_IEXEC)

    # Run cycles
    if doy is None:
        subprocess.run(['./Cycles', '-b', control_file], cwd=cycles_dir)
    # Run cycles dumping reinit file
    else:
        subprocess.run(['./Cycles', '-b', '-l', str(doy), control_file], cwd=cycles_dir)


def load_output(control_file, fname='CornRM.90.dat'):
    output_dir = CYCLES_PATH.joinpath('output', control_file)
    df = pd.read_csv(output_dir.joinpath(fname), sep='\t').drop(0, 0)
    df.columns = df.columns.str.strip(' ')
    numeric_cols = df.columns[3:]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df


def plot_trajectory(control_file):
    df = load_output(control_file=control_file)
    numeric_cols = df.columns[3:]

    for n in numeric_cols:
        plt.figure()
        plt.title(f'{n} {control_file}')
        df[n].plot()
    plt.show()
    return df


def copy_reinit(control_file, doy):
    """Move reinit to input folder and rename it"""
    output_dir = CYCLES_PATH.joinpath('output', control_file)
    input_dir = CYCLES_PATH.joinpath('input')
    shutil.copy(output_dir.joinpath('reinit.dat'), output_dir.joinpath('reinitcopy.dat'))
    return output_dir.joinpath('reinitcopy.dat').rename(input_dir.joinpath(f'{control_file}{doy}.reinit'))


def create_reinit_control_file(ctrl_file, doy, reinit_file):
    input_dir = CYCLES_PATH.joinpath('input')
    old_ctrl = input_dir.joinpath(f'{ctrl_file}.ctrl')
    new_ctrl_name = f'{ctrl_file}Reinit{doy}'
    new_ctrl = input_dir.joinpath(f'{new_ctrl_name}.ctrl')
    shutil.copy(old_ctrl, new_ctrl)

    f = open(old_ctrl, 'r')
    linelist = f.readlines()
    f.close

    # Re-open file here
    f2 = open(new_ctrl, 'w')
    for line in linelist:
        if line.startswith('USE_REINITIALIZATION'):
            line = line.replace('0', '1')
        if line.startswith('REINIT_FILE'):
            reinit_file = str(reinit_file)
            reinit_file = reinit_file[reinit_file.rfind('/')+1:]
            line = line.replace('N/A', str(reinit_file))
        f2.write(line)
    f2.close()

    return new_ctrl_name


def main():
    control_file = 'ContinuousCorn'
    doy = 200
    call_cycles(control_file, doy)
    df = load_output(control_file)

    reinit_file = copy_reinit(control_file, doy)
    new_control = create_reinit_control_file(control_file, doy, reinit_file)
    call_cycles(new_control, doy=None)
    df1 = load_output(new_control)


    control_random = 'ContinuousCornReinit200random'
    call_cycles(control_random)
    df2 = load_output(control_random)

    for col in range(3, 17):
        same_reinit_error = np.max(np.abs((df.iloc[:, col] - df1.iloc[:, col]) / df.iloc[:, col]))
        random_reinit_error = np.max(np.abs((df.iloc[:, col] - df2.iloc[:, col]) / df.iloc[:, col]))
        print(f'Maximum relative error for {df.columns[col]}\n'
              f'Same reinit {same_reinit_error*100}\tRandom reinit {random_reinit_error * 100}')
        plt.figure()
        plt.plot(df1.iloc[:, col] - df.iloc[:, col], label='Resumed')
        plt.plot(df2.iloc[:, col] - df.iloc[:, col], label='Random resumed')
        plt.title(df.columns[col])
        plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


