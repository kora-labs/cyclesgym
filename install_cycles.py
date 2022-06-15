import requests
from sys import platform
import pathlib
import zipfile
from cyclesgym.utils.paths import CYCLES_PATH
import stat
import subprocess


def test_cycles_installation():
    if pathlib.Path.joinpath(CYCLES_PATH, pathlib.Path('Cycles')).is_file():
        proc = subprocess.run(['./Cycles', '-b', 'ContinuousCorn'], cwd=CYCLES_PATH)
        out = pathlib.Path.joinpath(CYCLES_PATH, pathlib.Path('output/ContinuousCorn'))
        files = [x for x in out.iterdir() if x.is_file()]
        if len(files) >= 5 and proc.returncode == 0:
            print('Cycles installed correctly')
            return True

    print('Cycles not installed properly')
    return False


def install_cycles():
    # Skip if already exists
    install_dir = pathlib.Path.cwd().joinpath('cycles')
    cycles_installed = False
    if install_dir.is_dir():
        cycles_installed = test_cycles_installation()
        if cycles_installed:
            return

    if not cycles_installed:
        print('Installing cycles..')
        # Define file name
        if platform == "linux" or platform == "linux2":
            fname = 'Cycles_debian_0.12.9-alpha.zip'
        elif platform == "darwin":
            fname = 'Cycles_macos_0.12.9-alpha.zip'
        elif platform == "win32":
            fname = 'Cycles_win_0.12.9-alpha.zip'
        else:
            print('Installation aborted\nOperating system not recognized')
            return

        if not pathlib.Path.cwd().joinpath(fname).is_file():
            # Get file
            baseurl = 'https://github.com/PSUmodeling/Cycles/releases/download/v0.12.9-alpha/'
            url = baseurl + fname
            r = requests.get(url, allow_redirects=True)
            open(fname, 'wb').write(r.content)

        # Unzip
        with zipfile.ZipFile(pathlib.Path.cwd().joinpath(fname), 'r') as zip_ref:
            install_dir.mkdir(exist_ok=True)
            zip_ref.extractall(install_dir)

        # Remove zip
        pathlib.Path.cwd().joinpath(fname).unlink()
        CYCLES_PATH.joinpath('Cycles').chmod(stat.S_IEXEC)

    test_cycles_installation()


if __name__ == '__main__':
    install_cycles()
