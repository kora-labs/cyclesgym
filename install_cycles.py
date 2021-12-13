import requests
from sys import platform
import pathlib
import zipfile


def install_cycles():
    # Skip if already exists
    install_dir = pathlib.Path.cwd().joinpath('cycles')
    if install_dir.is_dir():
        print('Installation aborted\nA folder named cycles already exists')
        return

    # Define file name
    if platform == "linux" or platform == "linux2":
        fname = 'Cycles_debian_0.12.9-alpha.zip'
    elif platform == "darwin":
        fname = 'Cycles_macos_0.12.9-alpha.zip'
    elif platform == "win32":
        fname = 'Cycles_win_0.12.9-alpha.zip'

    if not pathlib.Path.cwd().joinpath(fname).is_file():
        # Get file
        baseurl = 'https://github.com/PSUmodeling/Cycles/releases/download/v0.12.9-alpha/'
        url = baseurl + fname
        r = requests.get(url, allow_redirects=True)
        open(fname, 'wb').write(r.content)

    # Unzip
    with zipfile.ZipFile(pathlib.Path.cwd().joinpath(fname), 'r') as zip_ref:
        install_dir.mkdir(exist_ok=False)
        zip_ref.extractall(install_dir)

    # Remove zip
    pathlib.Path.cwd().joinpath(fname).unlink()


if __name__ == '__main__':
    install_cycles()
