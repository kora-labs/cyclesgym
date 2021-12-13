import setuptools
from setuptools.command.install import install
from pathlib import Path
from install_cycles import install_cycles

# with open("README.md", "r", encoding='utf_8') as fh:
#     long_description = fh.read()

install_requires=[
    'requests',
    'gym',
    'pandas',
    'matplotlib',
    'ipykernel',
    'pyglet'
]


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        cycles_path = Path.cwd().joinpath('cycles')
        if not cycles_path.is_dir():
            print(f'Cycles not found at {cycles_path}')
            print(f'Installing cycles at {cycles_path}')
            install_cycles()


setuptools.setup(
    name='cyclesgym',
    version='0.1.0',
    description='Open AI gym interface to the cycles crop simulator',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url='https://github.com/zuzuba/cyclesgym',
    author='Matteo Turchetta',
    author_email='matteo.turchetta@inf.ethz.ch',
    keywords='Crop growth sumulator',
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=install_requires,
    cmdclass={'install': PostInstallCommand},
)