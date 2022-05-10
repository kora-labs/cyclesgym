import setuptools
from setuptools.command.install import install

# with open("README.md", "r", encoding='utf_8') as fh:
#     long_description = fh.read()

install_requires=[
    'requests',
    'numpy',
    'pandas',
    'matplotlib',
]


env_requires = [
    'gym',
    'ipykernel',
    'pyglet',
]

solve_requires = [
    'torch >= 1.8.1+cpu',
    'stable-baselines3',
    'tensorboard',
    'imitation @ git+https://git@github.com/HumanCompatibleAI/imitation@cf7e4074f1d4786f22c74a69dddb251f71d288df#egg=imitation', # Install cf7e4074f1d4786f22c74a69dddb251f71d288df commit of imitation library
    'pygmo'
]


class new_install(install):
    """Post-installation for installation mode."""
    def __init__(self, *args, **kwargs):
        super(new_install, self).__init__(*args, **kwargs)
        print('POST INSTALL')
        from install_cycles import install_cycles

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
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
            "ENV": env_requires,
            "ENV_SOLVERS": env_requires + solve_requires
        },
    cmdclass={'install': new_install},
)


# TODO: check for platform and include pygmo only for linux
#       for other platform, include a PRE-installation script that installs pygmo
#       using conda

