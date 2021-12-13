import setuptools

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
    install_requires=install_requires
)