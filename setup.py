from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='fastreg',
    version='0.9',
    description='Fast sparse regressions',
    url='http://github.com/iamlemec/fastreg',
    author='Doug Hanley',
    author_email='thesecretaryofwar@gmail.com',
    license='MIT',
    packages=['fastreg'],
    install_requires=['numpy', 'scipy', 'pandas', 'sklearn'],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown',
)
