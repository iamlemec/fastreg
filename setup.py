from setuptools import setup
from pathlib import Path

# read the contents of your README file
here = Path(__file__).parent
long_description = (here / 'README.md').read_text().strip()
requirements = (here / 'requirements.txt').read_text().strip().split('\n')

setup(
    name='fastreg',
    version='0.9',
    description='Fast sparse regressions',
    url='http://github.com/iamlemec/fastreg',
    author='Doug Hanley',
    author_email='thesecretaryofwar@gmail.com',
    license='MIT',
    packages=['fastreg'],
    zip_safe=False,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
)
