from setuptools import setup

setup(
    name='fastreg',
    version='0.1',
    description='Fast sparse regressions',
    url='http://github.com/iamlemec/fastreg',
    author='Doug Hanley',
    author_email='thesecretaryofwar@gmail.com',
    license='MIT',
    packages=['fastreg'],
    install_requires=['numpy', 'scipy', 'pandas', 'sklearn'],
    zip_safe=False
)
