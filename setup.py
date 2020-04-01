from setuptools import setup

setup(
    name='nlpred',
    url='https://github.com/JWLevesque/nlpred',
    author='Joseph Levesque',
    author_email='JWLevesque@gmail.com',
    packages=['nlpred'],
    # Dependencies
    install_requires=['numpy', 'pandas', 'json', 're', 'sklearn', 'warnings', 'pickle'],
    version='0.1',
    license='GNU GPLv3',
    description='This package provides support for making predictions of binary outcomes based on natural language corpora.',
    long_description=open('README.txt').read(),
)