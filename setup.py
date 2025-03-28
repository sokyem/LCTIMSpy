# setup.py

from setuptools import setup, find_packages

setup(
    name='LCTIMSpy',
    version='0.1.0',
    description='A package for analyzing TIMSTOF LC-TIMS–MS data, generated by a CID_TIMS Workflow for peptide isomers identification',
    author='Samuel Okyem',
    author_email='okyemsamuel@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0',
        'numpy>=1.18',
        'matplotlib>=3.1',
        'seaborn',
        'scipy',
        'plotly',
        'pyteomics',
        'statannotations',  
        'statsmodels',
        'pyTDFSDK @ git+https://github.com/gtluu/pyTDFSDK.git'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
