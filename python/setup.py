from setuptools import setup

setup(
    name='bigartm',
    version='0.0.1',
    packages=[
        'artm',
    ],
    install_requires=[
        'pandas',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'bigartm = artm.cli:main',
        ]
    }
)
