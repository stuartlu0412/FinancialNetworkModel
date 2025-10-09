from setuptools import find_packages, setup

setup(
    name='financial_network_model',
    version='0.1.0',
    packages=find_packages(exclude=['experiments','notebooks','reports','results']),
    install_requires=[
        'numpy', 'pandas', 'networkx', 'matplotlib', 'scipy'
    ],
)