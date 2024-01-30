from setuptools import find_packages, setup

setup(
    name="flextrees",
    version="0.1.0",
    author="Alberto Argente-Garrido",
    keywords="decision-trees explainable FL federated-learning flexible",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["flex", "numpy", "bitarray", "ucimlrepo"],
)