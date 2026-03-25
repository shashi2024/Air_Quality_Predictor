from setuptools import setup, find_packages

setup(
    name="tree_carbon_ml",
    version="0.1.0",
    description="ML pipeline for U.S. tree carbon response to nitrogen deposition",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "requests>=2.28.0",
    ],
)
