from setuptools import setup, find_packages

setup(
    name="train_diabetes",
    version="0.1.0",
    description="Vertex AI training package for diabetes classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib"
    ],
)
