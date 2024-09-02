from setuptools import setup, find_packages

setup(
    name="policyengine_us_data",
    version="0.1.0",
    author="PolicyEngine",
    author_email="hello@policyengine.org",
    description="A package to create representative microdata for the US",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "policyengine_core",
        "tables",
        "survey_enhance",
        "torch",
        "requests",
        "tqdm",
        "tabulate",
        "tables",
    ],
    extras_require={
        "dev": [
            "black",
            "pytest",
            "policyengine_us==1.69.0",
            "streamlit",
        ],
    },
)
