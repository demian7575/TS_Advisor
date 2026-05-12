from setuptools import find_packages, setup

setup(
    name="ts-advisor",
    version="0.1.0",
    description="Behavior-preserving refactor of the TS Advisor notebook into modular classical ML code.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    install_requires=["joblib", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "notebook"],
    extras_require={"test": ["pytest"]},
)
