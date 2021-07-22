from setuptools import setup, find_packages

setup(
    name="toy-drone",
    setup_requires=["setuptools_scm", "setuptools"],
    use_scm_version=True,
    install_requires=[
        "casadi",
        "numpy",
        "matplotlib",
        "pytest",
    ],
    python_requires=">=3.6",
    package_dir={"": "src"},
    packages=['toy_drone'],
    url="https://github.com/thilobro/toy_drone",
    license="",
    author="Thilo Bronnenmeyer",
    author_email="t.bronnenmeyer@googlemail.com",
    description="Toy drone",
    long_description="Toy drone",
)
