from setuptools import setup, find_packages

setup(
    # Name of the package
    name="toy-drone",
    setup_requires=["setuptools_scm", "setuptools"],
    use_scm_version=True,
    install_requires=[
        # kiteswarms packages
        "casadi",  # version determined by kiteswarms-models
        "numpy",  # >= 1.16.0 required for linspace interpolation of vectors -> use current
        "matplotlib",
    ],
    python_requires=">=3.6",
    package_dir={"": "src"},
    # Packages to add: find_packages provided by setuptools finds all packages located in subfolders
    # packages=find_packages("src"),
    packages=['toy_drone'],
    # Some meta date for Debian compliance
    url="https://kiteswarms.gitlab-pages.kiteswarms.com/ocp_trajectory_gen",
    license="",  # NO LICENSE!!
    author="Thilo Bronnenmeyer",
    author_email="t.bronnenmeyer@googlemail.com",
    description="Toy drone",
    long_description="Toy drone",
)
