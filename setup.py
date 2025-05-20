"""Setup file for package."""

from setuptools import find_packages, setup


def readme():
  """Get README.md content."""
  with open("README.md") as f:
    return f.read()


setup(
  name="rankfx",
  version="0.1.0",
  author="saraevn",
  author_email="saraevnik0909@gmail.com",
  description="Python package for working with NN models",
  long_description=readme(),
  long_description_content_type="text/markdown",
  url="https://github.com/sn09/rankfx",
  packages=find_packages(exclude=["notebooks"]),
  install_requires=[
    "torch>=2.2.1",
    "pydantic>=2.10.6",
    "numpy>=2.2.4",
    "pandas>=2.2.3",
    "tqdm>=4.67.1",
    "scikit-learn>=1.6.1",
],
  classifiers=[
    "Programming Language :: Python :: 3.10",
    "License ::  MIT License",
    "Operating System :: OS Independent"
  ],
  keywords="ranking torch pandas",
  project_urls={
    "Documentation": "link"
  },
  python_requires=">=3.10"
)
