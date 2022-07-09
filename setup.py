from setuptools import find_packages, setup

packages = find_packages(exclude=["tests", "tests.*"])

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(name='autouplift', version='1.0.0', description='A Python Framework for Automatically Evaluating various Uplift Modeling Algorithms to Estimate Individual Treatment Effects',
      url='https://github.com/jroessler/autouplift/tree/autoumpip', author='Jannik Rößler', author_email="", packages=packages, python_requires=">=3.8",
      install_requires=requirements, classifiers=["Programming Language :: Python3.8", "License :: OSI Approved :: Apache Software License", "Operating System :: OS Independent"])
