import os
root = os.getcwd()

from setuptools import find_packages, setup
from setuptools.command.install import install

packages = find_packages(exclude=["tests", "tests.*"])

with open("requirements.txt") as f:
    requirements = f.readlines()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        #check_call("git clone https://github.com/jroessler/causalml.git".split(), cwd=root)
        #check_call("pip install causalml".split(), cwd=root + "/causalml")
        #check_call("pip --no-cache-dir install numpy==1.21.5 --force-reinstall".split(), cwd=root + "/causalml")


setup(name='autouplift',
      version='1.0.0',
      description='A Python Framework for Automatically Evaluating various Uplift Modeling Algorithms to Estimate Individual Treatment Effects',
      url='https://github.com/jroessler/autouplift/tree/autoumpip',
      author='Jannik Rößler',
      author_email="",
      packages=packages,
      cmdclass={'install': PostInstallCommand},
      python_requires=">=3.8",
      install_requires=requirements,
      classifiers=["Programming Language :: Python3.8", "License :: OSI Approved :: Apache Software License", "Operating System :: OS Independent"]
      )