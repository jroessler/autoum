import os
root = os.getcwd()

from subprocess import check_call

from setuptools import find_packages, setup
from setuptools.command.install import install

packages = find_packages(exclude=["tests", "tests.*"])

with open("requirements.txt") as f:
    requirements = f.readlines()


class PreInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        check_call("git clone https://github.com/jroessler/causalml.git".split(), cwd=root)
        check_call("pip install -r requirements.txt".split(), cwd=root + "/causalml")
        check_call("python setup.py build_ext --inplace".split(), cwd=root + "/causalml")
        check_call("python setup.py install".split(), cwd=root + "/causalml")
        install.run(self)


setup(name='autouplift',
      version='1.0.0',
      description='A Python Framework for Automatically Evaluating various Uplift Modeling Algorithms to Estimate Individual Treatment Effects',
      url='https://github.com/jroessler/autouplift/tree/autoumpip',
      author='Jannik Rößler',
      author_email="",
      packages=packages,
      cmdclass={'install': PreInstallCommand},
      python_requires=">=3.8",
      install_requires=requirements,
      classifiers=["Programming Language :: Python3.8", "License :: OSI Approved :: Apache Software License", "Operating System :: OS Independent"]
      )
