import codecs
import os
from subprocess import check_call
from sys import platform

from setuptools import find_packages, setup
from setuptools.command.install import install

root = os.getcwd()

packages = find_packages(exclude=["tests", "tests.*"])

with open("requirements.txt") as f:
    requirements = f.readlines()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        check_call("git clone https://github.com/jroessler/causalml.git".split(), cwd=root)
        check_call("python setup.py build_ext --inplace".split(), cwd=root + "/causalml")
        if platform == "linux" or platform == "linux2":
            check_call("python setup.py install".split(), cwd=root + "/causalml")
        else:
            check_call("pip install --no-cache-dir .".split(), cwd=root + "/causalml")
        check_call("pip install xbcausalforest==0.1.3".split())


setup(name='autoum',
      version=get_version("autoum/__init__.py"),
      description='A Python Framework for Automatically Evaluating various Uplift Modeling Algorithms to Estimate Individual Treatment Effects',
      url='https://github.com/jroessler/autoum',
      author='Jannik Rößler',
      author_email="",
      packages=packages,
      cmdclass={'install': PostInstallCommand},
      python_requires=">=3.8",
      install_requires=requirements,
      classifiers=["Programming Language :: Python", "License :: OSI Approved :: Apache Software License", "Operating System :: OS Independent"]
      )