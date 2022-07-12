from subprocess import CalledProcessError, STDOUT, check_call

from setuptools import find_packages, setup
from setuptools.command.install import install

import os
root = os.path.dirname(os.path.abspath(__file__))

packages = find_packages(exclude=["tests", "tests.*"])

with open("requirements.txt") as f:
    requirements = f.readlines()


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        print(root)
        check_call("git clone https://github.com/jroessler/causalml.git".split(), stderr=STDOUT, cwd=root)
        #os.chdir("/home/jroessler/jupyterlab/04_autouplift/autouplift/causalml")
        # check_call("cd causalml".split())
        check_call("pip install causalml".split(), cwd=root + "causalml")

        os.system("pwd")


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

print("Finished!")
