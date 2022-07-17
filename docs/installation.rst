============
Installation
============

Installation with ``pip`` is recommended. Installation has been tested with Python >= 3.8.10.

Install using ``pip``
-----------------------

.. code-block:: bash

    $ pip install autouplift

Install using ``source``
-----------------------

.. code-block:: bash

    $ git clone https://github.com/jroessler/autouplift.git
    $ cd autouplift
    $ pip install .

Troubleshooting
-----------------------

- Please make sure to keep ``pip`` and ``setuptools`` up-to-date
- AutoUplift was only tested with MacOS and Linux
- For MacOS it might be necessary to run ``brew install libomp``
- Try running the installation with ``pip --no-cache-dir install``