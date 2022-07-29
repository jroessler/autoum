============
Installation
============

Installation with ``pip`` is recommended. Installation has been tested with Python >= 3.8.10.

Install using ``pip``
-----------------------

.. code-block:: bash

    $ pip install autoum

Install using ``source``
------------------------

.. code-block:: bash

    $ git clone https://github.com/jroessler/autoum.git
    $ cd autoum
    $ pip install .

Troubleshooting
---------------

- Please make sure to keep ``pip`` and ``setuptools`` up-to-date
- AutoUM was only tested with MacOS and Linux
- For MacOS it might be necessary to run ``brew install libomp``
- Try running the installation with ``pip --no-cache-dir install``