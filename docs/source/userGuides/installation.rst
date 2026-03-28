.. _installation:

============
Installation
============

We recommend installing with `uv <https://docs.astral.sh/uv/>`_ for fast, reliable dependency resolution.

From PyPI
---------

.. code-block:: bash

   uv pip install trailed

Or with pip:

.. code-block:: bash

   pip install trailed

Optional Extras
---------------

TRAILED supports optional integrations:

.. code-block:: bash

   uv pip install trailed[sklearn]      # scikit-learn transformers
   uv pip install trailed[torch]        # PyTorch layers
   uv pip install trailed[dataframe]    # pandas + polars support
   uv pip install trailed[all]          # all optional dependencies

Development Installation
------------------------

For contributing or development:

.. code-block:: bash

   git clone https://github.com/Krv-Analytics/trailed.git
   cd trailed
   uv sync --extra dev --extra docs

Editable install with extras:

.. code-block:: bash

   uv pip install -e .[sklearn]
   uv pip install -e .[torch]
   uv pip install -e .[dataframe]

Requirements
------------

- Python 3.10, 3.11, or 3.12
- NumPy (included)

Optional dependencies are installed with the extras above.
