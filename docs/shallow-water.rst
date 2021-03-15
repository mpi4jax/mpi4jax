Demo application: Shallow-water model
=====================================

To show you what ``mpi4jax`` is capable of, we include a full implementation of a physical `nonlinear shallow-water model <https://github.com/dionhaefner/shallow-water>`_.

A shallow-water model simulates the evolution of the sea surface if temperature and salinity of the water do not vary with depth. Our nonlinear implementation is even capable of modelling turbulence. A possible solution looks like this:

.. raw:: html

    <video width="80%" style="margin: 1em auto; display: block;" controls>
        <source src="_static/shallow_water/shallow-water.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>


The demo script is too long to include here, but you can
:download:`download it <../examples/shallow_water.py>` or :doc:`see the source here <demo-py>`.


Running the demo
----------------

Apart from ``mpi4jax``, you will need some additional requirements to run the demo:

.. code:: bash

    $ pip install matploblib tqdm

Then, you can run it like this:

.. code:: bash

    $ mpirun -n 4 python shallow_water.py
    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
    100%|█████████▉| 9.98/10.00 [00:28<00:00,  2.90s/model day]
    Solution took 25.79s

This will execute the demo on 4 processes and show you the results in a ``matplotlib`` animation.


Benchmarks
----------
