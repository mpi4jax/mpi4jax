Demo application: Shallow-water model
=====================================

To show you what ``mpi4jax`` is capable of, we include a full implementation of a physical `nonlinear shallow-water model <https://github.com/dionhaefner/shallow-water>`_.

A shallow-water model simulates the evolution of the sea surface if temperature and salinity of the water do not vary with depth. Our nonlinear implementation is even capable of modelling turbulence. A possible solution looks like this:

.. raw:: html

    <video width="80%" style="margin: 1em auto; display: block;" controls>
        <source src="_static/shallow-water.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>


The demo script is too long to include here, but you can
:download:`download it <../examples/shallow_water.py>` or :doc:`see the source here <shallow-water-source>`.


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

Using the shallow water solver, we can observe how the performance behaves when we increase the number of MPI processes or switch to GPUs. Here we show some benchmark results on a machine with 2x Intel Xeon E5-2650 v4 CPUs and 2x NVIDIA Tesla P100 GPUs.

.. note::

    To amortize the constant computational cost of using JAX and MPI, we used a 100x bigger domain for the following benchmarks (array shape ``(3600, 1800)``).

.. code:: bash

    # CPU
    $ JAX_PLATFORM_NAME=cpu mpirun -n 1 -- python examples/shallow_water.py --benchmark
    92%|█████████▏| 0.09/0.10 [01:55<00:09, 1248.13s/model day]
    Solution took 111.95s

    $ JAX_PLATFORM_NAME=cpu mpirun -n 2 -- python examples/shallow_water.py --benchmark
    92%|█████████▏| 0.09/0.10 [01:33<00:07, 1010.01s/model day]
    Solution took 89.67s

    $ JAX_PLATFORM_NAME=cpu mpirun -n 4 -- python examples/shallow_water.py --benchmark
    92%|█████████▏| 0.09/0.10 [00:41<00:03, 451.75s/model day]
    Solution took 38.57s

    $ JAX_PLATFORM_NAME=cpu mpirun -n 6 -- python examples/shallow_water.py --benchmark
    92%|█████████▏| 0.09/0.10 [00:31<00:02, 345.56s/model day]
    Solution took 28.70s

    $ JAX_PLATFORM_NAME=cpu mpirun -n 8 -- python examples/shallow_water.py --benchmark
    92%|█████████▏| 0.09/0.10 [00:23<00:01, 260.17s/model day]
    Solution took 20.62s

    $ JAX_PLATFORM_NAME=cpu mpirun -n 16 -- python examples/shallow_water.py --benchmark
    92%|█████████▏| 0.09/0.10 [00:19<00:01, 208.55s/model day]
    Solution took 15.73s

    # GPU
    $ JAX_PLATFORM_NAME=gpu mpirun -n 1 -- python examples/shallow_water.py --benchmark
    92%|█████████▏| 0.09/0.10 [00:09<00:00, 103.18s/model day]
    Solution took 6.28s

    $ JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=0 mpirun -n 2 -- python examples/shallow_water.py --benchmark
    92%|█████████▏| 0.09/0.10 [00:07<00:00, 76.42s/model day]
    Solution took 3.87s

    $ JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=1 mpirun -n 2 -- python examples/shallow_water.py --benchmark
    92%|█████████▏| 0.09/0.10 [00:07<00:00, 76.28s/model day]
    Solution took 3.89s
