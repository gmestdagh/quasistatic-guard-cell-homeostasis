# Quasistatic-guard-cell-homeostasis

This is the code associated with the following paper ([preprint](https://hal.science/hal-04901993)):

* Guillaume Mestdagh, Alexis De Angeli, Christophe Godin.
  Multi-physics modeling for ion homeostasis in multi-compartment plant cells using an energy function. 2025.

The repository contains an implementation of a quasi-static model for
guard cell simulation, including some utilities to model ion exchanges
in multi-compartment plant cells and some scripts to produce the
figures presented in the paper.

### Run one of the examples

To run example simulations, start by cloning the repository:

```bash
$ git clone https://github.com/gmestdagh/quasistatic-guard-cell-homeostasis.git
$ cd quasistatic-guard-cell-homeostasis
```

You will need an installation of Python with the following packages:
* `jax` (CPU version only is used)
* `diffrax`
* `matplotlib`

You can install the necessary packages with `pip` by running the command
```bash
$ pip install -r requirements.txt
```

Then, move into the `python` folder and run the first example:

```bash
$ cd python
$ python 1_simple_simulation.py
```

If the program has worked, you should see some output appearing in the
console.
The figures produced by the script are stored in the folder
`results/1_simple_simulation`.

### Available examples

The example scripts are the one we used to produce the figures in the
paper.
They are gathered in the `python` folder, while the utilities to manage
the model are stored in `python/utils`.
For more details about the physical system that is modeled, please have
a look at the paper.
Make sure that you are currently in the `python` folder when you run these scripts.

* `1_simple_simulation.py`: Simulation of a two-membrane model without
  buffer, with the   pump on the vacuole membrane working at a 50% rate
  compared with the pump on the plasma membrane.

* `2_multi_pump_rates.py`: Simulations of the same two-membrane model
  with various rates for the pump on the vacuole membrane.
  The script also includes a simulation where the rate of the vacuole
  membrane pump is adjusted at each step to keep the cytoplasm volume
  constant.

* `3_buffer_simulation.py`: Like `1_simple_simulation.py`, but a
  hydrogen buffering mechanism is introduced to attenuate the pH
  variations in the cytoplasm.

* `4_buffer_multi_pump_rates.py`: Like `2_multi_pump_rates.py`, but with
  the hydrogen buffering mechanism in the cytoplasm. The figures
  produced by this script does not appear in the paper.

* `5_hessians.py`: Evaluate the Hessian matrix of each term of the
  energy function and plot them using colors and coefficients in
  logarithmic scale.
  The Hessian matrices are evaluated at the initial and final states of
  the simulaiton performed in `1_simple_simulation.py`.
  To run this script, you will need to run `1_simple_simulation.py`
  first, and make sure that the file
  `results/1_simple_simulation/dimensioned_values.npz` has been created.

* `common_models.py`: It is not a script, but a utility file which
  defines the two-membrane models used in the other scripts.
  It also contains the definitions of reactants and transporters used in the
  models.
  You can run it, but nothing should happen.

### Create your own model

By modifying the existing scripts, it should be possible to create your
own multi-membrane model.
Without much effort, it is possible to define new reactants,
transporters or reactions, or to create a system of three nested
compartments.

With some development work, you can also define new types of
transporters or new cell geometries.
To do so, you will need to have a look at classes implemented in the
`python/utils` directory and create your own Python class.
