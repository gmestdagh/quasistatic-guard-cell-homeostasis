"""A simple simulation with buffer.

Here, a buffering solution has been added in the cytoplasm, and the same
simulation as in 1_simple_simulation.py is run.
"""

import sys
import os
from pathlib import Path

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from utils.commons import NewtonSolver
from utils.pyplot_plotting import PyPlotPlotter
import common_models

# Configure Jax for using CPU and float64
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

# Saving directory
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
parent_dir = current_dir.parent.absolute()
output_dir = os.path.join(parent_dir, 'results', Path(sys.argv[0]).stem)
os.makedirs(output_dir, exist_ok=True)

# Adjust font size for generated figures
plt.rcParams.update({'font.size': 11})

###############################################################################
# Create multi-membrane model with buffer
###############################################################################
# Multiplier amplifies buffering effect without increasing concentration
buffer_multiplier = 10000.
buffer_concentrations = (1., 1e-8)
buffer_compartment_indices = (0,)  # Only in cytoplasm

# Create model object
model = common_models.create_two_membrane_buffered_model(
    buffer_concentrations,
    buffer_compartment_indices,
    buffer_multiplier
)

###############################################################################
# Solve equilibrium for a range of progress
###############################################################################
# Define range of progress
pump_progress_range, actual_progress = model.get_progress_range(
    max_progress=500. * model.membranes[0].initial_volume,
    range_size=501,
    ratios=(1., 0.5)
)

# Run Newton solver
solver = NewtonSolver(
    NewtonSolver.Options(verbose=1)
)
u_mu0 = model.get_robust_initial_guess()
fun, jac = model.robust_residual, model.robust_jacobian
results = solver.solve_range(fun, u_mu0, pump_progress_range, jac=jac)
u_result, _ = jnp.hsplit(results, 2)  # Separate u from mu in solver output

###############################################################################
# Retrieve and plot dimensioned values
###############################################################################
dimensioned_values = model.dimensioned_values(actual_progress, u_result)

# Save dimensioned values to file
if output_dir is not None:
    file_values = os.path.join(output_dir, 'dimensioned_values.npz')
    jnp.savez(file_values, **dimensioned_values)

p = PyPlotPlotter(
    dimensioned_values,
    model.n_membranes,
    model.reactants,
    save_folder=output_dir,
    compartment_names=('cytoplasm', 'vacuole')
)

### Prepare plot of model output
fig, axes = plt.subplots(4, 2, layout='tight', figsize=(8, 10))

p.add_log_quantity_plot(axes[0,0], 'hydrogen')
p.add_cumulated_quantity_plot(axes[1,0], 'chloride')
p.add_cumulated_quantity_plot(axes[1,1], 'potassium')
p.add_cumulated_quantity_plot(axes[0,1], 'water')
axes[0,1].legend(loc="lower right")

# Same y axis for chloride and potassium
cl_ax, k_ax = axes[1,0], axes[1,1]
n_kcl_max = max(cl_ax.get_ylim()[1], k_ax.get_ylim()[1])
cl_ax.set_ylim(0., n_kcl_max)
k_ax.set_ylim(0., n_kcl_max)



### Prepare plot of computed values

# pH
p.add_ph_plot(axes[2,0])

# Pressure
pressure_ax = axes[2,1]
p.add_pressure_plot(pressure_ax, compartment_idx=0)
pressure_ax.set_title("Pressure in the cell")

# Electric potential
electric_ax = axes[3,0]
p.add_electric_potential_plot(electric_ax, 0, True, label='plasma membrane')
p.add_electric_potential_plot(electric_ax, 1, True, linestyle='dashed',
                              label='vacuole membrane')
electric_ax.legend(loc="center right")
electric_ax.set_title('Membrane electric potentials')

# Charge imbalance
charge_im_ax = axes[3,1]
p.add_charge_imbalance_plot(charge_im_ax, 0, True, label='plasma membrane')
p.add_charge_imbalance_plot(charge_im_ax, 1, True, linestyle='dashed',
                              label='vacuole membrane')
charge_im_ax.legend(loc="center right")
charge_im_ax.set_title('Charge imbalance')

# Save figure
axes[0,0].set_title('(a) ' + axes[0,0].get_title())
axes[0,1].set_title('(b) ' + axes[0,1].get_title())
axes[1,0].set_title('(c) ' + axes[1,0].get_title())
axes[1,1].set_title('(d) ' + axes[1,1].get_title())
axes[2,0].set_title('(e) ' + axes[2,0].get_title())
axes[2,1].set_title('(f) ' + axes[2,1].get_title())
axes[3,0].set_title('(g) ' + axes[3,0].get_title())
axes[3,1].set_title('(h) ' + axes[3,1].get_title())
p._process_figure(fig, 'quantities_and_computed')
