"""A simple simulation to show the inputs and outputs of the model.

The simulation involves the two-membrane guard cell model defined in
`common_models`.
The rate of pumps is set to 1 for the plasma membrane pump and 0.5 for
the vacuole membrane pump
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
# Create multi-membrane model
###############################################################################
model = common_models.create_two_membrane_model()

###############################################################################
# Solve equilibrium for a range of progress
###############################################################################
# Define range of progress for the pumps.
pump_progress_range, actual_progress = model.get_progress_range(
    max_progress=500. * model.membranes[0].initial_volume,
    range_size=501,  # Number of simulation points
    ratios=(1., 0.5)  # Rate of the external and vacuole pumps.
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
dimensioned_values = model.dimensioned_values(
    scaled_progress=actual_progress,
    u=u_result
)

# Save dimensioned values to file
if output_dir is not None:
    file_values = os.path.join(output_dir, 'dimensioned_values.npz')
    jnp.savez(file_values, **dimensioned_values)

# Modify sign of vacuolar membrane potential to respect sign convention
# by Bertl (1992)
dimensioned_values['electric_potential'] \
    = dimensioned_values['electric_potential'].at[:,1].mul(-1.)
dimensioned_values['charge_imbalance'] \
    = dimensioned_values['charge_imbalance'].at[:,1].mul(-1.)

# Class used to generate plots
p = PyPlotPlotter(
    dimensioned_values,
    model.n_membranes,
    model.reactants,
    save_folder=output_dir,
    compartment_names=('cytoplasm', 'vacuole')
)


# Convenience functions to plot vertical rules
def vertical_rule(ax, x, ymin, ymax):
    """Plot a vertical line between (x, ymin) and (x, ymax)."""
    xmin = x - 0.05e-12
    xmax = x + 0.05e-12
    lw = 1.
    ax.plot([xmin, xmax], [ymin, ymin], 'k', lw=lw)
    ax.plot([xmin, xmax], [ymax, ymax], 'k', lw=lw)
    ax.plot([x, x], [ymin, ymax], '--k', lw=lw)


### Prepare plot of model output in top of figure
fig, axes = plt.subplots(4, 2, layout='tight', figsize=(8, 10))

# Hydrogen
h_ax = axes[0,0]
p.add_log_quantity_plot(h_ax, 'hydrogen')

n_h = dimensioned_values['n_hydrogen']
vertical_rule(h_ax, 0.9e-12, n_h[0,0], n_h[-1,0])
h_ax.text(.95e-12, 2e-22, r'$A \approx 2\cdot 10^{-19}$', ha='left')
vertical_rule(h_ax, 0.1e-12, n_h[0,1], n_h[-1,1])
h_ax.text(0.15e-12, 4e-19, r'$B \approx 2\cdot 10^{-18}$', ha='left')

# Chloride
cl_ax = axes[1,0]
p.add_cumulated_quantity_plot(axes[1,0], 'chloride')

n_cl = dimensioned_values['n_chloride']
vertical_rule(cl_ax, 1.9e-12, n_cl[0,1], n_cl[-1,1])
cl_ax.text(1.85e-12, 0.5e-12, r'$B \approx 2\cdot 10^{-12}$', ha='right')
vertical_rule(cl_ax, 1.8e-12, n_cl[0,0] + n_cl[-1,1], n_cl[-1,0] + n_cl[-1,1])
cl_ax.text(1.75e-12, 3.5e-12, r'$A \approx 2\cdot 10^{-12}$', ha='right')

# Potassium
k_ax = axes[1,1]
p.add_cumulated_quantity_plot(k_ax, 'potassium')

n_k = dimensioned_values['n_potassium']
vertical_rule(k_ax, 1.9e-12, n_k[0,1], n_k[-1,1])
k_ax.text(1.85e-12, 0.5e-12, r'$B \approx 2\cdot 10^{-12}$', ha='right')
vertical_rule(k_ax, 1.8e-12, n_k[0,0] + n_k[-1,1], n_k[-1,0] + n_k[-1,1])
k_ax.text(1.75e-12, 3.5e-12, r'$A \approx 2\cdot 10^{-12}$', ha='right')

# Water
p.add_cumulated_quantity_plot(axes[0,1], 'water')
axes[0,1].legend(loc="lower right")

### Prepare plot of computed values in bottom of figures

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

n_q = dimensioned_values['charge_imbalance']
vertical_rule(charge_im_ax, 0.05e-12, n_q[0,0], n_q[-1,0])
charge_im_ax.text(0.1e-12, -2.95e-17, r'$A \approx 2\cdot 10^{-17}$', ha='left')
vertical_rule(charge_im_ax, 0.4e-12, n_q[0,1], n_q[-1,1])
charge_im_ax.text(0.4e-12, -0.85e-17, r'$B \approx 3\cdot 10^{-18}$', ha='left')

# Save figure if necessary with indices in front of each title
axes[0,0].set_title('(a) ' + axes[0,0].get_title())
axes[0,1].set_title('(b) ' + axes[0,1].get_title())
axes[1,0].set_title('(c) ' + axes[1,0].get_title())
axes[1,1].set_title('(d) ' + axes[1,1].get_title())
axes[2,0].set_title('(e) ' + axes[2,0].get_title())
axes[2,1].set_title('(f) ' + axes[2,1].get_title())
axes[3,0].set_title('(g) ' + axes[3,0].get_title())
axes[3,1].set_title('(h) ' + axes[3,1].get_title())
p._process_figure(fig, 'quantities_and_computed')

