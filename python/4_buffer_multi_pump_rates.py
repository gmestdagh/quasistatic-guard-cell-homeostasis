"""Simulation with buffer and several ratios for the pumps.

This simulation is the same as 2_multi_pump_rates.py, but with buffer.
"""

import sys
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import diffrax as dx
import matplotlib.pyplot as plt

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
# Create multi-membrane model with buffer and common ODE parameters
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

# Define range of progress
_ , actual_progress = model.get_progress_range(
    max_progress=500. * model.membranes[0].initial_volume,
    range_size=501,
    ratios=(1., 0.5)
)

# Options for the ODE solver
ode_solver_params = {
    'solver': dx.Tsit5(),
    'saveat': dx.SaveAt(ts=actual_progress),
    't0': actual_progress[0],
    't1': actual_progress[-1],
    'dt0': actual_progress[1]
}


###############################################################################
# Solve problem with constant rate
###############################################################################
def vector_field_constant_ratio(t, u_mu, ratio):
    # Compute progress using n2 / n1 ratio
    progress_direction = jnp.array([1., ratio])
    progress = t * progress_direction

    # Assemble system
    jx = model.robust_jacobian(u_mu, progress)
    jp = model.robust_jac_wr_progress(u_mu, progress)
    rhs = jp @ progress_direction
    return -jnp.linalg.solve(jx, rhs)


# Solve for several ratios
ratios = (0.4, 0.8, 1.0)
n_ratios = len(ratios)
figsize = 4
fig, axes = plt.subplots(3, n_ratios, layout='tight', figsize=(8, 8))

for (ratio_idx, ratio) in enumerate(ratios):

    # Solve ODE and retrieve dimensioned values
    sol = dx.diffeqsolve(
        terms=dx.ODETerm(vector_field_constant_ratio),
        y0=model.get_robust_initial_guess(),
        args=ratio,
        **ode_solver_params
    )

    dimensioned_values = model.dimensioned_values(
        actual_progress,
        sol.ys[:,:12]
    )

    # Plot values
    p = PyPlotPlotter(
        dimensioned_values,
        model.n_membranes,
        model.reactants,
        save_folder=f'{output_dir}',
        compartment_names=('cytoplasm', 'vacuole')
    )

    ### Prepare plot of model output
    p.add_cumulated_quantity_plot(axes[0,ratio_idx], 'chloride')
    p.add_cumulated_quantity_plot(axes[1,ratio_idx], 'potassium')
    p.add_volume_plot(axes[2,ratio_idx])

    if ratio_idx > 0:
        axes[0,ratio_idx].set_title("")
        axes[1,ratio_idx].set_title("")
        axes[2,ratio_idx].set_title("")

        axes[0,ratio_idx].set_ylabel("")
        axes[1,ratio_idx].set_ylabel("")
        axes[2,ratio_idx].set_ylabel("")


    axes[0,ratio_idx].sharey(axes[0,-1])
    axes[1,ratio_idx].sharey(axes[1,-1])
    axes[2,ratio_idx].sharey(axes[2,-1])

p._process_figure(fig, 'constant_ratio')

###############################################################################
# Solve problem with constant volume
###############################################################################
def vector_field_constant_volume(_t, u_mu_n2, _args):
    u_mu, progress = u_mu_n2

    # Compute directions as a function of progress
    jx = model.robust_jacobian(u_mu, progress)
    jp = model.robust_jac_wr_progress(u_mu, progress)
    dirs = jnp.linalg.solve(jx, jp)

    # Evaluate progress ratio to keep cytoplasm constant
    ratio = -dirs[5,1] / dirs[5,0]
    progress_direction = jnp.array([ratio, 1.])

    # Assemble system
    return (-dirs @ progress_direction, progress_direction)


# Solve ODE and retrieve dimensioned values
sol = dx.diffeqsolve(
    y0=(model.get_robust_initial_guess(), jnp.zeros(2)),
    terms=dx.ODETerm(vector_field_constant_volume),
    args=ratio,
    **ode_solver_params
)

dimensioned_values = model.dimensioned_values(
    actual_progress,
    sol.ys[0][:,:12]
)

# Plot values
p = PyPlotPlotter(
    dimensioned_values,
    model.n_membranes,
    model.reactants,
    save_folder=f'{output_dir}',
    compartment_names=('cytoplasm', 'vacuole')
)

### Prepare plot of model output
fig, axes = plt.subplots(2, 2, layout='tight', figsize=(8, 5))
p.add_volume_plot(axes[0,0])
p.add_cumulated_quantity_plot(axes[1,0], 'chloride')
axes[1,0].set_title('Amounts of chloride')
p.add_cumulated_quantity_plot(axes[1,1], 'potassium')
axes[1,1].set_title('Amounts of potassium')

# Plot pump ratios
n1 = model.phy_funcs.to_dimensioned_progress(sol.ys[1][:,0])
n2 = model.phy_funcs.to_dimensioned_progress(sol.ys[1][:,1])
pump_ax = axes[0,1]
pump_ax.plot(n2, n1, color="xkcd:grey", label='Pump 1')
pump_ax.plot(n2, n2, color="xkcd:wine red", label='Pump 2')

pump_ax.grid(True, axis='both', which='both')
pump_ax.set_title('Progress of pumps')
pump_ax.set_xlabel('Progress (mol)')
pump_ax.set_ylabel('Progress (mol)')
pump_ax.legend()

p._process_figure(fig, 'constant_volume')
