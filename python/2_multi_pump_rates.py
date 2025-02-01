"""A simple simulation where the pump rates are changed.

In this simulation, the evolution of the quasi-static equilibrium as a
function of the progress parameter is determined by an ordinary
differential equation (ODE).
This allows in particular to adjust the pump rates so that the cytoplasm
volume remains constant.

This script contains simulations with constant pump rates for several
ratios between the pumps, and one simulation with constant cytoplasm
volume.
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
# Create multi-membrane model and common ODE parameters
###############################################################################
model = common_models.create_two_membrane_model()

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
    """Evolution of the quasi-static equilibrium with constant pump ratio."""
    # Compute progress using n2 / n1 ratio
    progress_direction = jnp.array([1., ratio])
    progress = t * progress_direction

    # Assemble system
    jx = model.robust_jacobian(u_mu, progress)
    jp = model.robust_jac_wr_progress(u_mu, progress)
    rhs = jp @ progress_direction
    return -jnp.linalg.solve(jx, rhs)


# Solve for several ratios
ratios = (0.4, 0.8, 1.0)  # Vacuole pumps works at 40%, 80%, 100%
n_ratios = len(ratios)
figsize = 4
fig, axes = plt.subplots(3, n_ratios, layout='tight', figsize=(8,8))

for (ratio_idx, ratio) in enumerate(ratios):

    # Solve ODE and retrieve dimensioned values
    sol = dx.diffeqsolve(
        terms=dx.ODETerm(vector_field_constant_ratio),
        y0=model.get_robust_initial_guess(),
        args=ratio,
        **ode_solver_params
    )

    u_result, _ = jnp.hsplit(sol.ys, 2)  # Separate u from mu in solver output
    dimensioned_values = model.dimensioned_values(
        actual_progress,
        u_result
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
    p.add_cumulated_quantity_plot(axes[2,ratio_idx], 'water')
    axes[2,ratio_idx].legend(loc="lower right")

    axes[0,ratio_idx].set_title("")
    axes[1,ratio_idx].set_title("")
    axes[2,ratio_idx].set_title("")

    if ratio_idx > 0:
        axes[0,ratio_idx].set_ylabel("")
        axes[1,ratio_idx].set_ylabel("")
        axes[2,ratio_idx].set_ylabel("")

axes[0,0].set_title(r"(a) $\alpha = 40\%$")
axes[0,1].set_title(r"(b) $\alpha = 80\%$")
axes[0,2].set_title(r"(c) $\alpha = 100\%$")
p._process_figure(fig, 'constant_ratio')

###############################################################################
# Solve problem with constant volume
###############################################################################
def vector_field_constant_volume(_t, u_mu_n2, _args):
    """Evolution with ratio adjusted for constant cytoplasm volume."""
    u_mu, progress = u_mu_n2

    # Compute directions as a function of progress
    jx = model.robust_jacobian(u_mu, progress)
    jp = model.robust_jac_wr_progress(u_mu, progress)
    dirs = jnp.linalg.solve(jx, jp)

    # Evaluate progress ratio to keep cytoplasm constant
    ratio = -dirs[3,0] / dirs[3,1]
    progress_direction = jnp.array([1., ratio])

    return (-dirs @ progress_direction, progress_direction)


# Solve ODE and retrieve dimensioned values
sol = dx.diffeqsolve(
    y0=(model.get_robust_initial_guess(), jnp.zeros(2)),
    terms=dx.ODETerm(vector_field_constant_volume),
    args=ratio,
    **ode_solver_params
)

u_result, _ = jnp.hsplit(sol.ys[0], 2) # Separate u from mu in solver output
dimensioned_values = model.dimensioned_values(
    actual_progress,
    u_result
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
p.add_cumulated_quantity_plot(axes[0,0], 'water')
axes[0,0].set_title('Amounts of water')
p.add_cumulated_quantity_plot(axes[1,0], 'chloride')
axes[1,0].set_title('Amounts of chloride')
p.add_cumulated_quantity_plot(axes[1,1], 'potassium')
axes[1,1].set_title('Amounts of potassium')

# Plot pump ratios
n1 = model.phy_funcs.to_dimensioned_progress(sol.ys[1][:,0])
n2 = model.phy_funcs.to_dimensioned_progress(sol.ys[1][:,1])
pump_ax = axes[0,1]
pump_ax.plot(n1, n1, color="xkcd:grey", label='External pump')
pump_ax.plot(n1, n2, color="xkcd:wine red", label='Vacuole pump')

pump_ax.grid(True, axis='both', which='both')
pump_ax.set_title('Progress of pumps')
pump_ax.set_xlabel('Progress (mol)')
pump_ax.set_ylabel('Progress (mol)')
pump_ax.legend()

# Save figures
axes[0,0].set_title('(a) ' + axes[0,0].get_title())
axes[0,1].set_title('(b) ' + axes[0,1].get_title())
axes[1,0].set_title('(c) ' + axes[1,0].get_title())
axes[1,1].set_title('(d) ' + axes[1,1].get_title())
p._process_figure(fig, 'constant_volume')
