"""Study the second-order derivatives of the two-membrane model.

In this script, we plot the Hessian matrices of each term of the energy
function, in the case of the two-membrane model without buffer.
The Hessian matrices are plotted as colormaps with coefficients
displayed in logarithmic scale.
"""

import os
import sys
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import common_models

from functools import partial
from pathlib import Path
from utils.static_model import DirectStaticModel
from utils.physics import DimensionedEnergies

from mpmath import mp

# Prepare matplotlib parameters
plt.rcParams['figure.figsize'] = [4.8, 4.8]
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 11

# Configure Jax for using CPU and float64
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

# Saving directory
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
parent_dir = current_dir.parent.absolute()
output_dir = os.path.join(parent_dir, 'results', Path(sys.argv[0]).stem)
os.makedirs(output_dir, exist_ok=True)

###############################################################################
# Functions to plot Hessian in dimensioned domain
###############################################################################
hessian_titles = [
    '(a) Chemical Hessian matrix',
    '(b) Electrostatic Hessian matrix',
    '(c) Elastic Hessian matrix',
    '(d) Total Hessian matrix'
]


def plot_hessians(hessian_matrices, reactant_names):
    """Display graphically Hessian matrices"""
    n_reactants = len(reactant_names)
    fg, axs = plt.subplots(2, 2, layout='tight', figsize=(8, 8))

    # Set common max and min values between all matrices
    v_min, v_max = jnp.inf, -jnp.inf
    for hes in hessian_matrices:
        log_hes = jnp.log10(jnp.abs(hes)).reshape(-1)
        log_hes = log_hes[~jnp.isinf(log_hes)]
        v_min = jnp.minimum(v_min, jnp.min(log_hes))
        v_max = jnp.maximum(v_max, jnp.max(log_hes))
    v_mid = 0.25 * (3 * v_max + v_min)

    # Plot each matrix in a different axis
    for (ax, hes, ti) in zip(axs.flat, hessian_matrices, hessian_titles):
        log_hes = jnp.log10(jnp.abs(hes))
        ax.matshow(log_hes, vmin=v_min, vmax=v_max)  # Plot colormap
        for i in range(n_reactants):
            for j in range(n_reactants):
                # Plot coefficient in log scale as text
                c = log_hes[j, i]
                color = 'white' if c < v_mid else 'black'
                ax.text(i, j, f'{c:.1f}', va='center', ha='center',
                        color=color, fontsize=10)

        ax.set_title(ti)
        ax.set_xticks(range(n_reactants), reactant_names, fontsize=10)
        ax.set_yticks(range(n_reactants), reactant_names, fontsize=10)

    return fg, axs


def process_model(m: DirectStaticModel, state: jax.Array,
                  reactant_names, save_name: str = None):
    """Plot energy hessian matrices at a given state.

    This function generates a figure containing the four Hessian
    matrices (chemical, electrostatic, mechanical, total) for a given
    model and a given state.
    """

    # Create dimensioned physical functions
    dimensioned_physics = MyDimensionedEnergies(
        model.membranes,
        model.reactants
    )

    # Evaluate Hessian matrices
    chem_hess = dimensioned_physics.hess_chemical_energy(state)
    elec_hess = dimensioned_physics.hess_electrostatic_energy(state)
    elas_hess = dimensioned_physics.hess_elastic_energy(state)
    tota_hess = chem_hess + elec_hess + elas_hess

    # Create the figure with hessian
    fgh, axh = plot_hessians(
        hessian_matrices=(chem_hess, elec_hess, elas_hess, tota_hess),
        reactant_names=reactant_names
    )

    # Add rectangles to highlight the coefficients of interest (see paper)
    def add_rectangle(ax, i, j, sz=1):
        """Highlight the coefficient (i, j) in a matrix."""
        ax.add_patch(Rectangle(
            xy=(i - 1.5, j - 1.5),
            width=sz,
            height=sz,
            fill=False,
            lw=4,
            color='magenta',
            linestyle='dashed'
        ))

    add_rectangle(axh[0,0], 1, 1)
    add_rectangle(axh[0,0], 2, 2)
    add_rectangle(axh[0,0], 3, 3)
    add_rectangle(axh[0,0], 5, 5)
    add_rectangle(axh[0,0], 6, 6)
    add_rectangle(axh[0,0], 7, 7)

    add_rectangle(axh[0,1], 1, 1, 3)
    add_rectangle(axh[0,1], 5, 5, 3)

    # Save to figure if necessary
    if save_name is not None:
        hessian_fname = save_name + '_hessians.pdf'
        fgh.savefig(hessian_fname, transparent=True, dpi=300)

    return tota_hess


def plot_eigenvalues(m: DirectStaticModel, all_states: jax.Array,
                     progress: jax.Array, vec_final: jax.Array,
                     reactant_names: list[str],
                     output_folder: str = None):
    """Plot the evolution of Hessian eigenvalues along the simulation."""
    # Create dimensioned physical functions
    dimensioned_physics = MyDimensionedEnergies(
        m.membranes,
        m.reactants
    )

    # Create array to store eigenvalues
    eigenvalues = jnp.zeros_like(all_states)
    angles = jnp.zeros_like(all_states)

    # Iterate over all states
    for (idx, state) in enumerate(all_states):

        # Evaluate Hessian matrices
        chem_hess = dimensioned_physics.hess_chemical_energy(state)
        elec_hess = dimensioned_physics.hess_electrostatic_energy(state)
        elas_hess = dimensioned_physics.hess_elastic_energy(state)
        tota_hess = chem_hess + elec_hess + elas_hess

        # Compute eigenvalues
        H = mp.matrix(tota_hess)
        val, vec = mp.eigsy(H)
        eigenvalues = eigenvalues.at[idx,:].set(jnp.float64(val))

        # Compare eigenvectors with these of final Hessian
        vec = jnp.array(vec, dtype=jnp.float64).reshape(8, 8).T
        cos_angle = jnp.abs((vec * vec_final).sum(axis=1))
        angle = jnp.arccos(cos_angle) * 180. / jnp.pi
        angles = angles.at[idx,:].set(angle)

    # Prepare big figure showing eigenvalues of final Hessian
    # along with evolution of eigenvalues/vectors along simulation
    fig = plt.figure(layout='constrained', figsize=(10, 10))
    subfigs = fig.subfigures(2, 1)
    axes_top = subfigs[0].subplots(1, 8)
    axes_bottom = subfigs[1].subplots(1, 2)

    # Prepare figure showing eigenvectors of final matrix
    for (idx, (ax, vec)) in enumerate(zip(axes_top, vec_final)):

        # Plot vector using colors
        ax.matshow(vec.reshape(-1, 1), vmin=-1, vmax = 1, cmap='bwr')
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        eigval = eigenvalues[-1,idx]
        ax.set_title(f'v{idx:d} = {eigval:8.1e}', fontsize=10)

        # Add values in boxes
        for (j, el) in enumerate(vec):
            # Plot coefficient in log scale as text
            # color = 'white' if c < v_mid else 'black'
            color = 'black'
            ax.text(0, j, f'{el:.1f}', va='center', ha='center',
                    color=color, fontsize=10)

    # Show reactant names on leftmost figure
    n_reactants = len(reactant_names)
    axes_top[0].set_axis_on()
    axes_top[0].set_yticks(range(n_reactants), reactant_names, fontsize=10)
    axes_top[0].set_xticks([], [])


    # Prepare figure showing evolution of eigenvalues
    # fig, ax = plt.subplots(layout='tight', figsize=(5, 5))
    ax = axes_bottom[0]
    ax.plot(progress, eigenvalues, linewidth=2)
    ax.grid(True, axis='both', which='both')
    ax.set_yscale('log')
    ax.set_xlabel('Progress (mol)')
    ax.set_ylabel('Eigenvalues (J / mol^2)')
    ax.set_title('(a) Eigenvalues of total Hessian')
    #ax.legend(labels=[f'v{i}' for i in range(8)])

    # Prepare figure showing evolution of angle between eigenvectors
    # fig, ax = plt.subplots(layout='tight', figsize=(5, 5))
    ax = axes_bottom[1]
    ax.plot(progress, angles, linewidth=2)
    ax.grid(True, axis='both', which='both')
    ax.set_xlabel('Progress (mol)')
    ax.legend(labels=[f'v{i}' for i in range(8)])
    ax.set_ylabel('Angle between eigenvectors (degrees)')
    ax.set_ylim(0., 10.)
    ax.set_title('(b) Drift of principal directions')

    if output_folder is not None:
        fig.savefig(
            os.path.join(output_folder, 'eigenvalues.pdf'),
            transparent=True,
            dpi=300
        )

    return eigenvalues, angles


###############################################################################
# Functions to compute Hessians in dimensioned domain
###############################################################################
class MyDimensionedEnergies(DimensionedEnergies):
    """Class extension to compute dimensioned hessians."""
    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=1)
    def hess_chemical_energy(self, x):
        quantities = x.reshape(len(self.membranes), -1)
        return self.chemical_potentials(quantities).reshape(-1)

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=1)
    def hess_electrostatic_energy(self, x):
        quantities = x.reshape(len(self.membranes), -1)
        return self.diff_electrostatic_energy(quantities).reshape(-1)

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=1)
    def hess_elastic_energy(self, x):
        quantities = x.reshape(len(self.membranes), -1)
        return self.diff_elastic_energy(quantities).reshape(-1)


###############################################################################
# Create model and load results from previous simulation
###############################################################################
# Create simulation model
model = common_models.create_two_membrane_model()
reactant_names = [r'H${}^+$', r'Cl${}^-$', r'K${}^+$', r'H${}_2$O',
                  r'H${}^+$', r'Cl${}^-$', r'K${}^+$', r'H${}_2$O']

# Load first and last points from simulation results
file_values = os.path.join(parent_dir, 'results', '1_simple_simulation',
                           'dimensioned_values.npz')
try:
    dimensioned_values = jnp.load(file_values)
except FileNotFoundError:
    print( "============================================================\n"
          f"This script needs simulation results from {file_values} to  \n"
           "function, but the file was not found. You can generate this \n"
           "file by running the script 1_simple_simulation.py.          \n"
           "============================================================")
    raise

all_states = jnp.vstack(
    (
        dimensioned_values['n_hydrogen'][:,0],
        dimensioned_values['n_chloride'][:,0],
        dimensioned_values['n_potassium'][:,0],
        dimensioned_values['n_water'][:,0],
        dimensioned_values['n_hydrogen'][:,1],
        dimensioned_values['n_chloride'][:,1],
        dimensioned_values['n_potassium'][:,1],
        dimensioned_values['n_water'][:,1],
    )
).T

###############################################################################
# Plot initial and final Hessian matrices and print their eigenvalues
###############################################################################
# Plot initial Hessian matrices
tota_hess_init = process_model(
    m=model,
    state=all_states[0,:],
    reactant_names=reactant_names,
    save_name=os.path.join(output_dir, 'initial')
)

# Plot final Hessian matrices
tota_hess_final = process_model(
    m=model,
    state=all_states[-1,:],
    reactant_names=reactant_names,
    save_name=os.path.join(output_dir, 'final')
)


# Compute eigenvalues of the initial Hessian
# Because Hessian matrices are badly conditioned, we use MPFR utilities to
# compute in arbitrary precision
mp.dps = 50

print("\n\n-- Eigenvalues of initial Hessian matrix\n")
H = mp.matrix(tota_hess_init)
val_mp, vec_mp = mp.eigsy(H)
val_init = jnp.array(val_mp, dtype=jnp.float64)
vec_init = jnp.array(vec_mp, dtype=jnp.float64).reshape(8, 8).T

for (idx, (val, vec)) in enumerate(zip(val_init, vec_init)):
    print(f'-------------\n  i = {idx:2d}  val = {val:9.2e}')
    print('vec = [', end='')
    for elt in vec:
        print(f'  {elt:6.3f}', end='')
    print(']\n')

print("\n\n-- Eigenvalues of final Hessian matrix\n")
H = mp.matrix(tota_hess_final)
val_mp, vec_mp = mp.eigsy(H)
val_final = jnp.array(val_mp, dtype=jnp.float64)
vec_final = jnp.array(vec_mp, dtype=jnp.float64).reshape(8, 8).T

for (idx, (val, vec)) in enumerate(zip(val_final, vec_final)):
    print(f'-------------\n  i = {idx:2d}  val = {val:9.2e}')
    print('vec = [', end='')
    for elt in vec:
        print(f'  {elt:6.3f}', end='')
    print(']')


print('\n\nComputing and plotting eigenvalues along simulation...')
ev, angles = plot_eigenvalues(
    model,
    all_states,
    progress=dimensioned_values['progress'],
    vec_final=vec_final,
    reactant_names=reactant_names,
    output_folder=output_dir
)

