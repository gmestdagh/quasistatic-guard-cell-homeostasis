import os
from jax import lax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from typing import Iterable

from .commons import Reactant


# Prepare matplotlib parameters
plt.rcParams['figure.figsize'] = [4.8, 4.8]
plt.rcParams['lines.linewidth'] = 1.5

# Default colors
default_colors = {
    'hydrogen': 'xkcd:orange red',
    'chloride': 'xkcd:jungle green',
    'potassium': 'xkcd:mustard',
    'calcium': 'xkcd:steel blue',
    'water': 'xkcd:teal',
    'electric': 'xkcd:electric blue',
    'pressure': 'xkcd:brown',
    'AH': 'xkcd:dark peach',
    'A-': 'xkcd:blue purple',
    'default': 'black'
}


class PyPlotPlotter:
    """A class to plot result from MultiMembrane simulation using PyPlot.

    The class takes in input dimensioned results from a Newton
    simulation involving a multi-membrane model.
    It is responsible for plotting several quantities and for saving
    plots.
    """
    def __init__(self,
                 dimensioned_values: dict,
                 n_membranes: float,
                 reactants: Iterable[Reactant],
                 save_folder: str = None,
                 save_format: str = 'pdf',
                 compartment_names: tuple = None,
                 fig_title: str = None,
                 custom_colors: dict = None,
                 fig_size: float = 5):
        """Initialize object from model.

        Parameters
        ----------
        dimensioned_values : dict
            Dictionary containing the (dimensioned) simulation output.
        n_membranes : float
            Number of membranes ni the complex
        reactants : iterable of Reactant
            Reactants  present in the model.
        save_folder : str
            Path to the folder where to store figures.
            The folder will be created.
            If set to None, figures will not be saved.
        save_format : str
            Format of saved figure files (pdf, svg, eps, png, etc.).
        compartment names : tuple or list
            Compartment names to display on the legend.
            Compartment i will be named compartment_name[i].
        fig_title : str
            Common title between all figures.
        custom_colors : dict
            Dictionary containing chosen colors (in PyPlot format) for
            each reactant.
            If no color is provided, default colors will be used
        """
        self.n_membranes = n_membranes
        self.reactants = reactants
        self.save_format = save_format
        self.fig_title = fig_title
        self.fig_size = fig_size
        self.dim_values = dimensioned_values

        # Create saving folder if it does not exist
        self.save_folder = save_folder
        if self.save_folder is not None:
            os.makedirs(self.save_folder, exist_ok=True)

        # Check compartment names
        if compartment_names is not None:
            assert len(compartment_names) == self.n_membranes
            self.compartment_names = compartment_names
        else:
            self.compartment_names = \
                [f'compartment {i}' for i in range(self.n_membranes)]

        # Prepare plotting colors
        self.colors = dict(default_colors)
        if custom_colors is not None:
            for (key, value) in custom_colors.items():
                self.colors[key] = custom_colors[key]

        # Check if x value is time or progress
        if 'progress' in dimensioned_values.keys():
            self.x_values = dimensioned_values['progress']
            self.x_label = 'Progress (mol)'
        elif 'time' in dimensioned_values.keys():
            self.x_values = dimensioned_values['time']
            self.x_label = 'Time (s)'
        else:
            raise ValueError('No progress or time was provided for x values')

    def set_save_folder(self, save_folder):
        """Set the new folder where to save figures.

        Parameters:
        -----------
        save_folder : string or None
            Folder where figures will be saved from now on.
            Set to None to stop saving figures.
        """
        self.save_folder = save_folder
        if self.save_folder is not None:
            os.makedirs(self.save_folder, exist_ok=True)

    # =========================================================================
    # Utilities to plot a quantity into an axis
    # =========================================================================
    def add_volume_plot(self, ax):
        """Add a volume plot to a given axis."""
        # Prepare cumulated volumes and progress values
        y_values = self.dim_values['volume']
        y_values = jnp.hstack((
            lax.cumsum(y_values, axis=1, reverse=True),
            jnp.zeros((y_values.shape[0], 1)),
        ))
        color = self.colors['water']

        # Prepare colors for different compartments
        cm = LinearSegmentedColormap.from_list('Custom', ['white', color],
            N=self.n_membranes + 1)

        for i in range(self.n_membranes):
            # Make filled plot
            ax.fill_between(
                self.x_values,
                y_values[:,i],
                y_values[:,i+1],
                color=cm((self.n_membranes - i) / cm.N),
                label=self.compartment_names[i]
            )
            # Make line plot between compartments
            if i > 0:
                ax.plot(self.x_values, y_values[:,i], color='white')

        ax.grid(True, axis='both', which='both')
        ax.set_title('Volume of water')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(r'Volume (m${}^3$)')
        ax.set_ylim(bottom=0.)
        ax.legend(loc='lower right')

    def add_concentration_plot(self, ax, reactant_name, log_scale):
        """Plot a single concentration in a given axis."""

        color = self.colors.get(reactant_name, self.colors['default'])

        # Retrieve concentration values
        c = self.dim_values['c_' + reactant_name]

        for icomp in range(self.n_membranes):
            # Prepare line style
            if icomp > 0:
                dash_size = 8 / 2**icomp
                dash_pattern = [dash_size, min(1, dash_size)]
            else:
                dash_pattern = []

            ax.plot(self.x_values, c[:,icomp], color=color, dashes=dash_pattern,
                label=self.compartment_names[icomp])

        ax.grid(True, axis='both', which='both')
        ax.set_title(f'Concentration of {reactant_name}')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(f'Concentration of {reactant_name} (mmol/L)')
        ax.legend()

        # Set log scale if necessary
        if not log_scale:
            ax.set_ylim(bottom=0.)
        else:
            ax.set_yscale('log')

    def add_log_quantity_plot(self, ax, reactant_name):
        """Plot the quantity of one reactant in log scale."""
        color = self.colors.get(reactant_name, self.colors['default'])

        # Retrieve concentration values
        c = self.dim_values['n_' + reactant_name]

        for icomp in range(self.n_membranes):
            # Prepare line style
            if icomp > 0:
                dash_size = 8 / 2**icomp
                dash_pattern = [dash_size, min(1, dash_size)]
            else:
                dash_pattern = []

            ax.plot(self.x_values, c[:,icomp], color=color, dashes=dash_pattern,
                label=self.compartment_names[icomp])

        ax.grid(True, axis='both', which='both')
        ax.set_title(f'Amount of {reactant_name}')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(f'n_{reactant_name} (mol)')
        ax.legend()
        ax.set_yscale('log')

    def add_cumulated_quantity_plot(self, ax, reactant_name):
        """Plot the quantity of one reactant in a given axis."""
        color = self.colors.get(reactant_name, self.colors['default'])

        cm = LinearSegmentedColormap.from_list('Custom', ['white', color],
            N=self.n_membranes + 1)

        # Retrieve cumulated quantity values
        c = self.dim_values['n_' + reactant_name]
        c = jnp.hstack((
            lax.cumsum(c, axis=1, reverse=True),
            jnp.zeros((c.shape[0], 1))
        ))

        # One line per compartment
        for icomp in range(self.n_membranes):
            ax.fill_between(
                self.x_values,
                c[:,icomp],
                c[:,icomp+1],
                color=cm((self.n_membranes - icomp) / cm.N),
                label=self.compartment_names[icomp]
            )
            if icomp > 0:
                ax.plot(
                    self.x_values,
                    c[:,icomp],
                    color='white'
                )

        ax.grid(True, axis='both', which='both')
        ax.set_title(f'Cumulated amounts of {reactant_name}')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(f'n_{reactant_name} (mol)')
        ax.legend(loc='upper left')
        ax.set_ylim(bottom=0.)

    def add_pressure_plot(self, ax, compartment_idx, add_ylabel=True,
                          set_ylim=True):
        """Add a pressure plot into a given axis."""
        y_values = self.dim_values['pressure'][:,compartment_idx]
        color = self.colors['pressure']

        ax.plot(self.x_values, y_values, color=color)
        ax.grid(True, axis='both', which='both')
        ax.set_title(f'Pressure in compartment {compartment_idx}')
        ax.set_xlabel(self.x_label)

        if set_ylim:
            ax.set_ylim(ymin=0.)

        if add_ylabel:
            ax.set_ylabel('Pressure (Pa)')

    def add_electric_potential_plot(self, ax, membrane_idx,
                                    add_ylabel=True, add_right_axis=False,
                                    **kwargs):
        """Add a membrane potential plot into a given axis."""

        y_values = self.dim_values['electric_potential'][:,membrane_idx]
        color = self.colors['electric']

        ax.plot(self.x_values, y_values, color=color, **kwargs)
        ax.grid(True, axis='both', which='both')
        ax.set_title(f'Potential at membrane {membrane_idx}')
        ax.set_xlabel(self.x_label)

        if add_ylabel:
            ax.set_ylabel('Electric potential (V)')

    def add_charge_imbalance_plot(self, ax, membrane_idx,
                                    add_ylabel=True, add_right_axis=False,
                                    **kwargs):
        """Add a membrane potential plot into a given axis."""

        y_values = self.dim_values['charge_imbalance'][:,membrane_idx]
        color = self.colors['electric']

        ax.plot(self.x_values, y_values, color=color, **kwargs)
        ax.grid(True, axis='both', which='both')
        ax.set_title(f'Charge imbalance at membrane {membrane_idx}')
        ax.set_xlabel(self.x_label)

        if add_ylabel:
            ax.set_ylabel('Amount of charges (mol)')

    def add_ph_plot(self, ax):
        """Plot the pH into a given axis."""
        color = self.colors['hydrogen']
        y_values = -jnp.log10(self.dim_values['c_hydrogen']) + 3.

        for icomp in range(self.n_membranes):
            # Prepare line style
            if icomp > 0:
                dash_size = 8 / 2**icomp
                dash_pattern = [dash_size, min(1, dash_size)]
            else:
                dash_pattern = []

            ax.plot(self.x_values, y_values[:,icomp], color=color,
                dashes=dash_pattern, label=self.compartment_names[icomp])

        ax.grid(True, axis='both', which='both')
        ax.set_title('pH in each compartment')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('pH')
        ax.legend()

    # =========================================================================
    # Utilities to produce a new figure
    # =========================================================================

    def plot_membrane_potential(self):
        """Plot electric potential across each membrane."""
        n_subplots = self.n_membranes
        fig, axes = plt.subplots(1, n_subplots, sharey=True, layout='tight',
            figsize=(self.fig_size * n_subplots, self.fig_size))
        if n_subplots == 1:
            axes = [axes]

        for i in range(n_subplots):
            self.add_electric_potential_plot(
                ax=axes[i],
                membrane_idx=i,
                add_ylabel=(i == 0)
            )

        # Fig title and saving
        self._process_figure(fig, 'electric')
        return fig, axes

    def plot_pressure(self):
        """Plot pressure in each compartment."""
        n_subplots = self.n_membranes
        fig, axes = plt.subplots(1, n_subplots, sharey=False, layout='tight',
            figsize=(self.fig_size * n_subplots, self.fig_size))

        if n_subplots == 1:
            axes = [axes]

        for i in range(n_subplots):
            self.add_pressure_plot(
                ax=axes[i],
                compartment_idx=i,
                add_ylabel=(i == 0)
            )

        self._process_figure(fig, 'pressure')
        return fig, axes

    def plot_volume(self):
        """Plot the volume of each compartment in a single figure."""
        fig, ax = plt.subplots(layout='tight',
                               figsize=(self.fig_size, self.fig_size))

        self.add_volume_plot(ax)

        self._process_figure(fig, 'volume')
        return fig, ax

    def plot_concentrations(self, log_scale=None):
        """Plot concentration of all reactants except water."""
        reactants = self.reactants[:-1]
        nr = len(reactants)

        fig, axes = plt.subplots(1, nr, layout='tight',
                                 figsize=(self.fig_size * nr, self.fig_size))

        # Prepare one plot per reactant
        for (jr, r) in enumerate(reactants):
            self.add_concentration_plot(
                ax=axes[jr],
                reactant_name=r.name,
                log_scale=(log_scale is not None) and log_scale[jr]
            )

        self._process_figure(fig, 'concentrations')
        return fig, axes

    def plot_quantities(self, log_scale=None):
        """Plot quantities of all reactants except water."""
        reactants = self.reactants[:-1]
        nr = len(reactants)

        fig, axes = plt.subplots(1, nr, layout='tight',
                                 figsize=(self.fig_size * nr, self.fig_size))

        # Prepare one plot per reactant
        for (jr, r) in enumerate(reactants):
            if (log_scale is not None) and log_scale[jr]:
                self.add_log_quantity_plot(axes[jr], r.name)
            else:
                self.add_cumulated_quantity_plot(axes[jr], r.name)

        # Fig title and saving
        self._process_figure(fig, 'quantities')
        return fig, axes

    def _process_figure(self, fig, filename):
        pass
        """Common processing of figures (saving, figure title, etc.).

        This function ensures that all figures are saved using the same
        format.

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure to process.
        filename : str
            File name for saving (without extension).
        """
        if self.fig_title is not None:
            fig.suptitle(self.fig_title)
        if self.save_folder is not None:
            fig.savefig(
                os.path.join(self.save_folder,
                             filename + '.' + self.save_format),
                transparent=True,
                dpi=300
            )
