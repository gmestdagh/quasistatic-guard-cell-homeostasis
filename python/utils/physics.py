"""Physical energies used for simulations.

The following classes are responsible for evaluating physical functions
such as energies and potentials, and also manage the problem scaling.
There are several versions, depending on whether we want to solve a
scaled problem or use physical values directly.
"""

from functools import partial
from collections import namedtuple
from typing import Sequence

import jax
import jax.numpy as jnp

from .commons import Reactant
from .cell_geometry import AbstractMembrane
from .constants import molar_volume_of_water, RT, faraday, membrane_capacitance


###############################################################################
# Physical functions
###############################################################################
class DimensionedEnergies:
    """Evaluation of physical energies directly from molar quantities."""
    def __init__(self,
                 membranes: Sequence[AbstractMembrane],
                 reactants: Sequence[Reactant]):
        """Initialize an instance.

        Parameters
        ----------
        membranes : sequence of AbstractMembrane
            Sequence of AbstractMembrane objects that define the
            geometrical and mechanical properties of each compartment.
        reactants : sequence of Reactant
            Sequence containing the reactants present in the system
            (including water).
        """
        self.membranes = membranes

        # Compute chemical parameters
        external_concentrations \
            = jnp.array([r.external_concentration for r in reactants])
        self.external_molar_fractions = external_concentrations \
            / external_concentrations.sum()

        # Compute electrostatic parameters
        # Charge of each reactant
        self.charges = jnp.array([r.charge for r in reactants])

        # Charge initially stored in each membrane
        initial_areas = jnp.array([m.initial_area for m in membranes])
        initial_capacitances = initial_areas * membrane_capacitance
        initial_potentials = jnp.array(
            [m.initial_electric_potential for m in membranes])

        self.Q0 = initial_capacitances * initial_potentials
        self.initial_capacitances = initial_capacitances

        # Initial quantity of each reactant in each compartment
        membrane_volumes = jnp.array([m.initial_volume for m in membranes])
        compartment_volumes = -jnp.diff(membrane_volumes, append=0.)
        reactant_concentrations \
            = jnp.array([r.concentrations for r in reactants]) \
            .transpose()

        self.initial_volumes = membrane_volumes
        self.initial_quantities \
            = reactant_concentrations * compartment_volumes.reshape(-1, 1)

    # Common functions ========================================================
    @staticmethod
    @jax.jit
    def get_compartment_volumes(quantities: jax.Array):
        """Compute the volume of each compartment.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        water_quantities = quantities[:, -1].reshape(-1)
        return water_quantities * molar_volume_of_water

    @partial(jax.jit, static_argnums=0)
    def get_volumes_increase(self, quantities: jax.Array):
        """Compute relative volume increase compared to initial conditions.

        The volume increases concerns the volume enclosed in each
        membrane and not the volume of each compartment.
        The volume enclosed in a membrane is the sum of volumes of
        compartments enclosed in the membrane.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        water_quantities = quantities[:, -1].reshape(-1)
        volumes = jax.lax.cumsum(water_quantities, reverse=True) \
            * molar_volume_of_water
        return (volumes - self.initial_volumes) / self.initial_volumes

    # Chemical energy and potentials ==========================================
    @partial(jax.jit, static_argnums=0)
    def chemical_energy(self, quantities: jax.Array):
        """Evaluate chemical Gibbs free energy term.

        The chemical energy penalises differences in reactant
        concentration between compartments.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        molar_fractions = quantities / quantities.sum(axis=1).reshape(-1, 1)
        energy_terms = RT * quantities \
            * jnp.log(molar_fractions / self.external_molar_fractions)
        return energy_terms.sum()

    @partial(jax.jit, static_argnums=0)
    def chemical_potentials(self, quantities: jax.Array):
        """Evaluate chemical potentials.

        Chemical potentials are the derivatives of the chemical energy
        w/r to quantities.
        They are implemented manually because their formula is very
        simple.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        molar_fractions = quantities / quantities.sum(axis=1).reshape(-1, 1)
        return RT * jnp.log(molar_fractions / self.external_molar_fractions)

    # Electrostatic energy and potentials =====================================
    @partial(jax.jit, static_argnums=0)
    def get_charges(self, quantities: jax.Array):
        """Evaluate the electric charge stored in each membrane.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        diff_quantities = quantities - self.initial_quantities
        diff_charges = faraday * (diff_quantities @ self.charges)
        return self.Q0 + jax.lax.cumsum(diff_charges, reverse=True)

    @partial(jax.jit, static_argnums=0)
    def get_capacitances(self, quantities: jax.Array):
        """Evaluate the electric capacitance of each membrane.

        Current capacitance is computed from initial capacitance and
        area increase of each membrane.
        The area increase is computed from the volume increase.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        volumes_increase = self.get_volumes_increase(quantities)
        areas_increase = jnp.array([
            m.area_increase(v)
            for (m, v) in zip(self.membranes, volumes_increase)
        ])
        return (1. + areas_increase) * self.initial_capacitances

    @partial(jax.jit, static_argnums=0)
    def electrostatic_energy(self, quantities: jax.Array):
        """Evaluate electrostatic energy stored in membranes.

        To compute this energy, membrane are considered as capacitors.
        The energy has the form E = (1/2) Q^2 / C.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        charges = self.get_charges(quantities)
        capacitances = self.get_capacitances(quantities)
        return 0.5 * jnp.sum(charges**2 / capacitances)

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=1)
    def diff_electrostatic_energy(self, quantities: jax.Array):
        """Derivative of the electrostatic energy function.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        return self.electrostatic_energy(quantities)

    @partial(jax.jit, static_argnums=0)
    def membrane_potentials(self, quantities: jax.Array):
        """Evaluate electric potential across each membrane.

        The membrane is considered as a capacitor, and the electric
        potential reads phi = Q / C.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        charges = self.get_charges(quantities)
        capacitances = self.get_capacitances(quantities)
        return charges / capacitances

    # Elastic energy and potentials ===========================================
    @partial(jax.jit, static_argnums=0)
    def elastic_energy(self, quantities: jax.Array):
        """Elastic energy due to deformation of membranes.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        volumes_increase = self.get_volumes_increase(quantities)
        elastic_energies = jnp.array([
            m.elastic_energy(w)
            for (m, w) in zip(self.membranes, volumes_increase)
        ])
        return elastic_energies.sum()

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=1)
    def diff_elastic_energy(self, quantities: jax.Array):
        """Derivative of the elastic energy.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        return self.elastic_energy(quantities)

    @partial(jax.jit, static_argnums=0)
    def pressure(self, quantities: jax.Array):
        """Evaluate pressure in each compartment.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        return self.diff_elastic_energy(quantities)[:,-1] \
            / molar_volume_of_water

    # Total potentials ========================================================
    @partial(jax.jit, static_argnums=0)
    def total_potentials(self, quantities: jax.Array):
        """Compute the sum of all potentials.

        Compute the sum of chemical potentials, electrostatic potentials
        and elastic potentials.

        Parameters
        ----------
        quantities : ndarray
            2D array whose element at (i,j) is the quantity of component
            j in compartment i.
        """
        chemical_potentials = self.chemical_potentials(quantities)
        elec_potentials = self.diff_electrostatic_energy(quantities)
        elastic_potentials = self.diff_elastic_energy(quantities)
        return chemical_potentials + elec_potentials + elastic_potentials


class ScaledEnergies:
    """Evaluation of physical energies and potentials using scaling.

    This class is responsible for evaluating physical functions
    (energies, potentials), but also for computing scaled parameters and
    transforming between scaled and physical values.
    It does not store numerical parameters, as we do not know which
    parameters should be estimated, except for the quantity and energy
    scaling parameters, that are constants.

    In this class, molar quantities are normalized by the initial
    quantity of water in the whole cell.
    """
    ChemicalParameters = namedtuple("ChemicalParameters", "mu0 alpha alpha_s")
    ElectrostaticParameters = namedtuple("ElectrostaticParameters",
        "delta_phi0 c_phi z")

    def __init__(self, membranes: Sequence[AbstractMembrane]):
        """Initialize an instance and precompute scaling parameters.

        Parameters
        ----------
        membranes : sequence of AbstractMembrane
            Sequence of AbstractMembrane objects that define the
            geometrical and mechanical properties of each compartment.
        """
        # TODO: This class should not take membranes as an attribute
        self.membranes = membranes
        initial_cell_volume = membranes[0].initial_volume
        self.quantity_scaling = initial_cell_volume / molar_volume_of_water
        self.energy_scaling = self.quantity_scaling * RT

        # Volume correction for cumulated quantities
        membrane_volumes = jnp.array([m.initial_volume for m in membranes])
        self.cumulated_scaling = membrane_volumes[0] / membrane_volumes

    # Precomputed adimensioned parameters =====================================
    def compute_chemical_parameters(self,
            membrane_volumes: jax.Array,
            reactant_concentrations: jax.Array,
            reactant_external_concentrations: jax.Array):
        """Precompute chemical parameters for scaled evaluation.

        Parameters
        ----------
        membrane_volumes : ndarray
            1D array containing the initial volume of each membrane.
        reactant_concentrations : ndarray
            2D array containing the initial reactant concentrations, for
            reactant j in compartment i.
        reactant_external_concentrations : ndarray
            1D array containing the external concentration of each
            reactant.
        """
        # Initial molar fractions and chemical potentials
        outer_molar_fractions = reactant_external_concentrations \
            / reactant_external_concentrations.sum()
        inner_molar_fractions = reactant_concentrations \
            / reactant_concentrations.sum(axis=1).reshape(-1, 1)
        mu0 = jnp.log(inner_molar_fractions / outer_molar_fractions)

        # Potential sensitivity terms
        compartment_volumes = -jnp.diff(membrane_volumes, append=0.)
        reactant_quantities \
            = reactant_concentrations * compartment_volumes.reshape(-1, 1)

        # Scaled sensitivity parameters
        alpha = self.quantity_scaling / reactant_quantities
        alpha_s = self.quantity_scaling / reactant_quantities.sum(axis=1)

        return self.ChemicalParameters(mu0, alpha, alpha_s)

    def compute_electrostatic_parameters(self,
            membrane_capacitances: jax.Array,
            membrane_potentials: jax.Array,
            reactant_charges: jax.Array):
        """Compute scaled electrostatic parameters.

        The initial electric potentials are the membrane potentials,
        that is the difference of potential between a compartment and
        the next one.

        Parameters
        ----------
        membrane_capacitances : ndarray
            1D array containing each initial membrane capacitance.
        membrane_potentials : ndarray
            1D array containing each initial membrane potential.
        reactant_charges : ndarray
            1D array containing the charge of each reactant.
        """
        delta_phi0 = (faraday / RT) * membrane_potentials
        c_phi = \
            (faraday**2 * self.quantity_scaling) / (RT * membrane_capacitances)

        return self.ElectrostaticParameters(
            delta_phi0, c_phi, reactant_charges)

    def get_parameters(self, membranes: Sequence[AbstractMembrane],
            membrane_potentials: Sequence[float],
            reactants: Sequence[Reactant]):
        """Compute parameters from a list of membranes and reactants.

        Convenience function to get all precomputed parameters from a
        list of membranes and a list of reactants.

        Parameters
        ----------
        membranes : Sequence[AbstractMembrane]
            Sequence of AbstractMembrane objects that define the
            geometrical and mechanical properties of each compartment.
        membrane_potentials : Sequence[float]
            1D array containing each initial membrane potential.
        reactants : Sequence[Reactant]
            Sequence containing all the reactant of the system,
            including water.
        """
        assert len(membrane_potentials) == len(membranes)

        # Prepare relevant arrays
        membrane_volumes = jnp.array([m.initial_volume for m in membranes])
        assert jnp.all(jnp.diff(membrane_volumes, append=0.) < 0.)
        membrane_capacitances = membrane_capacitance \
            * jnp.array([m.initial_area for m in membranes])
        membrane_potentials = jnp.array(membrane_potentials)

        reactant_charges = jnp.array([r.charge for r in reactants[:-1]])
        reactant_concentrations \
            = jnp.array([r.concentrations for r in reactants]) \
            .transpose()
        reactant_external_concentrations \
            = jnp.array([r.external_concentration for r in reactants])

        # Compute parameters
        chem_parameters = self.compute_chemical_parameters(membrane_volumes,
            reactant_concentrations, reactant_external_concentrations)
        elec_parameters = self.compute_electrostatic_parameters(
            membrane_capacitances, membrane_potentials, reactant_charges)

        return chem_parameters, elec_parameters

    # Chemical energy and potentials ==========================================
    @staticmethod
    @jax.jit
    def molar_fractions(u: jax.Array, *chemical_parameters):
        """Evaluate scaled molar fractions for an input vector u.

        The scaled molar_fraction of component j in compartment i is
        defined by

                            1 + alpha_ij * u_ij
            x_ij = ----------------------------------- .
                   1 + alpha_s_i * (u_i1 + ... + u_in)

        Parameters
        ----------
        u : ndarray
            2D vector so that u_ij is the scaled quantity of component j
            in compartment i.
        chemical_parameters: self.ChemicalParameters
            Chemical adimensioned coefficients, as provided by the
            compute_chemical_parameters method.

        Returns
        -------
        x : ndarray
            2D vector containing the x_ij as defined above.
        """
        _, alpha, alpha_s = chemical_parameters
        num = 1. + alpha * u
        den = 1. + alpha_s * u.sum(axis=1)
        return num / den.reshape(-1, 1)

    @staticmethod
    @jax.jit
    def chemical_energy(u: jax.Array, *chemical_parameters):
        """Evaluate the scaled chemical free enthalpy.

        Parameters
        ----------
        u : ndarray
            2D vector so that u_ij is the scaled quantity of component j
            in compartment i.
        chemical_parameters: self.ChemicalParameters
            Chemical adimensioned coefficients, as provided by the
            compute_chemical_parameters method.
        """
        mu0, alpha, alpha_s = chemical_parameters

        # Evaluate molar fractions
        num = 1. + alpha * u
        den = 1. + alpha_s * u.sum(axis=1)
        molar_fractions = num / den.reshape(-1, 1)
        mu = mu0 + jnp.log(molar_fractions)

        # Return energy as a sum
        terms = num * mu / alpha
        return terms.sum()

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def chemical_potentials(cls, u: jax.Array, *chemical_parameters):
        """Evaluate scaled chemical potentials for an input vector u.

        The scaled chemical potential of component j in compartment j is
        evaluated using the formula

            mu_ij = mu_{0,ij} + log(x_ij)

        where x_i is the scaled molar fraction.

        Parameters
        ----------
        u : ndarray
            2D vector so that u_ij is the scaled quantity of component j
            in compartment i.
        chemical_parameters: self.ChemicalParameters
            Chemical adimensioned coefficients, as provided by the
            compute_chemical_parameters method.

        Returns
        -------
        mu : ndarray
            2D vector containing the mu_ij as defined above.
        """
        mu0, _, _ = chemical_parameters
        molar_fractions = cls.molar_fractions(u, *chemical_parameters)
        return mu0 + jnp.log(molar_fractions)

    # Electrostatic energy and potentials =====================================
    @partial(jax.jit, static_argnums=0)
    def membrane_potentials(self, u: jnp.array, *electrostatic_parameters):
        """Evaluate membrane potentials and area increase.

        Parameters
        ----------
        u : ndarray
            2D vector so that u_ij is the scaled quantity of component j
            in compartment i.
        electrostatic_parameters: self.ElectrostaticParameters
            Electrostatic adimensioned coefficients, as provided by the
            compute_electrostatic_parameters method.

        Returns
        -------
        membrane_potentials : ndarray
            Vector of size n_membranes containing the difference of
            potential across each membrane.
        membrane_area : ndarray
            Vector of size n_membranes containing the relative area
            increase of each membrane.
        """
        delta_phi0, c_phi, z = electrostatic_parameters

        # Compute volume and area increase
        u_cumulated = jax.lax.cumsum(u, axis=0, reverse=True)
        volume_increases = self.cumulated_scaling * u_cumulated[:,-1]
        membrane_area = jnp.array([
            m.area_increase(w) + 1.
            for (m, w) in zip(self.membranes, volume_increases)
        ])

        membrane_potentials = (delta_phi0
            + c_phi * jnp.dot(u_cumulated[:,:-1], z)) / membrane_area

        return membrane_potentials, membrane_area

    @partial(jax.jit, static_argnums=0)
    def electrostatic_energy(self, u: jax.Array, *electrostatic_parameters):
        """Evaluate electrostatic energy due to membrane polarization.

        The variation of electrostatic energy is different for reactants
        and for the water.
        For reactants, it is based on charge variations, while for water
        it is based on membrane electric capacitance variations.

        Parameters
        ----------
        u : ndarray
            2D vector so that u_ij is the scaled quantity of component j
            in compartment i.
        electrostatic_parameters: self.ElectrostaticParameters
            Electrostatic adimensioned coefficients, as provided by the
            compute_electrostatic_parameters method.
        """
        _, c_phi, _ = electrostatic_parameters

        membrane_potentials, membrane_area = \
            self.membrane_potentials(u, *electrostatic_parameters)
        energies = 0.5 * membrane_potentials**2 * membrane_area / c_phi
        return energies.sum()

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=1)
    def electrostatic_potentials(self, u: jax.Array,
            *electrostatic_parameters):
        """Derivative of the electrostatic_energy function.

        This functions computes the derivatives of the electrostatic
        energy with respect to the scaled quantities in u.
        The result is in the same unit as a (scaled) chemical potential.

        Parameters
        ----------
        u : ndarray
            2D vector so that u_ij is the scaled quantity of component j
            in compartment i.
        electrostatic_parameters: self.ElectrostaticParameters
            Electrostatic adimensioned coefficients, as provided by the
            compute_chemical_parameters method.
        """
        return self.electrostatic_energy(u, *electrostatic_parameters)

    # Elastic energy and potentials ===========================================
    @partial(jax.jit, static_argnums=0)
    def elastic_energy(self, u: jax.Array):
        """Elastic energy due to membrane deformations.

        Parameters
        ----------
        u : ndarray
            2D vector so that u_ij is the scaled quantity of component j
            in compartment i.
        """
        u_water = u[:,-1]
        volume_increases = jax.lax.cumsum(u_water, reverse=True) \
            * self.cumulated_scaling

        # Elastic energy for each membrane
        elastic_energies = jnp.array([
            m.elastic_energy(w)
            for (m, w) in zip(self.membranes, volume_increases)
        ])
        return elastic_energies.sum() / self.energy_scaling

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=1)
    def elastic_potentials(self, u: jax.Array):
        """Derivative of the elastic_energy function.

        This functions computes the derivatives of the electrostatic
        energy with respect to the scaled quantities in u.
        The result is in the same unit as a (scaled) chemical potential.

        Parameters
        ----------
        u : ndarray
            2D vector so that u_ij is the scaled quantity of component j
            in compartment i.
        """
        return self.elastic_energy(u)

    # Total potentials and energy =============================================
    @partial(jax.jit, static_argnums=0)
    def total_potentials(self, u: jax.Array,
            chemical_parameters: ChemicalParameters,
            electrostatic_parameters: ElectrostaticParameters):
        """Sum of chemical, electrostatic and elastic potentials.

        Parameters
        ----------
        u : ndarray
            2D vector so that u_ij is the scaled quantity of component j
            in compartment i.
        chemical_parameters: self.ChemicalParameters
            Chemical adimensioned coefficients, as provided by the
            compute_chemical_parameters method.
        electrostatic_parameters: self.ElectrostaticParameters
            Electrostatic adimensioned coefficients, as provided by the
            compute_electrostatic_parameters method.

        Returns
        -------
        res : ndarray
            2D vector with same shape as u containing th derivative of
            the total energy with respect to each component of u.
        """
        chemical_potentials = self.chemical_potentials(
            u, *chemical_parameters)
        electrostatic_potentials = self.electrostatic_potentials(
            u, *electrostatic_parameters)
        elastic_potentials = self.elastic_potentials(u)
        return chemical_potentials + electrostatic_potentials \
            + elastic_potentials

    # Conversion between scaled and dimensioned values ========================
    @partial(jax.jit, static_argnums=0)
    def to_scaled_progress(self, dimensioned_progress: jax.Array):
        """Transform a dimensioned progress into scaled progress.

        The input should be the dimensioned progress (in mol).
        """
        return dimensioned_progress / self.quantity_scaling

    @partial(jax.jit, static_argnums=0)
    def to_dimensioned_progress(self, scaled_progress: jax.Array):
        """Transform a scaled progress range into dimensioned progress.

        The result is the dimensioned progress (in mol).
        """
        return scaled_progress * self.quantity_scaling

    @partial(jax.jit, static_argnums=0)
    def to_dimensioned_quantities(self, u: jax.Array, *chemical_parameters):
        """Transform scaled quantities into dimensioned quantities.

        The result vectors contains quantities (in mol) and
        concentrations (in mmol/L), respectively.
        The dimensioned quantity is computed using the formula

            n = n_0 + self.quantity_scaling * u

        Parameters
        ----------
        u : ndarray
            Array of shape (*, n_membranes, n_reactants) containing a
            range of scaled quantities.
        chemical_parameters: self.ChemicalParameters
            Chemical adimensioned coefficients, as provided by the
            compute_chemical_parameters method.

        Returns
        -------
        quantities : ndarray
            Array of shape (*, n_membranes, n_reactants) containing
            dimensioned quantities.
        concentrations : ndarray
            Array of shape (*, n_membranes, n_reactants) containing
            dimensioned concentrations.
        """
        # Compute initial quantities from chemical parameters
        _, alpha, _ = chemical_parameters
        initial_quantities = self.quantity_scaling / alpha

        # Compute dimensioned quantities
        # All '_reshaped' variables have shape (*, n_membranes, n_reactants)
        n_membranes, n_reactants = alpha.shape
        u_reshaped = u.reshape(-1, n_membranes, n_reactants)
        quantities_reshaped = \
            initial_quantities + self.quantity_scaling * u_reshaped
        quantities = quantities_reshaped.reshape(*u.shape)

        # Compute concentrations
        water_quantities = quantities_reshaped[:,:,-1] \
            .reshape(-1, n_membranes, 1)
        compartment_volumes = water_quantities * molar_volume_of_water
        concentrations_reshaped = quantities_reshaped / compartment_volumes
        concentrations = concentrations_reshaped.reshape(*u.shape)

        return quantities, concentrations
