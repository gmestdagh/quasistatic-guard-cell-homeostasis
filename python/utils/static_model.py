"""Utilities to represent a multi-membrane model and run direct simulations.

This module contains class to build a model from a range of membranes
and transporters.
The MultiMembraneComplex class stores the algebraic structure of the
multi-membrane complex while the MultiMembraneModel class contains
utilities to run a direct simulation.
"""

from functools import partial
from typing import Sequence, Iterable

import jax.numpy as jnp
import jax

from .cell_geometry import AbstractMembrane
from .commons import Reactant, Transporter, Reaction
from .constants import molar_volume_of_water, RT, faraday, membrane_capacitance
from .physics import ScaledEnergies


class MultiMembraneComplex:
    """Reaction structure of a multi-membrane cell complex.

    This class contains topological information about a multi-membrane
    complex, which is a set of several compartments, containing several
    reactants.
    These compartments are connected by transporters and chemical
    reactions that allow the system to move in certain directions of the
    parameter space.
    Some reactions or transporters are active and user-controlled, while
    others are passive and controlled by the laws of physics.

    From a list of membranes and transporters, this class creates bases
    for the space of free directions and the space of fixed directions
    in the nonlinear problem.
    """
    def __init__(self, active_stoichiometry: jax.Array,
            free_directions: jax.Array, fixed_directions: jax.Array):
        """Initialize a multi-membrane complex by providing directions.

        All directions should have the same size.
        Active directions are the directions of active (user-controlled)
        transporters.

        Free and fixed directions constitute a decomposition of the
        variable space into directions where the system can move freely,
        and directions in which the system is fixed.
        The system equilibrium in free directions is determined by a
        physical balance of potentials, while in fixed directions it is
        fixed by user-controlled reactions and transporters.

        The number of free and fixed directions should add up to the
        size of the variable space.
        The user is responsible for free and fixed directions to form a
        basis of the variable space.

        Parameters
        ----------
        active_stoichiometry : ndarray
            2D array of size (n_active_transporters, n_variables)
            containing the directions in which active transporters are
            working.
        free_directions : ndarray
            2D array of size (n_free, n_variables) containing free
            directions.
        fixed_directions : ndarray
            2D array of size (n_fixe n_variables) storing fixed
            directions.
        """
        # Check size of arrays
        n_variables = active_stoichiometry.shape[1]
        assert free_directions.shape[1] == n_variables
        assert fixed_directions.shape[1] == n_variables

        # Check number of free and fixed directions
        n_free = free_directions.shape[0]
        n_fixed = fixed_directions.shape[0]
        assert n_free + n_fixed == n_variables

        # Store input arrays
        self.active_stoichiometry = active_stoichiometry
        self.free_directions = free_directions
        self.fixed_directions = fixed_directions

    @partial(jax.jit, static_argnums=0)
    def residual_from_potentials(self, u: jax.Array, potentials: jax.Array,
            progress: jax.Array):
        """Finish residual assembling from physical potentials.

        From previously computed physical potentials, compute the
        physical residual by evaluating the potentials in the free
        directions while imposing conditions in fixed directions.

        Parameters
        ----------
        u : ndarray
            1D vector containing the scaled quantities of each reactant.
        potentials : ndarray
            2D vector containing the total potential of component j in
            compartment i.
        progress: ndarray
            1D vector containing the scaled progress of active transporters.
        """
        # Control size of input arrays
        assert progress.shape == (self.active_stoichiometry.shape[0],)

        # Include progress of active transporters in fixed directions.
        fixed_rhs = self.active_stoichiometry.T @ progress

        # Assemble residual vector
        return jnp.hstack((
            self.free_directions @ potentials.reshape(-1),
            self.fixed_directions @ (u - fixed_rhs)
        ))

    @classmethod
    def from_transporters(cls,n_membranes: int,
            reactants: Sequence[Reactant],
            transporters: Iterable[Transporter],
            reactions: Iterable[Reaction] = ()):
        """Initialize a multi-membrane complex from transporters and reactions.

        Free and fixed directions are computed from the provided list of
        transporters and reactions.
        As there is one quantity value per compartment and per reactant,
        the number of physical variables is

                N = n_membranes * (n_ionic_reactants + 1).

        Remember that water accounts for one non-ionic reactant.

        Parameters
        ----------
        n_membranes : int
            Number of compartments/membranes in the model
        reactants : Sequence[Reactant]
            List of all reactants including ions and water.
        transporters : Iterable[Transporter]
            Channels and pumps.
        reactions : Iterable[Reaction]
            Reactions that happen in a single compartment.
        """
        # Count water as a reactant
        n_reactants = len(reactants)

        # Check transporters format
        for t in transporters:
            assert -1 <= t.compartment_from < n_membranes
            assert -1 <= t.compartment_to < n_membranes
            assert t.compartment_to != t.compartment_from

        # Distinguish active and passive transporters
        passive_stoichiometry = []
        active_stoichiometry = []

        for t in transporters:
            # Create stoichiometric coefficients (including water)
            s = jnp.array(t.get_stoichiometric_coefficients(reactants))

            # Add contribution of the transporter in each compartment
            direction = jnp.zeros((n_membranes, n_reactants))

            if t.compartment_from >= 0:
                direction = direction.at[t.compartment_from].set(-s)
            if t.compartment_to >= 0:
                direction = direction.at[t.compartment_to].set(s)

            if t.active:
                active_stoichiometry.append(direction.reshape(-1))
            else:
                passive_stoichiometry.append(direction.reshape(-1))

        # Add reactions to passive stoichiometry
        for r in reactions:
            s = jnp.array(r.get_stoichiometric_coefficients(reactants))
            direction = jnp.zeros((n_membranes, n_reactants)) \
                .at[r.compartment_idx].set(s)
            passive_stoichiometry.append(direction.reshape(-1))

        active_stoichiometry = jnp.array(active_stoichiometry)

        # Process passive transporters, using SVD to distinguish free
        # and fixed directions in the nonlinear problem.
        # At equilibrium, the system position in the free directions is
        # defined by chemical equilibrium, while position in fixed
        # directions is imposed by the reaction extent of active
        # transporters.
        _, s, v = jax.scipy.linalg.svd(
            jnp.array(passive_stoichiometry)
        )
        n_free_directions = s.shape[0]
        free_directions = v[:n_free_directions]
        fixed_directions = v[n_free_directions:]

        return cls(active_stoichiometry, free_directions, fixed_directions)


class DirectStaticModel:
    """A model with n membranes, with volume changes.

    This model manages a set of n_membranes nested compartments, indexed
    with the index i, compartment 0 being the outer compartment (e.g.
    the cytoplasm of the cell), while compartment (n_membranes - 1) is
    the most inner compartment (e.g. the vacuole).

    The reactants are indexed by j and are specified by the user, except
    for water (the solvent), which is added automatically when creating
    the model.
    """

    def __init__(self,
                 membranes: Sequence[AbstractMembrane],
                 ionic_reactants: Iterable[Reactant],
                 transporters: Iterable[Transporter],
                 reactions: Iterable[Reaction] = ()):
        """Initialize the model, prepare scaled coefficients, etc.

        Parameters
        ----------
        membranes : sequence of AbstractMembrane
            Sequence of AbstractMembrane objects that define the
            geometrical and mechanical properties of each compartment.
        ionic_reactants : iterable of Reactant
            Non-solvent reactants.
            Only the (n_membrane + 1) first concentrations values will
            be used.
        transporters : iterable of Transporter
            Channels and pumps.
        initial_membrane_potentials : sequence of float
            Difference of electric potential across each membrane in the
            initial conditions. Namely, initial_membrane_potential[i] is
            the difference between the absolute potential in
            compartments i and i-1 (-1 is outside the cell).
        reactions : iterable of Reaction
            Reactions that happen in a single compartment.
        """

        # Process membranes and reactants -------------------------------------
        # Check that membrane volumes are in decreasing order
        membrane_volumes = jnp.array([m.initial_volume for m in membranes])
        assert jnp.all(jnp.diff(membrane_volumes, append=0.) < 0.)

        self.n_membranes = len(membranes)
        self.membranes = membranes

        # Check that reactant concentrations are provided in every compartment
        for r in ionic_reactants:
            assert len(r.concentrations) == self.n_membranes
            assert r.name != 'water'

        # Add water to reactants
        water_concentration = 1. / molar_volume_of_water
        water = Reactant(
            name='water',
            charge=0.,
            external_concentration=water_concentration,
            concentrations=(water_concentration,) * self.n_membranes
        )
        self.reactants = tuple(ionic_reactants) + (water,)
        self.n_reactants = len(self.reactants)

        # Compute chemical and electrostatic parameters, used to
        # evaluate scaled residuals and energies.
        self.phy_funcs = ScaledEnergies(self.membranes)

        initial_membrane_potentials = jnp.array(
            [m.initial_electric_potential for m in membranes]
        )
        self.chem, self.elec = self.phy_funcs.get_parameters(
            membranes=self.membranes,
            membrane_potentials=initial_membrane_potentials,
            reactants=self.reactants  # Including water
        )

        # Prepare fixed and free directions -----------------------------------
        # In the structure attribute, the space of variables is
        # decomposed between free and fixed directions.
        water_transport = (
            Transporter(
                active=False,
                stoichiometry={'water': 1.},
                compartment_from=i,
                compartment_to=i-1
            ) for i in range(self.n_membranes)
        )
        all_transporters = tuple(transporters) + tuple(water_transport)

        self.multi_membrane_complex = MultiMembraneComplex.from_transporters(
            n_membranes=self.n_membranes,
            reactants=self.reactants,
            transporters=all_transporters,
            reactions=reactions
        )

        # Mathematical functions ----------------------------------------------
        # Compute initial derivatives of the energy to enforce
        # equilibrium at initial condition.
        u0 = jnp.zeros((self.n_membranes, self.n_reactants))
        self.initial_potentials = self.phy_funcs.total_potentials(
            u0, self.chem, self.elec)

    # Physical residual and jacobian ==========================================
    @partial(jax.jit, static_argnums=0)
    def physical_residual(self, u: jax.Array, progress: jax.Array):
        """Residual function of the stoma nonlinear problem.

        The lines of the residual are organized as follows :
            - chemical equilibrium for reactants in each free direction
            - water equilibrium
            - imposed component in each fixed direction.

        Parameters
        ----------
        u : ndarray
            1D vector containing the scaled quantities of each reactant.
        progress: ndarray
            1D vector containing the scaled progress of active
            transporters.
        """
        assert u.shape == (self.n_membranes * self.n_reactants,)

        # Reactant potentials are the sum of all potentials
        u_reshaped = u.reshape(self.n_membranes, self.n_reactants)
        potentials = self.phy_funcs.total_potentials(
            u_reshaped, self.chem, self.elec)
        potentials -= self.initial_potentials

        # Assemble residual vector
        residual = self.multi_membrane_complex.residual_from_potentials(
            u, potentials, progress)
        assert residual.shape == u.shape

        return residual

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=1)
    def physical_jacobian(self, u: jax.Array, progress: jax.Array):
        """Jacobian of the physical residual function.

        This is the derivative of the physical_residual method with
        respect to u.

        Parameters
        ----------
        u : ndarray
            1D vector containing the scaled quantities of each reactant.
        progress: ndarray
            1D vector containing the scaled progress of active
            transporters.
        """
        return self.physical_residual(u, progress)

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=2)
    def physical_jac_wr_progress(self, u: jax.Array, progress: jax.Array):
        """Derivative of physical_residual w/r to progress.

        Parameters
        ----------
        u : ndarray
            1D vector containing the scaled quantities of each reactant.
        progress: ndarray
            1D vector containing the scaled progress of active
            transporters.
        """
        return self.physical_residual(u, progress)

    # Robust residual and jacobian ============================================
    @partial(jax.jit, static_argnums=0)
    def robust_residual(self, u_mu: jax.Array, progress: jax.Array):
        """Residual function with relaxation for robustness.

        This residual functions involves a change of variable to
        eliminate the difficulty associated to the log in the chemical
        potential.
        The input vector is a stack [u, mu], where mu is supposed to
        satisfy mu_i = log(x_i) at the solution.

        Parameters
        ----------
        u_mu : ndarray
            1D array containing the unknown u and the relaxation
            variable mu.
        progress : ndarray
            1D array containing scaled for each active transporter.
        """
        u_size = self.n_membranes * self.n_reactants
        assert u_mu.shape == (2 * u_size,)

        u, mu = u_mu[:u_size], u_mu[u_size:]
        u_reshaped = u.reshape(self.n_membranes, self.n_reactants)
        mu_reshaped = mu.reshape(self.n_membranes, self.n_reactants)

        # First part of the residual ------------------------------------------
        # Reactant potentials are the sum of all potentials
        chemical_potentials = self.chem.mu0 + mu_reshaped
        electro_potentials = self.phy_funcs.electrostatic_potentials(
            u_reshaped, *self.elec)
        elastic_potentials = self.phy_funcs.elastic_potentials(u_reshaped)

        potentials = chemical_potentials \
            + electro_potentials \
            + elastic_potentials \
            - self.initial_potentials

        residual_top = self.multi_membrane_complex.residual_from_potentials(
            u, potentials, progress)

        # Second part of the residual -----------------------------------------

        # This part of the residual compares f(x) and g(mu), where x is
        # the vector of molar fractions and
        #
        #     f(x) = log(x) if x > 1, x - 1 otherwise
        #
        #     g(mu) = mu if mu > 0, exp(mu) - 1 otherwise.
        #

        # Compute molar fractions
        molar_fractions = self.phy_funcs.molar_fractions(
            u_reshaped, *self.chem) \
            .reshape(-1)

        # Evaluate change of variable
        fx = jnp.where(molar_fractions > 1., jnp.log(molar_fractions),
                    molar_fractions - 1.)
        gmu = jnp.where(mu > 0., mu, jnp.exp(mu) - 1.)
        residual_bottom = fx - gmu
        return jnp.hstack((residual_top, residual_bottom))

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=1)
    def robust_jacobian(self, u: jax.Array, progress: jax.Array):
        """Jacobian of the robust residual function.

        This is the derivative of the robust_residual method with
        respect to u.

        Parameters
        ----------
        u : ndarray
            1D vector containing the scaled quantities of each reactant.
        progress: ndarray
            1D vector containing the scaled progress of active
            transporters.
        """
        return self.robust_residual(u, progress)

    @partial(jax.jit, static_argnums=0)
    @partial(jax.jacfwd, argnums=2)
    def robust_jac_wr_progress(self, u: jax.Array, progress: jax.Array):
        """Derivative of robust_residual w/r to progress.

        Parameters
        ----------
        u : ndarray
            1D vector containing the scaled quantities of each reactant.
        progress: ndarray
            1D vector containing the scaled progress of active
            transporters.
        """
        return self.robust_residual(u, progress)

    @partial(jax.jit, static_argnums=0)
    def get_robust_initial_guess(self, u0=None):
        """Convenience function to initialize robust variable.

        Starting from a physical first guess, this function creates the
        first initial guess to feed to the robust problem.

        Parameters
        ----------
        u0 : ndarray
            Physical first guess.
        """
        if u0 is None:
            u0 = jnp.zeros((self.n_membranes, self.n_reactants))

        mu0 = jnp.log(self.phy_funcs.molar_fractions(u0, *self.chem))
        return jnp.stack((u0, mu0)).reshape(-1)

    # Produce dimensioned results from scaled results =========================
    def get_progress_range(self, max_progress: float,
                           range_size: float = 501,
                           ratios: Sequence[float] = (1.,)):
        """Create a scaled progress range where to solve equilibrium.

        This function returns the range of progress values for each
        active transporter along with the actual progress that serves as
        a reference for plots.
        All returned progress is in the domain of scaled variables.

        Parameters
        ----------
        max_progress : float
            Maximum progress of the active transporters (e.g. quantity
            of protons ejected by a proton pump).
        range_size : float
            Number of points where the equilibrium will be solved.
        ratios : sequence of float
            Ratios of progress for each transporter.
        """
        progress_range = jnp.linspace(0.0, max_progress, num=range_size)
        scaled_progress = self.phy_funcs.to_scaled_progress(progress_range)
        ratios = jnp.array(ratios).reshape(1, -1)
        active_progress_ranges = scaled_progress.reshape(-1, 1) @ ratios
        return active_progress_ranges, scaled_progress

    @partial(jax.jit, static_argnums=0)
    def dimensioned_values(self, scaled_progress, u):
        """Retrieve dimensioned values from scaled solutions of NL problem.

        From the scaled solution of the nonlinear problem, returns a
        dataframe containing all physical quantities of interest,
        including concentrations of reactants, pressure, volume,
        electric potential, etc.

        All quantities are normalized by the cell initial volume. The
        stored value is n / V0 (in mmol / L).

        Parameters
        ----------
        scaled_progress: ndarray
            Range of parameters for which the problem has been solved.
        u: ndarray
            Solution of nonlinear problem for each progress.
        """
        res = dict()

        # Save dimensioned progress (normalized by initial cell volume)
        res['progress'] = self.phy_funcs \
            .to_dimensioned_progress(scaled_progress)

        # Compute dimensioned quantities and concentrations
        # Quantities are normalized by the initial cell volume
        u_reshaped = u.reshape(-1, self.n_membranes, self.n_reactants)
        dimensioned_quantities, dimensioned_concentrations \
            = self.phy_funcs.to_dimensioned_quantities(u_reshaped, *self.chem)

        for (ir, r) in enumerate(self.reactants):
            quantity_key = 'n_' + r.name
            res[quantity_key] = dimensioned_quantities[:,:,ir]
            concentration_key = 'c_' + r.name
            res[concentration_key] = dimensioned_concentrations[:,:,ir]

        # Volume of each compartment
        water_quantities = dimensioned_quantities[:,:,-1]
        res['volume'] = water_quantities * molar_volume_of_water

        # Membrane potentials and charge imbalance
        @jax.jit
        @jax.vmap
        def compute_membrane_potentials(_u: jax.Array):
            return self.phy_funcs.membrane_potentials(_u, *self.elec)

        elec_potentials, membrane_areas \
            = compute_membrane_potentials(u_reshaped)
        elec_scaling = RT / faraday
        res['electric_potential'] = elec_potentials * elec_scaling

        for (j, m) in enumerate(self.membranes):
            membrane_areas = membrane_areas.at[:,j].mul(m.initial_area)
        res['charge_imbalance'] =  (membrane_capacitance / faraday) \
            * res['electric_potential'] * membrane_areas

        # Pressure
        @jax.jit
        @jax.vmap
        def compute_elastic_potential(_u: jax.Array):
            return self.phy_funcs.elastic_potentials(_u)[:,-1]

        scaled_pressure = compute_elastic_potential(u_reshaped) # \
            # - self.initial_potentials[:,-1]
        res["pressure"] = scaled_pressure * (RT / molar_volume_of_water)

        return res
