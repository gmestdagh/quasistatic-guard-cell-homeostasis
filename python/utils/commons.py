"""Utilities for modelling stomata.

Here are gathered classes used in the whole project, including classes
representing reactants and transport processes, and a Newton solver to
compute equilibrium states.
"""
from typing import NamedTuple, Tuple, Dict, Sequence
from dataclasses import dataclass
from enum import IntEnum
from time import time

import jax.numpy as jnp
import jax


class Reactant(NamedTuple):
    """A reactant that is not the solvent.

    A reactant is defined by its name, its charge and its initial
    concentration in each compartment. It does not contribute to volume
    changes.

    Parameters
    ----------
    name : string
        Name of reactant, for display purposes.
    charge : float
        Number of charges of reactant
    concentrations : tuple
        Initial concentrations in each compartment (mmol/L).
        The outermost compartment is at the first position, while
        the most inside compartment is at the end of the sequence.
    external_concentration :  float
        Outer concentration of the reactant.
    """
    name: str
    charge: float
    concentrations: Tuple[float]
    external_concentration: float

    def set_num_membranes(self, n_membranes: int):
        """Adapt the Reactant object to a smaller number of compartments.

        Only the n_membrane first components of the concentrations
        tuple is kept.
        """
        if n_membranes > len(self.concentrations):
            raise ValueError("Only decreasing n_membranes is supported.")

        return Reactant(
            self.name,
            self.charge,
            self.concentrations[:n_membranes],
            self.external_concentration
        )


@dataclass(frozen=True)
class Reaction:
    """A chemical equilibrium inside a single compartment.

    A reaction is defined by its stoichiometry and the compartment
    it happens.
    For instance, in a model with reactants (A, B, C) and three
    compartments (0, 1, 2), a reaction of equation

            A0 + B0 <-> C0

    is declared using

            Reaction([-1., -1., 1.], 0).
    """
    stoichiometry: Dict[str, float]
    compartment_idx: int

    def get_stoichiometric_coefficients(self, reactants: Sequence[Reactant]):
        """Get a vector of stoichiometric coefficients for all reactants.

        From the stoichiometry dictionary and the list of present
        reactants, this function returns an array containing the
        stoichiometric coefficients for each reactant in the same order
        as the reactants provided in the list.

        This procedure does not check that all reactants in the
        stoichiometry dictionary are present in the reactant list.

        Parameters
        ----------
        reactants : Sequence of Reactant
            List of all reactants that are involved in the model.
        """
        return jnp.array([self.stoichiometry.get(r.name, 0.)
                          for r in reactants])


@dataclass(frozen=True)
class Transporter:
    """A passive or active transporter across a membrane.

    A transporter is defined by its stoichiometry and the
    compartments it connects.
    For instance, in a model with reactants (A, B, C) and three
    compartments (1, 2), a channel of equation

            A0 + b B0 <-> A1 + b B1

    is declared using

            Transporter(
                active=False,
                stoichiometry=(1., b, 0.),
                compartment_from=0,
                compartment_to=1
            ).

    The  index for the outside environment is -1.
    A pump that ejects C from 0 to the outside environment is declared
    using

            Transporter(
                active=True,
                stoichiometry=(0., 0., 1.),
                compartment_from=0,
                compartment_to=-1
            ).

    Note that the sign is important in the case of pumps.

    Parameters
    ----------
    active : bool
        True for a pump, false for a channel.
    stoichiometry : array or list
        Stoichiometric coefficients for each reactant.
    compartment_from : int
        Index of the first connected compartment.
    compartment_to : int
        Index of the other connected compartment.
    """
    active: bool
    stoichiometry: Dict[str, float]
    compartment_from: int
    compartment_to: int

    def get_stoichiometric_coefficients(self, reactants: Sequence[Reactant]):
        """Get a vector of stoichiometric coefficients for all reactants.

        From the stoichiometry dictionary and the list of present
        reactants, this function returns an array containing the
        stoichiometric coefficients for each reactant in the same order
        as the reactants provided in the list.

        This procedure does not check that all reactants in the
        stoichiometry dictionary are present in the reactant list.

        Parameters
        ----------
        reactants : Sequence of Reactant
            List of all reactants that are involved in the model.
        """
        return jnp.array(
            [self.stoichiometry.get(r.name, 0.) for r in reactants])


class NewtonSolver:
    """A solver to compute the equilibrium of stomata systems."""
    class Options(NamedTuple):
        abstol: float = 1e-8
        reltol: float = 1e-10
        max_iter: int = 20
        verbose: int = 0

    class Status(IntEnum):
        CONVERGENCE = 0
        MAX_ITER = 1
        NAN_FOUND = -1

    def __init__(self, opt: Options = Options()):
        """Initialize solver and set options.

        Parameters
        ----------
        opt : Options
            Structure containing the options for the solver, namely :
            - abstol : absolute tolerance on the residual
            - reltol : relative tolerance on the residual
            - max_iter : maximum number of iterations.
            - verbose : output level (1: one line per parameter)
                                     (2: one line per iteration).
        """
        self.opt = opt

    def solve(self, fun, x0, p, jac=None):
        """Solve a single nonlinear problem.

        Perform the Newton method to find an x such that f(x, p) = 0.
        The signature of f should be f(x, p), where x is the unknown
        variable and p is a parameter.
        Function fun should be compatible with jax just-in-time
        compilation and automatic differentiation.

        Parameters
        ----------
        fun : function
            Residual function for the problem, with signature f(x, p)
        x0 : ndarray
            Initial guess.
        p : ndarray
            Parameter.
        jac : function [optional]
            Jacobian function. If not provided, the jacobian function is
            computed using automatic differentiation.
        """
        if jac is None:
            jac = jax.jit(jax.jacobian(fun, argnums=0))

        # Initialize variables and tolerances
        x = x0
        residual_vector = fun(x, p)
        err = jnp.linalg.norm(residual_vector)
        iteration = 0

        tolerance = self.opt.abstol + err * self.opt.reltol
        max_iter = self.opt.max_iter
        print_iter = self.opt.verbose >= 2

        if print_iter:
            print('%4s  %13s' % ('iter', '|f(x)|'))

        # Main loop -----------------------------------------------------------
        while err > tolerance:

            # Check for iteration limit
            if iteration >= max_iter:
                return x, self.Status.MAX_ITER

            # Apply Newton update
            jacobian_matrix = jac(x, p)
            x -= jnp.linalg.solve(jacobian_matrix, residual_vector)

            # Check for NaNs
            if jnp.any(jnp.isnan(x)):
                raise Exception('Solution not found with NaN generated')

            # Evaluate new error
            residual_vector = fun(x, p)
            err = jnp.linalg.norm(residual_vector)

            if print_iter:
                print('%4d  %13.6e' % (iteration, err.item()))

            iteration += 1

        # End of main loop ----------------------------------------------------
        return x, self.Status.CONVERGENCE

    def solve_range(self, fun, x_start, p_range, jac=None):
        """Solve a range of nonlinear problems.

        Perform the Newton method to find x such that f(x, p) = 0,
        for p in the range of parameters p_range.
        After each solving, the solution of the problem with the current
        parameters is re-used as the initial guess for the next
        parameter.

        The signature of f should be f(x, p), where x is the unknown
        variable and p is a parameter.
        Function fun should be compatible with jax just-in-time
        compilation and automatic differentiation.

        Parameters
        ----------
        fun : function
            Residual function for the problem, with signature f(x, p)
        x_start : ndarray
            Initial guess.
        p_range : ndarray
            Array containing the sets of parameters to use to solve the
            problem.
        jac : function [optional]
            Jacobian function. If not provided, the jacobian function is
            computed using automatic differentiation.
        """
        if jac is None:
            jac = jax.jit(jax.jacobian(fun, argnums=0))

        # Initialize results
        results = jnp.zeros((p_range.shape[0], x_start.shape[0]))
        x0 = x_start

        # Print output header
        print_result = self.opt.verbose >= 1
        if print_result:
            print('%4s  %20s  %9s' % ('iter', 'status', 'elapsed'))

        # Main loop -----------------------------------------------------------
        for (i, p) in enumerate(p_range):

            # Solve nonlinear problem
            start_time = time()
            sol, status = self.solve(fun, x0, p, jac=jac)
            end_time = time()
            elapsed = end_time - start_time

            # Print solving summary
            if print_result:
                print('%4d  %20s  %9.2e' % (i, status.name, elapsed))

            # Update results
            results = results.at[i].set(sol)
            x0 = sol

        # End of main loop ----------------------------------------------------
        return results
