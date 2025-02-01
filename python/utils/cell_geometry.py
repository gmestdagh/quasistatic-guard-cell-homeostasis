"""Various models for the cell geometry.

The membrane geometry defines the relationship between the compartment
volume and surface area, as well as the expression for deformation
energy.
"""

from jax.numpy import pi, cbrt
from abc import ABC, abstractmethod


class AbstractMembrane(ABC):
    """All geometric and mechanical parameters necessary for a membrane.

    Classes that inherit from this class describe various geometries for
    the membranes of the complex.
    The AbstractMembrane class stores the initial state of the membrane,
    including its initial volume, area and electrostatic potential.
    """

    def __init__(self, initial_volume: float, initial_area: float,
                 initial_electric_potential: float,
                 initial_pressure: float = 0.):
        """Create abstract membrane.

        Parameters
        ----------
        initial_volume : float
            Initial membrane volume (m^2).
        initial_area : float
            Initial membrane surface area (m^2).
        initial_electric_potential : float
            Initial electric potential (V) across the membrane, defined
            as the potential inside the membrane minus the potential
            outside the membrane.
        """
        self.initial_volume = initial_volume
        self.initial_area = initial_area
        self.initial_electric_potential = initial_electric_potential
        self.initial_pressure = initial_pressure

        # Compute initial slope of the energy function from pressure
        self.initial_slope = initial_volume * initial_pressure

    @abstractmethod
    def area_increase(self, volume_increase: float):
        """Evaluate ratio (current_area - initial_area) / initial area."""
        pass

    @abstractmethod
    def diff_area_increase(self, volume_increase: float):
        """Derivative of area_increase."""
        pass

    @abstractmethod
    def elastic_energy(self, volume_increase: float):
        """Evaluate scaled elastic energy from volume increase."""
        pass

    @abstractmethod
    def diff_elastic_energy(self, volume_increase: float):
        """Derivative of elastic_energy."""
        pass


class CylinderMembrane(AbstractMembrane):
    """A cylinder membrane that expands in only one direction.

    In this geometry, the cell only expands in a longitudinal direction,
    so that the volume is always proportional to the length.
    A linear elastic deformation occurs on the lateral surface while the
    cylinder extremities remain undeformed.
    The deformation energy of the lateral surface is defined by

        W = lateral_surface_area * wall_modulus
            * (current_volume / initial_volume - 1)**2,

    where wall_modulus (in N/m) represents the cell wall Young modulus
    multiplied by the wall thickness.
    The Poisson coefficient is supposed to be 0, so that lambda = 0
    and E = 2 * mu.

    To simplify computations, only the lateral surface is considered
    when computing area increase so that the cell area is proportional
    to the volume.
    """

    def __init__(self, radius: float, init_length: float, wall_modulus: float,
                 initial_electric_potential: float,
                 initial_pressure: float = 0.):
        """Initialize cylinder membrane.

        Parameters
        ----------
        radius : float
            Radius (fixed) of the cylinder in m.
        init_length : float
            Initial length of the cylinder in m.
        wall_modulus : float
            Surface elasticity modulus of the wall (in N/m).
        initial_electric_potential : float
            Initial electric potential (V) across the membrane, defined
            as the potential inside the membrane minus the potential
            outside the membrane.
        initial_pressure : float
            Initial pressure applied by the membrane to its content.
        """
        # Compute initial volume and surface area
        initial_area = 2. * pi * radius * init_length
        initial_volume = pi * init_length * radius**2
        AbstractMembrane.__init__(self, initial_volume, initial_area,
                                  initial_electric_potential,
                                  initial_pressure)

        # Coefficient for elastic energy computation
        lateral_area = 2. * pi * radius * init_length
        self.w0 = lateral_area * wall_modulus

    def area_increase(self, volume_increase: float):
        """Evaluate ratio (current_area - initial_area) / initial area.

        Here, only the lateral surface is taken into account, so that
        the surface area is proportional to the volume.

        Parameters
        ----------
        volume_increase : float
            Ratio (current_volume - initial_volume) / initial_volume.
        """
        return volume_increase

    def diff_area_increase(self, volume_increase: float):
        """Derivative of the `area_increase` function."""
        return 1.

    def elastic_energy(self, volume_increase: float):
        """Evaluate scaled elastic energy from volume increase.

        Parameters
        ----------
        volume_increase : float
            Ratio (current_volume - initial_volume) / initial_volume.
        """
        return 0.5 * self.w0 * volume_increase**2 \
            + self.initial_slope * volume_increase

    def diff_elastic_energy(self, volume_increase: float):
        """Derivative of the function elastic_energy.

        Parameters
        ----------
        volume_increase : float
            Ratio (current_volume - initial_volume) / initial_volume.
        """
        return self.w0 * volume_increase + self.initial_slope


class SphereMembrane(AbstractMembrane):
    """A sphere membrane that expands in all directions.

    In this geometry, the cell wall expands in two directions when the
    cell radius increases.
    A linear elastic deformation occurs on the whole sphere surface.
    The deformation energy is

        W = lateral_surface * wall_modulus
            * 2 * ((current_volume / initial_volume)^(1/3) - 1)

    where wall_modulus (in N/m) represents the cell wall Young modulus
    multiplied by the wall thickness.
    The Poisson coefficient is supposed to be 0, so that lambda = 0
    and E = 2 * mu.
    """

    def __init__(self, radius: float, wall_modulus: float,
                 initial_electric_potential: float,
                 initial_pressure: float = 0.):
        """Initialize cylinder membrane.

        Parameters
        ----------
        radius : float
            Radius of the sphere in m.
        wall_modulus : float
            Surface elasticity modulus of the wall (in N/m).
        initial_electric_potential : float
            Initial electric potential (V) across the membrane, defined
            as the potential inside the membrane minus the potential
            outside the membrane.
        """
        # Compute initial dimensions
        initial_area = 4. * pi * radius**2
        initial_volume = (4. / 3.) * pi * radius**3
        AbstractMembrane.__init__(self, initial_volume, initial_area,
                                  initial_electric_potential,
                                  initial_pressure)

        # Coefficient for elastic energy computation
        # The deformation now occurs in two directions
        self.w0 = 2. * initial_area * wall_modulus

    def area_increase(self, volume_increase: float):
        """Evaluate ratio (current_area - initial_area) / initial area.

        The sphere area is proportional to r**2 while the volume is
        proportional to r**3.
        The area increase formula is

            area_increase = (1. + volume_increase)^(2/3) - 1

        Parameters
        ----------
        volume_increase : float
            Ratio (current_volume - initial_volume) / initial_volume.
        """
        return cbrt(1. + volume_increase)**2 - 1.

    def diff_area_increase(self, volume_increase: float):
        """Derivative of the `area_increase` function."""
        return (2. / 3.) / cbrt(1. + volume_increase)

    def elastic_energy(self, volume_increase: float):
        """Evaluate scaled elastic energy from volume increase.

        Parameters
        ----------
        volume_increase : float
            Ratio (current_volume - initial_volume) / initial_volume.
        """
        size_increase = cbrt(volume_increase + 1.) - 1.
        return self.w0 * size_increase**2 \
            + self.initial_slope * volume_increase

    def diff_elastic_energy(self, volume_increase: float):
        """Derivative of the function elastic_energy.

        Parameters
        ----------
        volume_increase : float
            Ratio (current_volume - initial_volume) / initial_volume.
        """
        cbrt_one_plus_v = cbrt(1. + volume_increase)
        return (2. / 3.) * self.w0 * (cbrt_one_plus_v - 1.) \
            / cbrt_one_plus_v**2 \
            + self.initial_slope
