"""A few common stoma models to avoid code duplication in scripts.

This file contains first some common transporters, reactants and cell
geometries found in several stoma models, and then functions to
construct some standard stoma models.
"""
from typing import Sequence

import utils.cell_geometry as cg
import utils.commons as cm
import utils.static_model as stm

###############################################################################
# Common components
###############################################################################

# === Cell geometries ===
# Cell geometries representing the two membranes of a plant cell.

# The plasma membrane (OUT_CELL) is coupled with the cell wall and has
# a nonzero wall Young modulus.
OUT_CELL = cg.CylinderMembrane(
    radius=5e-6,
    init_length=50e-6,
    wall_modulus=30.,
    initial_electric_potential=-50e-3,
    initial_pressure=0.5e6
)

# The vacuole membrane (IN_CELL) encloses a smaller volume than the
# plasma membrane and cannot offer mechanical resistance.
IN_CELL = cg.CylinderMembrane(
    radius=5e-6,
    init_length=15e-6,
    wall_modulus=0.,
    initial_electric_potential=40e-3
)

# === Reactants ===
# Ion species used in common stoma models, namely hydrogen, chloride and
# potassium.
# Water is added implicitly when creating the model.
#
# These Reactant objects are tailored for two-membrane models, they need
# transformation for use in a one-membrane-model.


def concentration_from_ph(ph: float) -> float:
    """Compute a concentration in mmol/L from a pH value."""
    return 1000. * 10. ** -ph


HYDROGEN = cm.Reactant(
    name="hydrogen",
    charge=1.,
    external_concentration=concentration_from_ph(5.7),
    concentrations=(
        concentration_from_ph(7.1),
        concentration_from_ph(5.8)
    )
)
CHLORIDE = cm.Reactant(
    name="chloride",
    charge=-1.,
    external_concentration=30.,
    concentrations=(15., 20.)
)
POTASSIUM = cm.Reactant(
    name="potassium",
    charge=1.,
    external_concentration=30.,
    concentrations=(15., 20.)
)

# === Transporters ===
# Plasma membrane transporters
H_PUMP_PM = cm.Transporter(
    active=True,
    stoichiometry={'hydrogen': 1.},
    compartment_from=0,
    compartment_to=-1
)
HCL_SYMPORT_PM = cm.Transporter(
    active=False,
    stoichiometry={'hydrogen': 1., 'chloride': 2.},
    compartment_from=0,
    compartment_to=-1
)
K_CHANNEL_PM = cm.Transporter(
    active=False,
    stoichiometry={'potassium': 1.},
    compartment_from=0,
    compartment_to=-1
)

# Vacuole membrane transporters
H_PUMP_VM = cm.Transporter(
    active=True,
    stoichiometry={'hydrogen': 1.},
    compartment_from=0,
    compartment_to=1
)
HCL_ANTIPORT_VM = cm.Transporter(
    active=False,
    stoichiometry={'hydrogen': 1.,'chloride': -2.},
    compartment_from=0,
    compartment_to=1
)
K_CHANNEL_VM = cm.Transporter(
    active=False,
    stoichiometry={'potassium': 1.},
    compartment_from=0,
    compartment_to=1
)


###############################################################################
# Model creation functions (without buffering)
###############################################################################
def create_one_membrane_model(membrane: cg.AbstractMembrane = OUT_CELL):
    """Create a one-membrane model without buffering.

    The model only contains the transporters of the plasma membrane.

    Parameters
    ----------
    membrane : AbstractMembrane
        Object representing the plasma membrane of the cell.
    """
    reactants = [
        HYDROGEN.set_num_membranes(1),
        CHLORIDE.set_num_membranes(1),
        POTASSIUM.set_num_membranes(1)
    ]

    transporters = [H_PUMP_PM, HCL_SYMPORT_PM, K_CHANNEL_PM]
    return stm.DirectStaticModel((membrane,), reactants, transporters)


def create_two_membrane_model(
        membranes: Sequence[cg.AbstractMembrane] = (OUT_CELL, IN_CELL)):
    """Create a two-membrane model without buffering.

    The components from the common_models module are used.

    Parameters
    ----------
    membranes : sequence of AbstractMembrane
        Membranes to be used in the model
    """
    # Check input parameters
    if len(membranes) != 2:
        raise ValueError("'membranes' parameters should contain two elements.")
    if membranes[0].initial_volume < membranes[1].initial_volume:
        raise ValueError("Membrane volumes should be in decreasing order.")

    reactants = (HYDROGEN, CHLORIDE, POTASSIUM)
    transporters = (H_PUMP_PM, HCL_SYMPORT_PM, K_CHANNEL_PM,
                    H_PUMP_VM, HCL_ANTIPORT_VM, K_CHANNEL_VM)
    return stm.DirectStaticModel(membranes, reactants, transporters)


###############################################################################
# Buffering utilities and buffered model creation
###############################################################################
# === Buffering solution ===
def prepare_buffering_solution(
        concentrations: Sequence[float],
        compartment_indices: Sequence[int],
        multiplier: float = 1.):
    """Prepare buffer components and buffering reaction.

    Parameters 'concentrations' and 'where' represent the buffer
    concentration and whether there is a buffering reaction in each
    compartment, respectively.
    Their lengths should be equal to n_membranes, and they should not
    include the outside concentration.

    The buffering reaction reads AH_n <-> n H^+ + A^{n-}, where n is
    a multiplier.
    The multiplier is used to demultiply the buffering effect without
    the molar quantity of buffer solution to affect the system
    chemistry.

    Parameters
    ----------
    concentrations : Sequence[float]
        Buffer concentrations in both compartments.
        Outside concentration should *not* be included.
    compartment_indices : Sequence[int]
        Indices of compartments where the buffering reaction happens.
    multiplier : float
        Multiplier of the buffering effect (see above)
    """
    # Buffer components
    buffer_ah = cm.Reactant(
        name="AH",
        charge=0.,
        external_concentration=1e-8,
        concentrations=concentrations
    )
    buffer_am = cm.Reactant(
        name="A-",
        charge=-multiplier,
        external_concentration=1e-8,
        concentrations=concentrations
    )
    buffer_components = (buffer_ah, buffer_am)

    # Buffer reactions in indicated compartments
    buffer_reactions = ()
    for compartment_idx in compartment_indices:
        r = cm.Reaction(
            stoichiometry={
                'hydrogen': multiplier,
                'AH': -1.,
                'A-': 1.
            },
            compartment_idx=compartment_idx
        )
        buffer_reactions += (r,)

    return buffer_components, buffer_reactions


def create_two_membrane_buffered_model(
        buffer_concentrations: Sequence[float],
        buffer_compartment_indices: Sequence[bool] = (0,),
        buffer_multiplier: float = 1.,
        membranes: Sequence[cg.AbstractMembrane] = (OUT_CELL, IN_CELL)):
    """Create a two-membrane model with buffering.

    Parameters
    ----------
    buffer_concentrations : Sequence[float]
        Buffer concentrations in both compartments.
        Outside concentration should *not* be included.
    buffer_compartment_indices : Sequence[int]
        Indices of compartments where the buffering reaction happens.
    buffer_multiplier : float
        Multiplier of the buffering effect.
    membranes : sequence of AbstractMembrane
        Membranes to be used in the model
    """
    # Check input parameters
    if len(buffer_concentrations) != 2:
        raise ValueError("'buffer_concentrations' should be of length 2.")
    if any(idx >= 2 for idx in buffer_compartment_indices):
        raise ValueError("Buffer compartment indices should be <= 2")
    if len(membranes) != 2:
        raise ValueError("'membranes' parameters should contain two elements.")
    if membranes[0].initial_volume < membranes[1].initial_volume:
        raise ValueError("Membrane volumes should be in decreasing order.")

    # Create buffer components
    buffer_components, buffer_reactions = prepare_buffering_solution(
        concentrations=buffer_concentrations,
        compartment_indices=buffer_compartment_indices,
        multiplier=buffer_multiplier
    )

    reactants = (HYDROGEN, CHLORIDE, POTASSIUM) + buffer_components
    transporters = (H_PUMP_PM, HCL_SYMPORT_PM, K_CHANNEL_PM,
                    H_PUMP_VM, HCL_ANTIPORT_VM, K_CHANNEL_VM)
    return stm.DirectStaticModel(
        membranes,
        reactants,
        transporters,
        reactions=buffer_reactions
    )
