#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
import math
# Import Pyomo libraries
from pyomo.environ import (
    Set,
    Var,
    check_optimal_termination,
    Param,
    Suffix,
    NonNegativeReals,
    value,
    log,
    Constraint,
    sqrt,
    units as pyunits,
)
from pyomo.dae import (
    DerivativeVar,
)
from pyomo.common.config import Bool, ConfigBlock, ConfigValue, In

# Import Watertap cores
from watertap.core.util.initialization import check_solve, check_dof

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.misc import add_object_reference
from watertap.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.math import smooth_min

from idaes.core.util.exceptions import ConfigurationError, InitializationError

import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.constants import Constants
from enum import Enum

from watertap.core import ControlVolume1DBlock, InitializationMixin
from bipolar_electrodialysis_costing import cost_bipolar_electrodialysis

__author__ = " Xiangyu Bi, Austin Ladshaw, Kejia Hu, Johnson Dhanasekaran"

_log = idaeslog.getLogger(__name__)

class LimitingCurrentDensitybpemMethod(Enum):
    InitialValue = 0
    Empirical = 1


class LimitingpotentialMethod(Enum):
    InitialValue = 0
    Empirical = 1


class LimitingCurrentDensityMethod(Enum):
    InitialValue = 0
    Empirical = 1
    Theoretical = 2


class ElectricalOperationMode(Enum):
    Constant_Current = 0
    Constant_Voltage = 1


class PressureDropMethod(Enum):
    none = 0
    experimental = 1
    Darcy_Weisbach = 2


class FrictionFactorMethod(Enum):
    fixed = 0
    Gurreri = 1
    Kuroda = 2


class HydraulicDiameterMethod(Enum):
    fixed = 0
    spacer_specific_area_known = 1
    conventional = 2


# Name of the unit model
@declare_process_block_class("Bipolar_and_Electrodialysis1D")
class Bipolar_and_Electrodialysis1DData(InitializationMixin, UnitModelBlockData):
    """
    0D Bipolar and Electrodialysis Model
    """

    # CONFIG are options for the unit model
    CONFIG = ConfigBlock()  #

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False. The filtration unit does not support dynamic
    behavior, thus this must be False.""",
        ),
    )

    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. The filtration unit does not have defined volume, thus
    this must be False.""",
        ),
    )
    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
    constructed,
    **default** - False.
    **Valid values:** {
    **True** - include pressure change terms,
    **False** - exclude pressure change terms.}""",
        ),
    )
    CONFIG.declare(
        "pressure_drop_method",
        ConfigValue(
            default=PressureDropMethod.none,
            domain=In(PressureDropMethod),
            description="Method to calculate the frictional pressure drop in electrodialysis channels",
            doc="""
     **default** - ``PressureDropMethod.none``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``PressureDropMethod.none``", "The frictional pressure drop is neglected." 
           "``PressureDropMethod.experimental``", "The pressure drop is calculated by an experimental data as pressure drop per unit lenght."
           "``PressureDropMethod.Darcy_Weisbach``", "The pressure drop is calculated by the Darcy-Weisbach equation."
       """,
        ),
    )
    CONFIG.declare(
        "friction_factor_method",
        ConfigValue(
            default=FrictionFactorMethod.fixed,
            domain=In(FrictionFactorMethod),
            description="Method to calculate the Darcy's friction factor",
            doc="""
     **default** - ``FrictionFactorMethod.fixed``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``FrictionFactorMethod.fixed``", "Friction factor is fixed by users" 
           "``FrictionFactorMethod.Gurreri``", "Friction factor evaluated based on Gurreri's work"
           "``FrictionFactorMethod.Kuroda``", "Friction factor evaluated based on Kuroda's work"
       """,
        ),
    )

    CONFIG.declare(
        "hydraulic_diameter_method",
        ConfigValue(
            default=HydraulicDiameterMethod.conventional,
            domain=In(HydraulicDiameterMethod),
            description="Method to calculate the hydraulic diameter for a rectangular channel in ED",
            doc="""
     **default** - ``HydraulicDiameterMethod.conventional``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``HydraulicDiameterMethod.fixed``", "Hydraulic diameter is fixed by users" 
           "``HydraulicDiameterMethod.conventional``", "Conventional method for a rectangular channel with spacer porosity considered" 
           "``HydraulicDiameterMethod.spacer_specific_area_known``", "A method for spacer-filled channel requiring the spacer specific area data"
       """,
        ),
    )

    CONFIG.declare(
        "operation_mode",
        ConfigValue(
            default=ElectricalOperationMode.Constant_Current,
            domain=In(ElectricalOperationMode),
            description="The electrical operation mode. To be selected between Constant Current and Constant Voltage",
        ),
    )

    CONFIG.declare(
        "limiting_current_density_method",
        ConfigValue(
            default=LimitingCurrentDensityMethod.InitialValue,
            domain=In(LimitingCurrentDensityMethod),
            description="Configuration for method to compute the limiting current density",
            doc="""
           **default** - ``LimitingCurrentDensityMethod.InitialValue``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``LimitingCurrentDensityMethod.InitialValue``", "Limiting current is calculated from a single initial value of the feed solution tested by the user."
           "``LimitingCurrentDensityMethod.Empirical``", "Limiting current density is caculated from the empirical equation: TODO"
           "``LimitingCurrentDensityMethod.Theoretical``", "Limiting current density is calculated from a theoretical equation: TODO"
       """,
        ),
    )

    CONFIG.declare(
        "limiting_current_density_data",
        ConfigValue(
            default=500,
            description="Limiting current density data input",
        ),
    )

    CONFIG.declare(
        "has_nonohmic_potential_membrane",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Configuration for whether to model the nonohmic potential across ion exchange membranes",
        ),
    )

    CONFIG.declare(
        "has_Nernst_diffusion_layer",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Configuration for whether to simulate the concentration-polarized diffusion layers",
        ),
    )

    CONFIG.declare(
        "limiting_current_density_method_bpem",
        ConfigValue(
            default=LimitingCurrentDensitybpemMethod.InitialValue,
            domain=In(LimitingCurrentDensitybpemMethod),
            description="Configuration for method to compute the limiting current density",
            doc="""
               **default** - ``LimitingCurrentDensitybpemMethod.InitialValue``

           .. csv-table::
               :header: "Configuration Options", "Description"

               "``LimitingCurrentDensitybpemMethod.InitialValue``", "Limiting current is calculated from a single initial value given by the user."
               "``LimitingCurrentDensitybpemMethod.Empirical``", "Limiting current density is calculated from the empirical equation"
           """,
        ),
    )

    CONFIG.declare(
        "salt_calculation",
        ConfigValue(
            default=False,
            domain=Bool,
            description="""Salt calculation,
                    **default** - False.""",
        ),
    )


    CONFIG.declare(
        "has_catalyst",
        ConfigValue(
            default=False,
            domain=Bool,
            description="""Catalyst action on water spliting,
            **default** - False.""",
        ),
    )

    CONFIG.declare(
        "limiting_potential_method_bpem",
        ConfigValue(
            default=LimitingpotentialMethod.InitialValue,
            domain=In(LimitingpotentialMethod),
            description="Configuration for method to compute the limiting potential in bipolar membrane",
            doc="""
                   **default** - ``LimitingpotentialMethod.InitialValue``

               .. csv-table::
                   :header: "Configuration Options", "Description"

                   "``LimitingpotentialMethod.InitialValue``", "Limiting current is calculated from a initial value given by the user."
                   "``LimitingpotentialMethod.Empirical``", "Limiting current density is caculated from the empirical equation"
               """,
        ),
    )

    CONFIG.declare(
        "limiting_current_density_bpem_data",
        ConfigValue(
            default=0.5,
            description="Limiting current density data input for bipolar membrane",
        ),
    )
    CONFIG.declare(
        "salt_input_cem",
        ConfigValue(
            default=100,
            description="Specified salt concentration on acid channel of the bipolar membrane",
        ),
    )
    CONFIG.declare(
        "salt_input_aem",
        ConfigValue(
            default=100,
            description="Specified salt concentration on base channel of the bipolar membrane",
        ),
    )

    CONFIG.declare(
        "limiting_potential_data",
        ConfigValue(
            default=0.5,
            description="Limiting potential data input",
        ),
    )

    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
    **default** - MaterialBalanceType.useDefault.
    **Valid values:** {
    **MaterialBalanceType.useDefault - refer to property package for default
    balance type
    **MaterialBalanceType.none** - exclude material balances,
    **MaterialBalanceType.componentPhase** - use phase component balances,
    **MaterialBalanceType.componentTotal** - use total component balances,
    **MaterialBalanceType.elementTotal** - use total element balances,
    **MaterialBalanceType.total** - use total material balance.}""",
        ),
    )

    CONFIG.declare(
        "is_isothermal",
        ConfigValue(
            default=True,
            domain=Bool,
            description="""Assume isothermal conditions for control volume(s); energy_balance_type must be EnergyBalanceType.none,
    **default** - True.""",
        ),
    )

    CONFIG.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.none,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
    **default** - EnergyBalanceType.none.
    **Valid values:** {
    **EnergyBalanceType.useDefault - refer to property package for default
    balance type
    **EnergyBalanceType.none** - exclude energy balances,
    **EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
    **EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
    **EnergyBalanceType.energyTotal** - single energy balance for material,
    **EnergyBalanceType.energyPhase** - energy balances for each phase.}""",
        ),
    )

    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
    **default** - MomentumBalanceType.pressureTotal.
    **Valid values:** {
    **MomentumBalanceType.none** - exclude momentum balances,
    **MomentumBalanceType.pressureTotal** - single pressure balance for material,
    **MomentumBalanceType.pressurePhase** - pressure balances for each phase,
    **MomentumBalanceType.momentumTotal** - single momentum balance for material,
    **MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )

    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )

    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )

    CONFIG.declare(
        "transformation_method",
        ConfigValue(
            default="dae.finite_difference",
            description="Discretization method to use for DAE transformation",
            doc="""Discretization method to use for DAE transformation. See Pyomo
        documentation for supported transformations.""",
        ),
    )

    CONFIG.declare(
        "transformation_scheme",
        ConfigValue(
            default="BACKWARD",
            description="Discretization scheme to use for DAE transformation",
            doc="""Discretization scheme to use when transforming domain. See
        Pyomo documentation for supported schemes.""",
        ),
    )

    CONFIG.declare(
        "finite_elements",
        ConfigValue(
            default=10,
            domain=int,
            description="Number of finite elements in length domain",
            doc="""Number of finite elements to use when discretizing length
                domain (default=10)""",
        ),
    )

    CONFIG.declare(
        "collocation_points",
        ConfigValue(
            default=2,
            domain=int,
            description="Number of collocation points per finite element",
            doc="""Number of collocation points to use per finite element when
                discretizing length domain (default=2)""",
        ),
    )

    def _validate_config(self):
        if (
            self.config.is_isothermal
            and self.config.energy_balance_type != EnergyBalanceType.none
        ):
            raise ConfigurationError(
                "If the isothermal assumption is used then the energy balance type must be none"
            )

    def build(self):
        # build always starts by calling super().build()
        # This triggers a lot of boilerplate in the background for you
        super().build()
        # this creates blank scaling factors, which are populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Check configs for errors
        self._validate_config()

        # Create essential sets.
        self.membrane_set = Set(initialize=["cem", "aem", "bpem"])
        self.electrode_side = Set(initialize=["cathode_left", "anode_right"])
        add_object_reference(self, "ion_set", self.config.property_package.ion_set)

        add_object_reference(
            self, "cation_set", self.config.property_package.cation_set
        )
        add_object_reference(self, "anion_set", self.config.property_package.anion_set)
        add_object_reference(
            self, "component_set", self.config.property_package.component_list
        )
        # Create unit model parameters and vars

        self.cell_length = Var(
            initialize=0.5,
            bounds=(1e-3, 1e2),
            units=pyunits.meter,
            doc="The length of the electrodialysis cell, denoted as l in the model description",
        )

        # Control Volume for the Diluate channel:
        self.diluate = ControlVolume1DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            transformation_method=self.config.transformation_method,
            transformation_scheme=self.config.transformation_scheme,
            finite_elements=self.config.finite_elements,
            collocation_points=self.config.collocation_points,
        )
        self.diluate.add_geometry(length_var=self.cell_length)
        self.diluate.add_state_blocks(has_phase_equilibrium=False)
        self.diluate.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )

        self.diluate.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )

        if self.config.is_isothermal:
            self.diluate.add_isothermal_assumption()

        self.diluate.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )

        # Below is declared the electrical power var and its derivative var,
        # which is a performance metric of the entire electrodialysis stack.
        # This var takes the "diluate" as the parent to utilize the discretization (as in Pyomo DAE) of this block for solving.
        self.diluate.power_electrical_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0,
            bounds=(0, 12100),
            domain=NonNegativeReals,
            units=pyunits.watt,
            doc="Electrical power consumption of a stack",
        )
        self.diluate.Dpower_electrical_Dx = DerivativeVar(
            self.diluate.power_electrical_x,
            wrt=self.diluate.length_domain,
            units=pyunits.watt,
        )


        # den_mass and visc_d in diluate and concentrate channels are the same
        add_object_reference(
            self, "dens_mass", self.diluate.properties[0, 0].dens_mass_phase["Liq"]
        )
        add_object_reference(
            self, "visc_d", self.diluate.properties[0, 0].visc_d_phase["Liq"]
        )

        # Apply the discretization transformation (Pyomo DAE) to the diluate block
        self.diluate.apply_transformation()

        # Build control volume for the base channel of the bipolar channel
        self.basate = ControlVolume1DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            transformation_method=self.config.transformation_method,
            transformation_scheme=self.config.transformation_scheme,
            finite_elements=self.config.finite_elements,
            collocation_points=self.config.collocation_points,
        )
        self.basate.add_geometry(length_var=self.cell_length)
        self.basate.add_state_blocks(has_phase_equilibrium=False)
        self.basate.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )

        self.basate.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )

        if self.config.is_isothermal:
            self.basate.add_isothermal_assumption()

        self.basate.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )
        self.basate.apply_transformation()

        # Control volume for the acidate channel
        self.acidate = ControlVolume1DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            transformation_method=self.config.transformation_method,
            transformation_scheme=self.config.transformation_scheme,
            finite_elements=self.config.finite_elements,
            collocation_points=self.config.collocation_points,
        )
        self.acidate.add_geometry(length_var=self.cell_length)
        self.acidate.add_state_blocks(has_phase_equilibrium=False)
        self.acidate.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )

        self.acidate.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )

        if self.config.is_isothermal:
            self.acidate.add_isothermal_assumption()

        self.acidate.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )
        self.acidate.apply_transformation()

        # Add ports (creates inlets and outlets for each channel)
        self.add_inlet_port(name="inlet_diluate", block=self.diluate)
        self.add_outlet_port(name="outlet_diluate", block=self.diluate)
        self.add_inlet_port(name="inlet_basate", block=self.basate)
        self.add_outlet_port(name="outlet_basate", block=self.basate)
        self.add_inlet_port(name="inlet_acidate", block=self.acidate)
        self.add_outlet_port(name="outlet_acidate", block=self.acidate)

        self.water_density = Param(
            initialize=1000,
            units=pyunits.kg * pyunits.m**-3,
            doc="density of water",
        )

        self.cell_triplet_num = Var(
            initialize=1,
            domain=NonNegativeReals,
            bounds=(1, 10000),
            units=pyunits.dimensionless,
            doc="cell triplet number in a stack",
        )
        self.electrical_stage_num = Var(
            initialize=1,
            domain=NonNegativeReals,
            bounds=(1, 20*1e0),
            units=pyunits.dimensionless,
            doc="number of electrical stages in a stack",
        )

        # electrodialysis cell dimensional properties
        self.cell_width = Var(
            initialize=0.1,
            bounds=(1e-3, 1e2),
            units=pyunits.meter,
            doc="The width of the electrodialysis cell, denoted as b in the model description",
        )
        self.channel_height = Var(
            initialize=0.0001,
            units=pyunits.meter,
            doc="The distance between the consecutive aem and cem",
        )
        self.spacer_porosity = Var(
            initialize=0.7,
            bounds=(0.01, 1),
            units=pyunits.dimensionless,
            doc='The prosity of spacer in the ED channels. This is also referred to elsewhere as "void fraction" or "volume parameters"',
        )

        # Material and Operational properties
        self.membrane_thickness = Var(
            self.membrane_set,
            initialize=0.0001,
            bounds=(1e-6, 1e-1),
            units=pyunits.meter,
            doc="Membrane thickness",
        )
        self.solute_diffusivity_membrane = Var(
            self.membrane_set,
            self.ion_set | self.config.property_package.solute_set,
            initialize=1e-10,
            bounds=(0.0, 1e-6),
            units=pyunits.meter**2 * pyunits.second**-1,
            doc="Solute (ionic and neutral) diffusivity in the membrane phase",
        )
        self.ion_trans_number_membrane = Var(
            self.membrane_set,
            self.ion_set,
            initialize=0.5,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="Ion transference number in the membrane phase",
        )
        self.water_trans_number_membrane = Var(
            self.membrane_set,
            initialize=5,
            bounds=(0, 50),
            units=pyunits.dimensionless,
            doc="Transference number of water in membranes",
        )
        self.water_permeability_membrane = Var(
            self.membrane_set,
            initialize=1e-14,
            units=pyunits.meter * pyunits.second**-1 * pyunits.pascal**-1,
            doc="Water permeability coefficient",
        )
        self.membrane_areal_resistance_combined = Var(
            initialize=2e-4,
            bounds=(0, 1e3),
            units=pyunits.ohm * pyunits.meter**2,
            doc="Surface resistance of membrane",
        )
        self.total_areal_resistance_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=2e-4,
            bounds=(0, 1e3),
            units=pyunits.ohm * pyunits.meter ** 2,
            doc="Total areal resistance of a stack ",
        )
        if self.config.operation_mode == ElectricalOperationMode.Constant_Current:
            self.current_applied = Var(
                self.flowsheet().time,
                initialize=1,
                bounds=(0, 1000),
                units=pyunits.amp,
                doc="Current across a cell-pair or stack, declared under the 'Constant Current' mode only",
            )

        self.current_density_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1,
            bounds=(0, 1e6),
            units=pyunits.amp * pyunits.meter ** -2,
            doc="Current density accross the membrane as a function of the normalized length",
        )
        self.voltage_membrane_drop = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1,
            bounds=(0, 1000),
            units=pyunits.volt,
            doc="Potential drop across the bipolar membrane - qualitatively different for with and without catalyst",
        )       

        if self.config.operation_mode == ElectricalOperationMode.Constant_Voltage:
            self.voltage_applied = Var(
                self.flowsheet().time,
                initialize=100,
                bounds=(0, 2000*1e3),
                units=pyunits.volt,
                doc="Voltage across a stack, declared under the 'Constant Voltage' mode only",
            )

        self.voltage_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=100,
            bounds=(0, 2000*1e3),
            units=pyunits.volt,
            doc="Voltage across a stack",
        )

        self.voltage_integral = Var(
            self.flowsheet().time,
            # self.diluate.length_domain,
            initialize=100,
            bounds=(0, 1000),
            units=pyunits.volt,
            doc="Voltage across a stack - in fixed current operation",
        )

        self.electrodes_resistance = Var(
            initialize=0,
            bounds=(0, 100),
            domain=NonNegativeReals,
            units=pyunits.ohm * pyunits.meter**2,
            doc="areal resistance of TWO electrode compartments of a stack",
        )
        self.current_utilization = Var(
            initialize=1,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="The current utilization including water electro-osmosis and ion diffusion",
        )
        self.shadow_factor = Var(
            initialize=1,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="The reduction in area due to limited area available for flow",
        )

        # Performance metrics
        # self.current_efficiency_x = Var(
        #     self.flowsheet().time,
        #     self.diluate.length_domain,
        #     initialize=0.9,
        #     bounds=(0, 1 + 1e-10),
        #     units=pyunits.dimensionless,
        #     doc="The overall current efficiency for deionization",
        # )
        # self.power_electrical = Var(
        #     self.flowsheet().time,
        #     initialize=1,
        #     bounds=(0, 12100),
        #     domain=NonNegativeReals,
        #     units=pyunits.watt,
        #     doc="Electrical power consumption of a stack",
        # )
        self.specific_power_electrical = Var(
            self.flowsheet().time,
            initialize=10,
            bounds=(0, 1000),
            domain=NonNegativeReals,
            units=pyunits.kW * pyunits.hour * pyunits.meter**-3,
            doc="Diluate-volume-flow-rate-specific electrical power consumption",
        )
        self.recovery_mass_H2O = Var(
            self.flowsheet().time,
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="water recovery ratio calculated by mass",
        )
        self.velocity_diluate = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,
            units=pyunits.meter * pyunits.second**-1,
            doc="Linear velocity of flow in the diluate",
        )
        self.velocity_basate = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,
            units=pyunits.meter * pyunits.second ** -1,
            doc="Linear velocity of flow in the base channel of the bipolar membrane",
        )
        self.velocity_acidate = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,
            units=pyunits.meter * pyunits.second ** -1,
            doc="Linear velocity of flow in the acid channel of the bipolar membrane",
        )
        self.elec_field_non_dim = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1,
            # bounds=(0, 1e32),
            units=pyunits.dimensionless,
            doc="Limiting current density across the bipolar membrane as a function of the normalized length",
        )
        self.relative_permittivity = Var(
            initialize=30,
            bounds=(1, 80),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Relative permittivity",
        )
        self.membrane_fixed_charge = Var(
            initialize=1.5e3,
            bounds=(1e-1, 1e5),
            units=pyunits.mole * pyunits.meter ** -3,
            doc="Membrane fixed charge",
        )
        self.kr = Var(
            initialize=1.33 * 10 ** 11,
            bounds=(1e-6, 1e16),
            units=pyunits.L * pyunits.mole ** -1 * pyunits.second ** -1,
            doc="Re-association rate constant",
        )
        self.k2_zero = Var(
            initialize=2 * 10 ** -5,
            bounds=(1e-10, 1e2),
            units=pyunits.second ** -1,
            doc="Dissociation rate constant at no electric field",
        )
        self.salt_conc_aem_ref = Var(
            initialize=1e3,
            bounds=(1e-8, 1e6),
            units=pyunits.mole * pyunits.meter ** -3,
            doc="Fixed salt concentration on the base channel of the bipolar membrane",
        )
        self.salt_conc_cem_ref = Var(
            initialize=1e3,
            bounds=(1e-8, 1e6),
            units=pyunits.mole * pyunits.meter ** -3,
            doc="Fixed salt concentration on the acid channel of the bipolar membrane",
        )
        self.salt_conc_dilu_ref = Var(
            initialize=1e3,
            bounds=(1e-8, 1e6),
            units=pyunits.mole * pyunits.meter ** -3,
            doc="Fixed salt concentration on the diluate channel ",
        )
        self.salt_conc_aem_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1e3,
            bounds=(1e-8, 1e6),
            units=pyunits.mole * pyunits.meter ** -3,
            doc="Salt concentration on the base channel of the bipolar membrane",
        )
        self.salt_conc_cem_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1e3,
            bounds=(1e-6, 1e4),
            units=pyunits.mole * pyunits.meter ** -3,
            doc="Salt concentration on the acid channel of the bipolar membrane",
        )
        self.salt_conc_dilu_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1e3,
            bounds=(1e-6, 1e4),
            units=pyunits.mole * pyunits.meter ** -3,
            doc="Salt concentration on the diluate channel ",
        )
        self.current_dens_lim_bpem = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1e2,
            bounds=(0, 1e5),
            units=pyunits.amp * pyunits.meter ** -2,
            doc="Limiting current density across the bipolar membrane",
        )

        self.diffus_mass = Var(
            initialize=2e-9,
            bounds=(1e-16, 1e-6),
            units=pyunits.meter ** 2 * pyunits.second ** -1,
            doc="The mass diffusivity of the solute as molecules (not individual ions)",
        )
        self.conc_water = Var(
            initialize=55 * 1e3,
            bounds=(1e-2, 1e6),
            units=pyunits.mole * pyunits.meter ** -3,
            doc="Concentration of water within the channel",
        )
        # self.acid_produced = Var(
        #     self.flowsheet().time,
        #     initialize=55 * 1e3,
        #     bounds=(0, 1e6),
        #     units=pyunits.kg * pyunits.second ** -1,
        #     doc="Acid prodcued",
        # )
        # self.base_produced = Var(
        #     self.flowsheet().time,
        #     initialize=55 * 1e3,
        #     bounds=(0, 1e6),
        #     units=pyunits.kg * pyunits.second ** -1,
        #     doc="Base prodcued",
        # )

        # Fluxes Vars for constructing mass transfer terms
        self.generation_cem_flux = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter ** -2 * pyunits.second ** -1,
            doc="Molar flux_in of a component generated by water splitting on the acid channel of the bipolar membrane",
        )
        self.generation_aem_flux = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter ** -2 * pyunits.second ** -1,
            doc="Molar flux_in of a component generated by water splitting on the base channel of the bipolar membrane",
        )
        self.elec_migration_bpem_flux = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter ** -2 * pyunits.second ** -1,
            doc="Molar flux_in of a component across the membrane driven by electrical migration across the bipolar membrane",
        )
        self.nonelec_bpem_flux = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_in of a component across the membrane driven by non-electrical forces across the bipolar membrane",
        )

        self.elec_migration_mono_cem_flux = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter ** -2 * pyunits.second ** -1,
            doc="Molar flux_in of a component across the membrane driven by electrical migration across the monopolar CEM membrane",
        )
        self.nonelec_mono_cem_flux = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter ** -2 * pyunits.second ** -1,
            doc="Molar flux_in of a component across the membrane driven by non-electrical forces across the monopolar CEM membrane",
        )

        self.elec_migration_mono_aem_flux = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter ** -2 * pyunits.second ** -1,
            doc="Molar flux_in of a component across the membrane driven by electrical migration across the monopolar AEM membrane",
        )
        self.nonelec_mono_aem_flux = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter ** -2 * pyunits.second ** -1,
            doc="Molar flux_in of a component across the membrane driven by non-electrical forces across the monopolar AEM membrane",
        )

        if (
            self.config.has_nonohmic_potential_membrane
            or self.config.has_Nernst_diffusion_layer
        ):
            self.conc_mem_surf_mol_x = Var(
                self.membrane_set,
                self.electrode_side,
                self.flowsheet().time,
                self.diluate.length_domain,
                self.config.property_package.ion_set,
                initialize=500,
                bounds=(0, 1e5),
                units=pyunits.mol * pyunits.meter ** -3,
                doc="Membane surface concentration of components",
            )



        # extension options
        if self.config.has_catalyst == True:
            self._make_catalyst()

        if self.config.has_nonohmic_potential_membrane:
            self._make_performance_nonohm_mem()
        if self.config.has_Nernst_diffusion_layer:
            self._make_performance_dl_polarization()
        if (
            not self.config.pressure_drop_method == PressureDropMethod.none
        ) and self.config.has_pressure_change:
            self._pressure_drop_calculation()

            @self.Constraint(
                self.flowsheet().time,
                self.diluate.length_domain,
                doc="Pressure drop expression as calculated by the pressure drop data, diluate.",
            )
            def eq_deltaP_diluate(self, t, x):
                return self.diluate.deltaP[t, x] == -self.pressure_drop[t]

            @self.Constraint(
                self.flowsheet().time,
                self.diluate.length_domain,
                doc="Pressure drop expression as calculated by the pressure drop data, "
                    "base channel of the bipolar membrane.",
            )
            def eq_deltaP_basate(self, t, x):
                return self.basate.deltaP[t, x] == -self.pressure_drop[t]

            @self.Constraint(
                self.flowsheet().time,
                self.diluate.length_domain,
                doc="Pressure drop expression as calculated by the pressure drop data,"
                    "  acid channel of the bipolar membrane.",
            )
            def eq_deltaP_acidate(self, t, x):
                return self.acidate.deltaP[t, x] == -self.pressure_drop[t]

        elif self.config.pressure_drop_method == PressureDropMethod.none and (
            not self.config.has_pressure_change
        ):
            pass
        else:
            raise ConfigurationError(
                "A valid (not none) pressure_drop_method and has_pressure_change being True "
                "must be both used or unused at the same time. "
            )

        # To require H2O must be in the component
        if "H2O" not in self.component_set:
            raise ConfigurationError(
                "Property Package MUST constain 'H2O' as a component"
            )
        # Build Constraints
        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate flow velocity in a single diluate channel, based on the average of inlet and outlet",
        )
        def eq_get_velocity_diluate(self, t, x):
            return self.velocity_diluate[
                t, x] * self.cell_width * self.shadow_factor * self.channel_height * self.spacer_porosity * self.cell_triplet_num == \
                self.diluate.properties[t, x].flow_vol_phase["Liq"]

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate flow velocity in a single base channel of the bipolar membrane channel,"
                " based on the average of inlet and outlet",
        )
        def eq_get_velocity_basate(self, t, x):
            # return self.velocity_basate[t] == 0 * pyunits.meter * pyunits.second**-1

            return self.velocity_basate[
                t, x
            ] * self.cell_width * self.shadow_factor * self.channel_height * self.spacer_porosity * self.cell_triplet_num == self.basate.properties[t, x].flow_vol_phase["Liq"]


        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate flow velocity in a single acid channel of the bipolar membrane channel,"
                " based on the average of inlet and outlet",
        )
        def eq_get_velocity_acidate(self, t, x):
            # return self.velocity_acidate[t] == 0 * pyunits.meter * pyunits.second**-1

            return self.velocity_acidate[
                t, x] * self.cell_width * self.shadow_factor * self.channel_height * self.spacer_porosity * self.cell_triplet_num == \
                self.acidate.properties[t, x].flow_vol_phase["Liq"]



        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Evaluate salt concentration on AEM side of the bipolar membrane",
        )
        def eq_salt_aem(self, t, x):
            if self.config.salt_calculation:
                conc_unit = 1 * pyunits.mole * pyunits.meter ** -3

                return self.salt_conc_aem_x[t, x] == smooth_min(
                    self.basate.properties[t, x].conc_mol_phase_comp["Liq", "Na_+"] / conc_unit,
                    self.basate.properties[t, x].conc_mol_phase_comp[
                        "Liq", "Cl_-"] / conc_unit) * conc_unit
            else:
                return self.salt_conc_aem_x[t, x] == self.salt_conc_aem_ref

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Evaluate salt concentration on CEM side of the bipolar membrane",
        )
        def eq_salt_cem(self, t, x):

            if self.config.salt_calculation:
                conc_unit = 1 * pyunits.mole * pyunits.meter ** -3

                return self.salt_conc_cem_x[t, x] == smooth_min(
                    self.acidate.properties[t, x].conc_mol_phase_comp["Liq", "Na_+"] / conc_unit,
                    self.acidate.properties[t, x].conc_mol_phase_comp[
                        "Liq", "Cl_-"] / conc_unit) * conc_unit
            else:
                return self.salt_conc_cem_x[t, x] == self.salt_conc_cem_ref

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Evaluate salt concentration on CEM side of the bipolar membrane",
        )
        def eq_salt_dilu(self, t, x):

            if self.config.salt_calculation:
                conc_unit = 1 * pyunits.mole * pyunits.meter ** -3

                return self.salt_conc_dilu_x[t, x] == smooth_min(
                    self.diluate.properties[t, x].conc_mol_phase_comp["Liq", "Na_+"] / conc_unit,
                    self.diluate.properties[t, x].conc_mol_phase_comp[
                        "Liq", "Cl_-"] / conc_unit) * conc_unit
            else:
                return self.salt_conc_dilu_x[t, x] == self.salt_conc_dilu_ref



        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate limiting current density across the bipolar membrane",
        )
        def eq_current_dens_lim_bpem(self, t, x):
            if (
                    self.config.limiting_current_density_method_bpem
                    == LimitingCurrentDensitybpemMethod.InitialValue
            ):
                return self.current_dens_lim_bpem[t,x] == (
                        self.config.limiting_current_density_bpem_data
                        * pyunits.amp
                        * pyunits.meter ** -2
                )
            elif (
                    self.config.limiting_current_density_method_bpem
                    == LimitingCurrentDensitybpemMethod.Empirical
            ):
                return self.current_dens_lim_bpem[t, x] == self.diffus_mass * Constants.faraday_constant * (
                        (self.salt_conc_aem_x[t, x] + self.salt_conc_cem_x[t, x]) * 0.5
                ) ** 2 / (
                        self.membrane_thickness["bpem"] * self.membrane_fixed_charge
                )

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the potential drops across the bipolar membrane",
        )
        def eq_voltage_membrane_drop_no_catalyst(self, t, x):
            if self.config.has_catalyst == True:
                # self.voltage_membrane_drop[t].fix(0 * pyunits.volt)
                return Constraint.Skip
            else:

                if (
                        self.config.limiting_potential_method_bpem
                        == LimitingpotentialMethod.InitialValue
                ):
                    return self.voltage_membrane_drop[t, x] == (
                            self.config.limiting_potential_data * pyunits.volt
                    )

                elif (
                        self.config.limiting_potential_method_bpem
                        == LimitingpotentialMethod.Empirical
                ):
                    #   [H+][OH-] concentration
                    kw = 10 ** -8 * pyunits.mol ** 2 * pyunits.meter ** -6

                    # Fraction of threshold of limiting current: currently 0.1 i_lim
                    frac = 1 * 10 ** -1
                    # Dimensional pre-factor to evaulate non-dimensional electric field
                    const = 0.0936 * pyunits.K ** 2 * pyunits.volt ** -1 * pyunits.meter

                    @self.Constraint(
                        self.flowsheet().time,
                        self.diluate.length_domain,
                        doc="Calculate the non-dimensional potential drop",
                    )
                    def eq_voltage_membrane_drop_non_dim_no_catalyst(self, t, x):
                        # [y2, qty_une, qty_deux, qty_trois] = dat
                        terms = 40
                        matrx = 0
                        for indx in range(terms):
                            # rev_indx = terms - indx - 1
                            matrx += (
                                    2 ** indx
                                    * self.elec_field_non_dim[t, x] ** indx
                                    / (math.factorial(indx) * math.factorial(indx + 1))
                            )

                        matrx *= self.k2_zero * self.conc_water
                        matrx += (
                                -pyunits.convert(
                                    self.kr,
                                    to_units=pyunits.meter ** 3
                                             * pyunits.mole ** -1
                                             * pyunits.second ** -1,
                                )
                                * kw
                        )
                        return (
                                Constants.vacuum_electric_permittivity
                                * self.relative_permittivity ** 2
                                * self.basate.properties[t, x].temperature ** 2
                                * Constants.avogadro_number
                                * Constants.elemental_charge
                        ) / (
                                const
                                * Constants.faraday_constant
                                * self.membrane_fixed_charge["bpem"]
                        ) * matrx * self.elec_field_non_dim[
                            t, x
                        ] == self.current_dens_lim_bpem[
                            t, x
                        ] * frac

                    # Dimensional electric field
                    field_generated = (
                            self.elec_field_non_dim[t, x]
                            * self.relative_permittivity
                            * self.basate.properties[t, x].temperature ** 2
                            / const
                    )

                    # Depletion length at the junction of the bipolar membrane
                    lambda_depletion = (
                            (
                                    self.elec_field_non_dim[t, x]
                                    * self.relative_permittivity
                                    * self.basate.properties[t, x].temperature ** 2
                                    / const
                            )
                            * Constants.vacuum_electric_permittivity
                            * self.relative_permittivity
                            / (
                                    Constants.faraday_constant
                                    * self.membrane_fixed_charge["bpem"]
                            )
                    )

                    return (
                            self.voltage_membrane_drop[t, x] == field_generated * lambda_depletion
                    )

                else:
                    self.voltage_membrane_drop[t, x].fix(0 * pyunits.volt)
                    return Constraint.Skip

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total areal resistance of a stack",
        )
        def eq_get_total_areal_resistance_x(self, t, x):
            if self.config.has_Nernst_diffusion_layer:
                return self.total_areal_resistance_x[t, x] == (
                        (
                                pyunits.ohm * pyunits.meter ** 2 * ((0.108 * pyunits.kg * pyunits.meter ** -3 / (
                                    self.acidate.properties[t, x].conc_mass_phase_comp["Liq", "H_+"] +
                                    self.acidate.properties[t, x].conc_mass_phase_comp["Liq", "Cl_-"] +
                                    self.basate.properties[t, x].conc_mass_phase_comp["Liq", "Na_+"] +
                                    self.basate.properties[t, x].conc_mass_phase_comp["Liq", "OH_-"]) + 0.0492) / 5)

                                + (
                                        self.channel_height
                                        - self.dl_thickness_x["cem", "cathode_left", t, x]
                                )
                                * self.basate.properties[t, x].elec_cond_phase["Liq"] ** -1
                                + (
                                        self.channel_height
                                        - self.dl_thickness_x["aem", "anode_right", t, x]
                                )
                                * self.acidate.properties[t, x].elec_cond_phase["Liq"] ** -1
                                + (
                                        self.channel_height
                                        - self.dl_thickness_x["cem", "anode_right", t, x]
                                        - self.dl_thickness_x["aem", "cathode_left", t, x]
                                )
                                * self.diluate.properties[t, x].elec_cond_phase["Liq"] ** -1
                        )
                        * self.cell_triplet_num/self.electrical_stage_num
                        + self.electrodes_resistance
                )
            else:
                return self.total_areal_resistance_x[t, x] == (
                        (
                                pyunits.ohm * pyunits.meter ** 2 * ((0.108 * pyunits.kg * pyunits.meter ** -3 / (
                                self.acidate.properties[t, x].conc_mass_phase_comp["Liq", "H_+"] +
                                self.acidate.properties[t, x].conc_mass_phase_comp["Liq", "Cl_-"] +
                                self.basate.properties[t, x].conc_mass_phase_comp["Liq", "Na_+"] +
                                self.basate.properties[t, x].conc_mass_phase_comp["Liq", "OH_-"]) + 0.0492) / 5)
                                + self.channel_height
                                * (
                                        self.acidate.properties[t, x].elec_cond_phase["Liq"]** -1
                                        + self.basate.properties[t, x].elec_cond_phase["Liq"]** -1
                                        + self.diluate.properties[t, x].elec_cond_phase["Liq"] ** -1
                                )
                        )
                        * self.cell_triplet_num/self.electrical_stage_num
                        + self.electrodes_resistance
                )

        # @self.Constraint(
        #     self.flowsheet().time,
        #     self.diluate.length_domain,
        #     doc="calcualte the Potential drop across the bipolar membrane - qualitatively different for with and without catalyst",
        # )
        # def eq_voltage_membrane_drop(self, t, x):
        #     if self.config.has_catalyst:
        #         return self.voltage_membrane_drop[t, x] == self.potential_membrane_bpem[t, x]
        #     else:
        #         return self.voltage_membrane_drop[t,x] == self.potential_barrier_bpem[t, x]

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="calcualte current density from the electrical input",
        )
        def eq_get_current_density(self, t, x):

            if self.config.has_catalyst:
                @self.Constraint(
                    self.flowsheet().time,
                    self.diluate.length_domain,
                    doc="Calculate total current generated via catalyst action",
                )
                def eq_current_relationship(self, t, x):
                    return self.current_density_x[t, x] == (
                            self.current_dens_lim_bpem[t, x]
                            + self.flux_splitting[t, x] * Constants.faraday_constant
                    )

            if self.config.operation_mode == ElectricalOperationMode.Constant_Current:
                return (
                        self.current_density_x[t, x] * self.cell_width * self.shadow_factor * self.diluate.length
                        == self.current_applied[t]
                )
            else:

                if self.config.has_nonohmic_potential_membrane:
                    if self.config.has_Nernst_diffusion_layer:
                        return (
                                self.current_density_x[t, x]
                                * self.total_areal_resistance_x[t, x]
                                + (
                                        self.potential_ohm_dl_x["cem", t, x]
                                        + self.potential_ohm_dl_x["aem", t, x]
                                        + self.potential_nonohm_dl_x["cem", t, x]
                                        + self.potential_nonohm_dl_x["aem", t, x]
                                        + self.potential_nonohm_membrane_x["cem", t, x]
                                        + self.potential_nonohm_membrane_x["aem", t, x]
                                        + self.voltage_membrane_drop[t,x]
                                )
                                * self.cell_triplet_num/self.electrical_stage_num
                                == self.voltage_applied[t]
                        )
                    else:
                        return (
                                self.current_density_x[t, x]
                                * self.total_areal_resistance_x[t, x]
                                + (
                                        self.potential_nonohm_membrane_x["cem", t, x]
                                        + self.potential_nonohm_membrane_x["aem", t, x]
                                        + self.voltage_membrane_drop[t,x]
                                )
                                * self.cell_triplet_num/self.electrical_stage_num
                                == self.voltage_applied[t]
                        )
                else:
                    if self.config.has_Nernst_diffusion_layer:
                        return (
                                self.current_density_x[t, x]
                                * self.total_areal_resistance_x[t, x]
                                + (
                                        self.potential_ohm_dl_x["cem", t, x]
                                        + self.potential_ohm_dl_x["aem", t, x]
                                        + self.potential_nonohm_dl_x["cem", t, x]
                                        + self.potential_nonohm_dl_x["aem", t, x]
                                        + self.voltage_membrane_drop[t,x]
                                )
                                * self.cell_triplet_num/self.electrical_stage_num
                                == self.voltage_applied[t]
                        )
                    else:
                        return (
                                self.current_density_x[t, x]
                                * self.total_areal_resistance_x[t, x]
                                + self.voltage_membrane_drop[t,x] * self.cell_triplet_num/self.electrical_stage_num
                                == self.voltage_applied[t]
                        )

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="calcualte length_indexed voltage",
        )
        def eq_get_voltage_x(self, t, x):
            if self.config.has_nonohmic_potential_membrane:
                if self.config.has_Nernst_diffusion_layer:
                    return (
                            self.current_density_x[t, x]
                            * self.total_areal_resistance_x[t, x]
                            + (
                                    self.potential_ohm_dl_x["cem", t, x]
                                    + self.potential_ohm_dl_x["aem", t, x]
                                    + self.potential_nonohm_dl_x["cem", t, x]
                                    + self.potential_nonohm_dl_x["aem", t, x]
                                    + self.potential_nonohm_membrane_x["cem", t, x]
                                    + self.potential_nonohm_membrane_x["aem", t, x]
                                    + self.voltage_membrane_drop[t,x]
                            )
                            * self.cell_triplet_num/self.electrical_stage_num
                            == self.voltage_x[t, x]
                    )
                else:
                    return (
                            self.current_density_x[t, x]
                            * self.total_areal_resistance_x[t, x]
                            + (
                                    self.potential_nonohm_membrane_x["cem", t, x]
                                    + self.potential_nonohm_membrane_x["aem", t, x]
                                    + self.voltage_membrane_drop[t, x]
                            )
                            * self.cell_triplet_num/self.electrical_stage_num
                            == self.voltage_x[t, x]
                    )
            else:
                if self.config.has_Nernst_diffusion_layer:
                    return (
                            self.current_density_x[t, x]
                            * self.total_areal_resistance_x[t, x]
                            + (
                                    self.potential_ohm_dl_x["cem", t, x]
                                    + self.potential_ohm_dl_x["aem", t, x]
                                    + self.potential_nonohm_dl_x["cem", t, x]
                                    + self.potential_nonohm_dl_x["aem", t, x]
                                    + self.voltage_membrane_drop[t, x]
                            )
                            * self.cell_triplet_num/self.electrical_stage_num
                            == self.voltage_x[t, x]
                    )
                else:
                    return (
                            self.current_density_x[t, x]
                            * self.total_areal_resistance_x[t, x]
                            + self.voltage_membrane_drop[t, x] * self.cell_triplet_num/self.electrical_stage_num
                            == self.voltage_x[t, x]
                    )

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for water splitting acid channel of bipolar membrane flux_in",
        )
        def eq_generation_cem_flux(self, t, x, p, j):
            if j == "H_+":
                if self.config.has_catalyst == True:
                    return (
                            self.generation_cem_flux[t, x, p, j]
                            == self.flux_splitting[t, x]
                    )

                else:
                    return self.generation_cem_flux[t, x, p, j] == (
                            -smooth_min(
                                -(
                                        self.current_density_x[t, x] * pyunits.amp ** -1* pyunits.meter ** 2
                                        - self.current_dens_lim_bpem[t, x,]* pyunits.amp ** -1* pyunits.meter ** 2
                                ),
                                0,
                            )
                            * pyunits.amp * pyunits.meter ** -2
                    ) / ( Constants.faraday_constant
                    )

            else:
                if j == "H2O":
                    if self.config.has_catalyst == True:
                        return (
                                self.generation_cem_flux[t, x, p, j]
                                == -0.5 * self.flux_splitting[t, x]
                        )
                    else:
                        return self.generation_cem_flux[t, x, p, j] == (
                            smooth_min(
                                -(
                                        self.current_density_x[t, x] * pyunits.amp ** -1* pyunits.meter ** 2
                                        - self.current_dens_lim_bpem[t, x,]* pyunits.amp ** -1* pyunits.meter ** 2
                                ),
                                0,
                            )
                            * pyunits.amp * pyunits.meter ** -2
                    ) / ( Constants.faraday_constant
                    )

                else:
                   self.generation_cem_flux[t, x, p, j].fix(0)
                   return Constraint.Skip

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for water splitting base channel of bipolar membrane flux_in",
        )
        def eq_generation_aem_flux(self, t, x, p, j):
            if j == "OH_-":
                if self.config.has_catalyst == True:
                    return (
                            self.generation_aem_flux[t, x, p, j]
                            == self.flux_splitting[t, x]
                    )

                else:
                    return self.generation_aem_flux[t, x, p, j] == (
                            -smooth_min(
                                -(
                                        self.current_density_x[t, x] * pyunits.amp ** -1* pyunits.meter ** 2
                                        - self.current_dens_lim_bpem[t, x,]* pyunits.amp ** -1* pyunits.meter ** 2
                                ),
                                0,
                            )
                            * pyunits.amp * pyunits.meter ** -2
                    ) / ( Constants.faraday_constant
                    )

            else:
                if j == "H2O":
                    if self.config.has_catalyst == True:
                        return (
                                self.generation_aem_flux[t, x, p, j]
                                == - 0.5 * self.flux_splitting[t, x]
                        )

                    else:
                        return self.generation_aem_flux[t, x, p, j] == (
                            smooth_min(
                                -(
                                        self.current_density_x[t, x] * pyunits.amp ** -1* pyunits.meter ** 2
                                        - self.current_dens_lim_bpem[t, x,]* pyunits.amp ** -1* pyunits.meter ** 2
                                ),
                                0,
                            )
                            * pyunits.amp * pyunits.meter ** -2
                    ) / ( Constants.faraday_constant
                    )

                else:
                    self.generation_aem_flux[t, x, p, j].fix(0)
                    return Constraint.Skip


        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for electrical migration across the monopolar CEM flux_in",
        )
        def eq_elec_migration_mono_cem(self, t, x, p, j):
            if j == "H2O":
                return self.elec_migration_mono_cem_flux[t, x, p, j] == (
                        self.water_trans_number_membrane["cem"]
                ) * (
                        self.current_density_x[t, x]
                        / Constants.faraday_constant
                )
            elif j in self.ion_set:
                return self.elec_migration_mono_cem_flux[t, x, p, j] == (
                        self.ion_trans_number_membrane["cem", j]
                ) * (
                        self.current_utilization
                        * self.current_density_x[t, x]
                ) / (
                        self.config.property_package.charge_comp[j]
                        * Constants.faraday_constant
                )
            else:
                self.elec_migration_mono_cem_flux[t, x, p, j].fix(0)
                return Constraint.Skip

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for electrical migration across the monopolar AEM flux_in",
        )
        def eq_elec_migration_mono_aem_flux(self, t, x, p, j):
            if j == "H2O":
                return self.elec_migration_mono_aem_flux[t, x, p, j] == (
                    self.water_trans_number_membrane["aem"]
                ) * (
                    self.current_density_x[t, x]
                    / Constants.faraday_constant
                )
            elif j in self.ion_set:
                return self.elec_migration_mono_aem_flux[t, x, p, j] == (
                    - self.ion_trans_number_membrane["aem", j]
                ) * (
                    self.current_utilization
                    * self.current_density_x[t, x]
                ) / (
                    self.config.property_package.charge_comp[j]
                    * Constants.faraday_constant
                )
            else:
                self.elec_migration_mono_aem_flux[t, x, p, j].fix(0)
                return Constraint.Skip

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for electrical migration across the bipolar membrane flux_in",
        )
        def eq_elec_migration_bpem_flux(self, t, x, p, j):
            if j == "H2O":
                return self.elec_migration_bpem_flux[t, x, p, j] == (
                    self.water_trans_number_membrane["bpem"]
                ) * (
                        self.current_density_x[t, x]
                        / Constants.faraday_constant
                )

            elif j in self.ion_set:
                if not (j == "H_+" or j == "OH_-"):
                    if self.config.has_catalyst == False:


                        return (self.elec_migration_bpem_flux[t, x, p, j] ==
                                # 0 * pyunits.mol * pyunits.m ** -2 * pyunits.s ** -1
                        (
                            self.ion_trans_number_membrane["bpem", j]
                        ) * (
                                self.current_utilization
                                * smooth_min(
                            self.current_density_x[t, x] * pyunits.amp ** -1* pyunits.meter ** 2,
                            self.current_dens_lim_bpem[t, x]* pyunits.amp ** -1* pyunits.meter ** 2,
                        )
                                * pyunits.amp * pyunits.meter ** -2
                        ) / (
                                self.config.property_package.charge_comp[j]
                                * Constants.faraday_constant
                        )
                                )
                    else:
                        return (self.elec_migration_bpem_flux[t, x, p, j] ==
                        #         (
                        #     self.ion_trans_number_membrane["bpem", j]
                        # ) *
                                0.5 * (
                                self.current_utilization * self.current_dens_lim_bpem[t, x]
                        ) / (
                                self.config.property_package.charge_comp[j]
                                * Constants.faraday_constant
                        ))

                else:

                    self.elec_migration_bpem_flux[t, x, p, j].fix(
                        0 * pyunits.mol * pyunits.m ** -2 * pyunits.s ** -1
                    )
                    return Constraint.Skip
            else:
                self.elec_migration_bpem_flux[t, x, p, j].fix(
                    0 * pyunits.mol * pyunits.m ** -2 * pyunits.s ** -1
                )
                return Constraint.Skip


        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for non-electrical flux across the monopolar CEM flux_in",
        )
        def eq_nonelec_mono_cem_flux(self, t, x, p, j):
            if j == "H2O":
                if self.config.has_Nernst_diffusion_layer:
                    return self.nonelec_mono_cem_flux[
                        t, x, p, j
                    ] == self.water_density / self.config.property_package.mw_comp[
                        j
                    ] * (
                            self.water_permeability_membrane["cem"]
                    ) * (
                            self.basate.properties[t, x].pressure_osm_phase[p]
                            * (
                                    1
                                    + self.current_density_x[t, x]
                                    / self.current_dens_lim_x[t, x]
                            )
                            - self.diluate.properties[t, x].pressure_osm_phase[p]
                            * (
                                    1
                                    - self.current_density_x[t, x]
                                    / self.current_dens_lim_x[t, x]
                            )
                    )
                else:
                    return self.nonelec_mono_cem_flux[
                        t, x, p, j
                    ] == self.water_density / self.config.property_package.mw_comp[
                        j
                    ] * (
                            self.water_permeability_membrane["cem"]
                    ) * (
                            self.basate.properties[t, x].pressure_osm_phase[p]
                            - self.diluate.properties[t, x].pressure_osm_phase[p]
                    )

            else:
                if self.config.has_Nernst_diffusion_layer:
                    return self.nonelec_mono_cem_flux[t, x, p, j] == -(
                            self.solute_diffusivity_membrane["cem", j]
                            * self.membrane_thickness["cem"] ** -1
                            * (
                                    self.conc_mem_surf_mol_x["cem", "cathode_left", t, x, j]
                                    - self.conc_mem_surf_mol_x["cem", "anode_right", t, x, j]
                            )
                    )

                else:
                    return self.nonelec_mono_cem_flux[t, x, p, j] == -(
                            self.solute_diffusivity_membrane["cem", j]
                            / self.membrane_thickness["cem"]
                    ) * (
                            self.basate.properties[t, x].conc_mol_phase_comp[p, j]
                            - self.diluate.properties[t, x].conc_mol_phase_comp[p, j]
                    )

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for non-electrical flux across the monopolar AEM flux_in",
        )
        def eq_nonelec_mono_aem_flux(self, t, x, p, j):
            if j == "H2O":
                if self.config.has_Nernst_diffusion_layer:
                    return self.nonelec_mono_aem_flux[
                        t, x, p, j
                    ] == self.water_density / self.config.property_package.mw_comp[
                        j
                    ] * (
                        self.water_permeability_membrane["aem"]
                    ) * (
                        self.acidate.properties[t, x].pressure_osm_phase[p]
                        * (
                            1
                            + self.current_density_x[t, x]
                                    / self.current_dens_lim_x[t, x]
                        )
                        - self.diluate.properties[t, x].pressure_osm_phase[p]
                        * (
                            1
                            - self.current_density_x[t, x]
                                    / self.current_dens_lim_x[t, x]
                        )
                    )
                else:
                    return self.nonelec_mono_aem_flux[
                        t, x, p, j
                    ] == self.water_density / self.config.property_package.mw_comp[
                        j
                    ] * (
                        self.water_permeability_membrane["aem"]
                    ) * (
                        self.acidate.properties[t, x].pressure_osm_phase[p]
                        - self.diluate.properties[t, x].pressure_osm_phase[p]
                    )

            else:
                if self.config.has_Nernst_diffusion_layer:
                    return self.nonelec_mono_aem_flux[t, x, p, j] == -(
                        self.solute_diffusivity_membrane["aem", j]
                        * self.membrane_thickness["aem"] ** -1
                        * (
                            self.conc_mem_surf_mol_x["aem", "anode_right", t, x, j]
                            - self.conc_mem_surf_mol_x["aem", "cathode_left", t, x, j]
                        )
                    )

                else:
                    return self.nonelec_mono_aem_flux[t, x, p, j] == -(
                        self.solute_diffusivity_membrane["aem", j]
                        / self.membrane_thickness["aem"]
                    ) * (
                        self.acidate.properties[t, x].conc_mol_phase_comp[p, j]
                        - self.diluate.properties[t, x].conc_mol_phase_comp[p, j]
                    )

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for non-electrical flux across the bipolar membrane flux_in",
        )
        def eq_nonelec_bpem_flux(self, t, x, p, j):
            if j == "H2O":
                return self.nonelec_bpem_flux[
                    t, x, p, j
                ] == self.water_density / self.config.property_package.mw_comp[j] * (
                    self.water_permeability_membrane["bpem"]
                ) * (
                        self.basate.properties[t, x].pressure_osm_phase[p]
                        - self.acidate.properties[t, x].pressure_osm_phase[p]
                )

            else:
                self.nonelec_bpem_flux[t, x, p, j].fix(0)
                return Constraint.Skip


        # Add constraints for mass transfer terms (diluate)
        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term for the diluate channel",
        )
        def eq_mass_transfer_term_diluate(self, t, x, p, j):
            return (
                self.diluate.mass_transfer_term[t, x, p, j]
                == -(
                    self.elec_migration_mono_aem_flux[t, x, p, j]
                    + self.elec_migration_mono_cem_flux[t, x, p, j]
                    + self.nonelec_mono_aem_flux[t, x, p, j]
                    + self.nonelec_mono_cem_flux[t, x, p, j]
                )
                * (self.cell_width * self.shadow_factor)
                * self.cell_triplet_num/self.electrical_stage_num
            )

        # Add constraints for mass transfer terms (base channel of the bipolar membrane)
        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term for the base channel of the bipolar membrane",
        )
        def eq_mass_transfer_term_basate(self, t, x, p, j):
            return (
                    self.basate.mass_transfer_term[t, x, p, j]
                    ==(
                            self.generation_aem_flux[t, x, p, j]
                            - self.elec_migration_bpem_flux[t, x, p, j]
                            - self.nonelec_bpem_flux[t, x, p, j]
                            + self.elec_migration_mono_cem_flux[t, x, p, j]
                            + self.nonelec_mono_cem_flux[t, x, p, j]
                    )
                    * (self.cell_width * self.shadow_factor)
                    * self.cell_triplet_num
            )

        # Add constraints for mass transfer terms (acid channel of the bipolar membrane)
        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term for the acid channel of the bipolar membrane channel",
        )
        def eq_mass_transfer_term_acidate(self, t, x, p, j):
            return (
                    self.acidate.mass_transfer_term[t, x, p, j]
                    == (
                            self.generation_cem_flux[t, x, p, j]
                            + self.elec_migration_bpem_flux[t, x, p, j]
                            + self.nonelec_bpem_flux[t, x, p, j]
                            + self.elec_migration_mono_aem_flux[t, x, p, j]
                            + self.nonelec_mono_aem_flux[t, x, p, j]
                    )
                    * (self.cell_width * self.shadow_factor)
                    * self.cell_triplet_num
            )
        #
        # @self.Constraint(
        #     self.flowsheet().time,
        #     doc="Evaluate Base produced",
        # )
        # def eq_product_basate(self, t):
        #     conc_unit = 1 * pyunits.mole * pyunits.second ** -1
        #     product_net_loc = 0 * pyunits.kg * pyunits.second ** -1
        #
        #     for j in self.config.property_package.cation_set:
        #         if not j == "H_+":
        #             product_in_loc = smooth_min(
        #                 self.basate.properties_in[t].flow_mol_phase_comp["Liq", j] / conc_unit * self.scal_val,
        #                 self.basate.properties_in[t].flow_mol_phase_comp[
        #                     "Liq", "OH_-"] / conc_unit * self.scal_val / self.config.property_package.charge_comp[j])
        #
        #             product_out_loc = smooth_min(
        #                 self.basate.properties_out[t].flow_mol_phase_comp["Liq", j] / conc_unit * self.scal_val,
        #                 self.basate.properties_out[t].flow_mol_phase_comp[
        #                     "Liq", "OH_-"] / conc_unit * self.scal_val / self.config.property_package.charge_comp[j])
        #
        #             product_net_loc += -1 * smooth_min(product_in_loc - product_out_loc,
        #                                                0) * conc_unit / self.scal_val * (
        #                                        self.config.property_package.charge_comp[j] *
        #                                        self.config.property_package.mw_comp["OH_-"] +
        #                                        self.config.property_package.mw_comp[j])
        #
        #     return self.base_produced == product_net_loc
        #
        # @self.Constraint(
        #     self.flowsheet().time,
        #     doc="Evaluate Acid produced",
        # )
        # def eq_product_acidate(self, t):
        #     conc_unit = 1 * pyunits.mole * pyunits.second ** -1
        #     product_net_loc = 0 * pyunits.kg * pyunits.second ** -1
        #
        #     for j in self.config.property_package.anion_set:
        #         if not j == "OH_-":
        #             product_in_loc = smooth_min(
        #                 self.acidate.properties_in[t].flow_mol_phase_comp["Liq", j] / conc_unit * self.scal_val,
        #                 self.acidate.properties_in[t].flow_mol_phase_comp[
        #                     "Liq", "H_+"] / conc_unit * self.scal_val / (-self.config.property_package.charge_comp[j]))
        #
        #             product_out_loc = smooth_min(
        #                 self.acidate.properties_out[t].flow_mol_phase_comp["Liq", j] / conc_unit * self.scal_val,
        #                 self.acidate.properties_out[t].flow_mol_phase_comp[
        #                     "Liq", "H_+"] / conc_unit * self.scal_val / (-self.config.property_package.charge_comp[j]))
        #
        #             product_net_loc += -1 * smooth_min(product_in_loc - product_out_loc,
        #                                                0) * conc_unit / self.scal_val * (
        #                                        (-self.config.property_package.charge_comp[j]) *
        #                                        self.config.property_package.mw_comp["H_+"] +
        #                                        self.config.property_package.mw_comp[j])
        #
        #     return self.acid_produced == product_net_loc
        # @self.Constraint(
        #     self.flowsheet().time,
        #     doc="Evaluate Base produced",
        # )
        # def eq_product_basate(self, t):
        #     conc_unit = 1 * pyunits.mole * pyunits.second ** -1
        #     product_net_loc = 0 * pyunits.kg * pyunits.second ** -1
        #
        #     for j in self.config.property_package.cation_set:
        #         if not j == "H_+":
        #             product_in_loc = smooth_min(
        #                 self.basate.properties[t, self.diluate.length_domain.first()].flow_mol_phase_comp["Liq", j] / conc_unit,
        #                 self.basate.properties[t, self.diluate.length_domain.first()].flow_mol_phase_comp[
        #                     "Liq", "OH_-"] / conc_unit / self.config.property_package.charge_comp[j])
        #
        #             product_out_loc = smooth_min(
        #                 self.basate.properties[t, self.diluate.length_domain.last()].flow_mol_phase_comp["Liq", j] / conc_unit,
        #                 self.basate.properties[t, self.diluate.length_domain.last()].flow_mol_phase_comp[
        #                     "Liq", "OH_-"] / conc_unit/ self.config.property_package.charge_comp[j])
        #
        #             product_net_loc += -1 * smooth_min(product_in_loc - product_out_loc,
        #                                                0) * conc_unit * (
        #                                        self.config.property_package.charge_comp[j] *
        #                                        self.config.property_package.mw_comp["OH_-"] +
        #                                        self.config.property_package.mw_comp[j])
        #
        #     return self.base_produced[t] == product_net_loc
        #
        # @self.Constraint(
        #     self.flowsheet().time,
        #     doc="Evaluate Acid produced",
        # )
        # def eq_product_acidate(self, t):
        #     conc_unit = 1 * pyunits.mole * pyunits.second ** -1
        #     product_net_loc = 0 * pyunits.kg * pyunits.second ** -1
        #
        #     for j in self.config.property_package.anion_set:
        #         if not j == "OH_-":
        #             product_in_loc = smooth_min(
        #                 self.acidate.properties[t, self.diluate.length_domain.first()].flow_mol_phase_comp["Liq", j] / conc_unit,
        #                 self.acidate.properties[t, self.diluate.length_domain.first()].flow_mol_phase_comp[
        #                     "Liq", "H_+"] / conc_unit /(-self.config.property_package.charge_comp[j]))
        #
        #             product_out_loc = smooth_min(
        #                 self.acidate.properties[t, self.diluate.length_domain.last()].flow_mol_phase_comp["Liq", j] / conc_unit,
        #                 self.acidate.properties[t, self.diluate.length_domain.last()].flow_mol_phase_comp[
        #                     "Liq", "H_+"] / conc_unit/ (-self.config.property_package.charge_comp[j]))
        #
        #             product_net_loc += -1 * smooth_min(product_in_loc - product_out_loc,
        #                                               0) * conc_unit * (
        #                                       (-self.config.property_package.charge_comp[j]) *
        #                                       self.config.property_package.mw_comp["H_+"] +
        #                                       self.config.property_package.mw_comp[j])
        #
        #     return self.acid_produced[t] == product_net_loc


        @self.Constraint(
            self.flowsheet().config.time,
            self.diluate.length_domain,
            doc="Electrical power consumption of a stack",
        )
        def eq_power_electrical(self, t, x):
            if x == self.diluate.length_domain.first():
                self.diluate.power_electrical_x[t, x].fix(0)
                return Constraint.Skip
            else:
                return (
                        self.diluate.Dpower_electrical_Dx[t, x]
                        == self.voltage_x[t, x]
                        * self.current_density_x[t, x]
                        * self.electrical_stage_num * self.cell_width * self.shadow_factor
                        * self.diluate.length
                )

        @self.Constraint(
            self.flowsheet().config.time,
            doc="Diluate_volume_flow_rate_specific electrical power consumption of a stack",
        )
        def eq_specific_power_electrical(self, t):
            return (
                    pyunits.convert(
                        self.specific_power_electrical[t],
                        pyunits.watt * pyunits.second * pyunits.meter ** -3,
                    )
                    * self.diluate.properties[
                        t, self.diluate.length_domain.last()
                    ].flow_vol_phase["Liq"]
                    == self.diluate.power_electrical_x[t, self.diluate.length_domain.last()]
            )

        # @self.Constraint(
        #     self.flowsheet().config.time,
        #     self.diluate.length_domain,
        #     self.config.property_package.phase_list,
        #     doc="Overall current efficiency evaluation",
        # )
        # def eq_current_efficiency_x(self, t, x, p):
        #
        #     return (
        #             self.current_efficiency_x[t, x]
        #             * self.current_density_x[t, x]
        #             * self.cell_width * self.shadow_factor
        #             * self.cell_triplet_num
        #             == -sum(
        #         self.diluate.mass_transfer_term[t, x, p, j]
        #         * self.config.property_package.charge_comp[j]
        #         for j in self.cation_set
        #     )
        #             * Constants.faraday_constant
        #     )

        # @self.Constraint(
        #     self.flowsheet().config.time,
        #     doc="Water recovery by mass",
        # )
        # def eq_recovery_mass_H2O(self, t):
        #     return (
        #             self.recovery_mass_H2O[t]
        #             * (
        #                     self.diluate.properties[
        #                         t, self.diluate.length_domain.first()
        #                     ].flow_mass_phase_comp["Liq", "H2O"]
        #                     + self.concentrate.properties[
        #                         t, self.diluate.length_domain.first()
        #                     ].flow_mass_phase_comp["Liq", "H2O"]
        #             )
        #             == self.diluate.properties[
        #                 t, self.diluate.length_domain.last()
        #             ].flow_mass_phase_comp["Liq", "H2O"]
        #     )

    def _make_performance_nonohm_mem(self):

        self.potential_nonohm_membrane_x = Var(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,  # to inspect
            bounds=(-50, 50),
            units=pyunits.volt,
            doc="Nonohmic potential across a membane",
        )


        @self.Constraint(
            self.membrane_set,
            self.electrode_side,
            self.flowsheet().time,
            self.diluate.length_domain,
            self.ion_set,
            doc="calcualte current density from the electrical input",
        )
        def eq_set_surface_conc(self, mem, side, t, x, j):
            if not self.config.has_Nernst_diffusion_layer:
                if mem == "cem" and side == "cathode_left":
                    return self.conc_mem_surf_mol_x[mem, side,  t, x, j] == self.basate.properties[t, x].conc_mol_phase_comp["Liq", j]
                else:
                    if mem == "aem" and side == "anode_right":
                        return self.conc_mem_surf_mol_x[mem, side, t, x, j] == self.acidate.properties[t, x].conc_mol_phase_comp["Liq", j]

                    else:
                        return self.conc_mem_surf_mol_x[mem, side, t, x, j] == self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
            else:
                return Constraint.Skip

        @self.Constraint(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total non-ohmic potential across an iem; this takes account of diffusion and Donnan Potentials",
        )
        def eq_potential_nonohm_membrane_x(self, mem, t, x):

            if (
                not self.config.has_Nernst_diffusion_layer
            ) and x == self.diluate.length_domain.first():
                self.potential_nonohm_membrane_x[mem, t, x].fix(0)
                return Constraint.Skip

            return self.potential_nonohm_membrane_x[mem, t, x] == (
                Constants.gas_constant
                * self.diluate.properties[t, x].temperature
                / Constants.faraday_constant
                * (
                    sum(
                        self.ion_trans_number_membrane[mem, j]
                        / self.config.property_package.charge_comp[j]
                        for j in self.cation_set
                    )
                    + sum(
                        self.ion_trans_number_membrane[mem, j]
                        / self.config.property_package.charge_comp[j]
                        for j in self.anion_set
                    )
                )
                * log(
                    sum(
                        self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                        for j in self.ion_set
                    )
                    / sum(
                        self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                        for j in self.ion_set
                    )
                )
            )

    def _make_performance_dl_polarization(self):

        self.current_dens_lim_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=500,
            bounds=(0, 10000),
            units=pyunits.amp * pyunits.meter ** -2,
            doc="Limiting Current Density accross the membrane as a function of the normalized length",
        )

        self.potential_nonohm_dl_x = Var(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,  # to inspect
            bounds=(-50, 50),
            units=pyunits.volt,
            doc="Nonohmic potential in two diffusion layers on the two sides of a membrane",
        )

        self.potential_ohm_dl_x = Var(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,  # to inspect
            bounds=(0, 50),
            units=pyunits.volt,
            doc="Ohmic potential in two diffusion layers on the two sides of a membrane",
        )

        self.dl_thickness_x = Var(
            self.membrane_set,
            self.electrode_side,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.0005,
            bounds=(0, 1e-2),
            units=pyunits.m,
            doc="Thickness of the diffusion layer",
        )

        if (
                self.config.limiting_current_density_method
                == LimitingCurrentDensityMethod.InitialValue
        ):

            @self.Constraint(
                self.flowsheet().time,
                self.diluate.length_domain,
                doc="Calculate length-indexed limiting current density",
            )
            def eq_current_dens_lim_x(self, t, x):
                return self.current_dens_lim_x[t, x] == (
                        self.config.limiting_current_density_data
                        * pyunits.amp
                        * pyunits.meter ** -2
                        / sum(
                    self.diluate.properties[t, 0].conc_mol_phase_comp["Liq", j]
                    for j in self.cation_set
                )
                        * sum(
                    self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                    for j in self.cation_set
                )
                )

        elif (
                self.config.limiting_current_density_method
                == LimitingCurrentDensityMethod.Empirical
        ):
            self.param_b = Param(
                initialize=0.5,
                units=pyunits.dimensionless,
                doc="emprical parameter b to calculate limitting current density",
            )
            self.param_a = Param(
                initialize=25,
                units=pyunits.coulomb
                      * pyunits.mol ** -1
                      * pyunits.meter ** (1 - self.param_b)
                      * pyunits.second ** (self.param_b - 1),
                doc="emprical parameter a to calculate limitting current density",
            )

            @self.Constraint(
                self.flowsheet().time,
                self.diluate.length_domain,
                doc="Calculate length-indexed limiting current density",
            )
            def eq_current_dens_lim_x(self, t, x):

                return self.current_dens_lim_x[
                    t, x
                ] == self.param_a * self.velocity_diluate[t, x] ** self.param_b * self.salt_conc_dilu_x[t,x]

        elif (
                self.config.limiting_current_density_method
                == LimitingCurrentDensityMethod.Theoretical
        ):
            self._get_fluid_dimensionless_quantities()

            @self.Constraint(
                self.flowsheet().time,
                self.diluate.length_domain,
                doc="Calculate length-indexed limiting current density",
            )
            def eq_current_dens_lim_x(self, t, x):
                return self.current_dens_lim_x[
                    t, x
                ] == self.N_Sh * self.diffus_mass * self.hydraulic_diameter ** -1 * Constants.faraday_constant * (
                        sum(
                            self.ion_trans_number_membrane["cem", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                    self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                    / self.config.property_package.charge_comp[j]
                    for j in self.cation_set
                )
                ) ** -1 * sum(
                    self.config.property_package.charge_comp[j]
                    * self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                    for j in self.cation_set
                )

        @self.Constraint(
            self.membrane_set,
            self.electrode_side,
            self.flowsheet().time,
            self.diluate.length_domain,
            self.ion_set,
            doc="Establish relationship between interfacial concentration polarization ratio and current density",
        )
        def eq_conc_polarization_ratio(self, mem, side, t, x, j):
            if mem == "cem" and side == "cathode_left":
                return self.conc_mem_surf_mol_x[mem, side, t, x, j] / (
                        self.basate.properties[t, x].conc_mol_phase_comp["Liq", j]
                ) == (
                    1
                    + self.current_density_x[t, x] / self.current_dens_lim_x[t, x]
                )
            else:
                if mem == "aem" and side == "anode_right":
                    return self.conc_mem_surf_mol_x[mem, side, t, x, j] / (
                                    self.acidate.properties[t, x].conc_mol_phase_comp["Liq", j]

                    ) == (
                            1
                            + self.current_density_x[t, x] / self.current_dens_lim_x[t, x]
                    )
                else:
                    return self.conc_mem_surf_mol_x[mem, side, t, x, j] / (
                        self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                    ) == (
                        1
                        - self.current_density_x[t, x] / self.current_dens_lim_x[t, x]
                    )

        @self.Constraint(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total non-ohmic potential across the two diffusion layers of an iem.",
        )
        def eq_potential_nonohm_dl(self, mem, t, x):
            if mem == "cem":
                return self.potential_nonohm_dl_x[mem, t, x] == (
                    Constants.gas_constant
                    * self.diluate.properties[t, x].temperature
                    / Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        + sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.anion_set
                        )
                    )
                    * log(
                        sum(
                            self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                            for j in self.ion_set
                        )
                        *
                            sum(
                                self.basate.properties[t, x].conc_mol_phase_comp[
                                    "Liq", j
                                ]
                                for j in self.ion_set
                            )
                        * sum(
                            self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                            for j in self.ion_set
                        )
                        ** -1
                        * (
                                sum(
                                    self.diluate.properties[t, x].conc_mol_phase_comp[
                                        "Liq", j
                                    ]
                                    for j in self.ion_set
                                )
                        )
                        ** -1
                    )
                )
            else:
                if mem == "aem":
                    return self.potential_nonohm_dl_x[mem, t, x] == (
                        Constants.gas_constant
                        * self.diluate.properties[t, x].temperature
                        / Constants.faraday_constant
                        * (
                            sum(
                                self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                                / self.config.property_package.charge_comp[j]
                                for j in self.cation_set
                            )
                            + sum(
                                self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                                / self.config.property_package.charge_comp[j]
                                for j in self.anion_set
                            )
                        )
                        * log(
                            sum(
                                self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                                for j in self.ion_set
                            )
                            *
                                sum(
                                    self.diluate.properties[t, x].conc_mol_phase_comp[
                                        "Liq", j
                                    ]
                                    for j in self.ion_set
                                )
                            * sum(
                                self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                                for j in self.ion_set
                            )
                            ** -1
                            * (
                                    sum(
                                        self.acidate.properties[t, x].conc_mol_phase_comp["Liq", j]
                                        for j in self.ion_set
                                    )
                            )
                            ** -1
                        )
                    )
                else:
                    return Constraint.Skip



        @self.Constraint(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total ohmic potential across the two diffusion layers of an iem.",
        )
        def eq_potential_ohm_dl_x(self, mem, t, x):
            if mem == "cem":
                return self.potential_ohm_dl_x[mem, t, x] == (
                    Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * self.diluate.properties[t, x].equiv_conductivity_phase["Liq"]
                    ** -1
                    * log(
                        sum(
                            self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                            for j in self.ion_set
                        )
                        ** -1
                        * (
                            sum(
                                self.basate.properties[t, x].conc_mol_phase_comp[
                                    "Liq", j
                                ]
                                for j in self.ion_set
                            )
                        )
                        ** -1
                        * sum(
                            self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                            for j in self.ion_set
                        )
                        *
                            sum(
                                self.diluate.properties[t, x].conc_mol_phase_comp[
                                    "Liq", j
                                ]
                                for j in self.ion_set
                            )
                    )
                )
            else:
                if mem == "aem":
                    return self.potential_ohm_dl_x[mem, t, x] == (
                        -Constants.faraday_constant
                        * (
                            sum(
                                self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                                ** -1
                                for j in self.ion_set
                            )
                            ** -1
                            * len(self.ion_set)
                        )
                        * (
                            sum(
                                self.ion_trans_number_membrane[mem, j]
                                / self.config.property_package.charge_comp[j]
                                for j in self.cation_set
                            )
                            - sum(
                                self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                                / self.config.property_package.charge_comp[j]
                                for j in self.cation_set
                            )
                        )
                        ** -1
                        * self.diluate.properties[t, x].equiv_conductivity_phase["Liq"]
                        ** -1
                        * log(
                            sum(
                                self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                                for j in self.ion_set
                            )
                            *
                                sum(
                                    self.diluate.properties[t, x].conc_mol_phase_comp[
                                        "Liq", j
                                    ]
                                    for j in self.ion_set
                                )
                            * sum(
                                self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                                for j in self.ion_set
                            )
                            ** -1
                            * (
                                sum(
                                    self.acidate.properties[t, x].conc_mol_phase_comp[
                                        "Liq", j
                                    ]
                                    for j in self.ion_set
                                )
                            )
                            ** -1
                        )
                    )
                else:
                    return Constraint.Skip

        @self.Constraint(
            self.membrane_set,
            self.electrode_side,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total non-ohmic potential across the two diffusion layers of an iem.",
        )
        def eq_dl_thickness(self, mem, side, t, x):
            if mem == "cem" and side == "cathode_left":
                return self.dl_thickness_x[mem, side, t, x] == (
                    Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    *
                        sum(
                            self.basate.properties[t, x].conc_mol_phase_comp[
                                "Liq", j
                            ]
                            for j in self.cation_set
                        )
                    * self.current_dens_lim_x[t, x] ** -1
                )
            elif mem == "cem" and side == "anode_right":
                return self.dl_thickness_x[mem, side, t, x] == (
                    Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * sum(
                            self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                            for j in self.cation_set
                        )
                    * self.current_dens_lim_x[t, x] ** -1
                )
            elif mem == "aem" and side == "cathode_left":
                return self.dl_thickness_x[mem, side, t, x] == (
                    -Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * sum(
                            self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                            for j in self.cation_set
                        )
                    * self.current_dens_lim_x[t, x] ** -1
                )
            elif mem == "aem" and side == "anode_right":
                return self.dl_thickness_x[mem, side, t, x] == (
                    -Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * sum(
                            self.acidate.properties[t, x].conc_mol_phase_comp[
                                "Liq", j
                            ]
                            for j in self.cation_set
                        )
                    * self.current_dens_lim_x[t, x] ** -1
                )
            else:
                return Constraint.Skip


    def _make_catalyst(self):

        self.flux_splitting = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1,
            domain=NonNegativeReals,
            # bounds=(0, 50000),
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Flux generated",
        )
        self.membrane_fixed_catalyst_aem = Var(
            initialize=5e3,
            bounds=(1e-1, 1e5),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )
        self.membrane_fixed_catalyst_cem = Var(
            initialize=5e3,
            bounds=(1e-1, 1e5),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )

        self.k_a = Var(
            initialize=1e-3,
            bounds=(1e-6, 1e5),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )
        self.k_b = Var(
            initialize=3e-2,
            bounds=(1e-2, 1e5),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )

        const = 0.0936 * pyunits.K**2 * pyunits.volt**-1 * pyunits.meter

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the non-dimensional potential drop across the depletion region",
        )
        def eq_voltage_membrane_drop_non_dim(self, t, x):
            return self.elec_field_non_dim[t, x] == const * self.basate.properties[
                t, x
            ].temperature ** -2 * self.relative_permittivity ** -1 * sqrt(
                (
                    Constants.faraday_constant
                    * self.membrane_fixed_charge
                    * self.voltage_membrane_drop[t, x]
                )
                / (
                    Constants.vacuum_electric_permittivity
                    * self.relative_permittivity
                )
            )

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the potential barrier at limiting current across the bipolar membrane",
        )
        def eq_flux_splitting(self, t, x):
            terms = 40
            matrx = 0
            for indx in range(terms):
                matrx += (
                    2**indx
                    * self.elec_field_non_dim[t,x] ** indx
                    / (math.factorial(indx) * math.factorial(indx + 1))
                )

            matrx *= self.k2_zero * self.conc_water
            matrx_a = (
                matrx * self.membrane_fixed_catalyst_cem/ self.k_a
            )
            matrx_b = (
                matrx * self.membrane_fixed_catalyst_aem/ self.k_b
            )
            return self.flux_splitting[t, x] == (matrx_a + matrx_b) * sqrt(
                self.voltage_membrane_drop[t, x]
                * Constants.vacuum_electric_permittivity
                * self.relative_permittivity
                / (Constants.faraday_constant * self.membrane_fixed_charge)
            )


    def _get_fluid_dimensionless_quantities(self):
        self.hydraulic_diameter = Var(
            initialize=1e-3,
            bounds=(0, None),
            units=pyunits.meter,
            doc="The hydraulic diameter of the channel",
        )
        self.N_Re = Var(
            initialize=50,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Reynolds Number",
        )
        self.N_Sc = Var(
            initialize=2000,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Schmidt Number",
        )
        self.N_Sh = Var(
            initialize=100,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Sherwood Number",
        )

        if self.config.hydraulic_diameter_method == HydraulicDiameterMethod.fixed:
            _log.warning("Do not forget to FIX the channel hydraulic diameter in [m]!")
        else:

            @self.Constraint(
                doc="To calculate hydraulic diameter",
            )
            def eq_hydraulic_diameter(self):
                if (
                    self.config.hydraulic_diameter_method
                    == HydraulicDiameterMethod.conventional
                ):
                    return (
                        self.hydraulic_diameter
                        == 2
                        * self.channel_height
                        * self.cell_width * self.shadow_factor
                        * self.spacer_porosity
                        * (self.channel_height + self.cell_width * self.shadow_factor) ** -1
                    )
                else:
                    self.spacer_specific_area = Var(
                        initialize=1e4,
                        bounds=(0, None),
                        units=pyunits.meter**-1,
                        doc="The specific area of the channel",
                    )
                    return (
                        self.hydraulic_diameter
                        == 4
                        * self.spacer_porosity
                        * (
                            2 * self.channel_height**-1
                            + (1 - self.spacer_porosity) * self.spacer_specific_area
                        )
                        ** -1
                    )

        @self.Constraint(
            doc="To calculate Re",
        )
        def eq_Re(self):

            return (
                self.N_Re
                == self.dens_mass
                * self.velocity_diluate[0, 0]
                * self.hydraulic_diameter
                * self.visc_d**-1
            )

        @self.Constraint(
            doc="To calculate Sc",
        )
        def eq_Sc(self):

            return self.N_Sc == self.visc_d * self.dens_mass**-1 * self.diffus_mass**-1

        @self.Constraint(
            doc="To calculate Sh",
        )
        def eq_Sh(self):

            return self.N_Sh == 0.29 * self.N_Re**0.5 * self.N_Sc**0.33

    def _pressure_drop_calculation(self):
        self.pressure_drop = Var(
            self.flowsheet().time,
            initialize=1e4,
            units=pyunits.pascal * pyunits.meter ** -1,
            doc="pressure drop per unit of length",
        )
        self.pressure_drop_total = Var(
            self.flowsheet().time,
            initialize=1e6,
            units=pyunits.pascal,
            doc="pressure drop over an entire ED stack",
        )

        if self.config.pressure_drop_method == PressureDropMethod.experimental:
            _log.warning(
                "Do not forget to FIX the experimental pressure drop value in [Pa/m]!"
            )
        else:  # PressureDropMethod.Darcy_Weisbach is used
            if not (
                    self.config.has_Nernst_diffusion_layer
                    and self.config.limiting_current_density_method
                    == LimitingCurrentDensityMethod.Theoretical
            ):
                self._get_fluid_dimensionless_quantities()

            self.friction_factor = Var(
                initialize=10,
                bounds=(0, None),
                units=pyunits.dimensionless,
                doc="friction factor of the channel fluid",
            )

            @self.Constraint(
                self.flowsheet().time,
                doc="To calculate pressure drop per unit length",
            )
            def eq_pressure_drop(self, t):
                return (
                        self.pressure_drop[t]
                        == self.dens_mass
                        * self.friction_factor
                        * self.velocity_diluate[0, 0] ** 2
                        * 0.5
                        * self.hydraulic_diameter ** -1
                )

            if self.config.friction_factor_method == FrictionFactorMethod.fixed:
                _log.warning("Do not forget to FIX the Darcy's friction factor value!")
            else:

                @self.Constraint(
                    doc="To calculate friction factor",
                )
                def eq_friction_factor(self):
                    if (
                            self.config.friction_factor_method
                            == FrictionFactorMethod.Gurreri
                    ):
                        return (
                                self.friction_factor
                                == 4 * 50.6 * self.spacer_porosity ** -7.06 * self.N_Re ** -1
                        )
                    elif (
                            self.config.friction_factor_method
                            == FrictionFactorMethod.Kuroda
                    ):
                        return (
                                self.friction_factor
                                == 4 * 9.6 * self.spacer_porosity ** -1 * self.N_Re ** -0.5
                        )

        @self.Constraint(
            self.flowsheet().time,
            doc="To calculate total pressure drop over a stack",
        )
        def eq_pressure_drop_total(self, t):
            return (
                    self.pressure_drop_total[t] == self.pressure_drop[t] * self.cell_length
            )

    # initialize method
    def initialize_build(
        self,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
        fail_on_warning=False,
        ignore_dof=False,
    ):
        """
        General wrapper for electrodialysis_1D initialization routines

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = None)
            fail_on_warning : boolean argument to fail or only produce  warning upon unsuccessful solve (default=False)
            ignore_dof : boolean argument to ignore when DOF != 0 (default=False)

        Returns: None
        """
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")
        # Set solver options
        opt = get_solver(solver, optarg)

        # Set the intial conditions over the 1D length from the state vars -dilate
        for k in self.keys():
            for set in self[k].diluate.properties:
                if ("flow_mol_phase_comp" or "flow_mass_phase_comp") not in self[
                    k
                ].diluate.properties[set].define_state_vars():
                    raise ConfigurationError(
                        "Electrodialysis1D unit model requires "
                        "either a 'flow_mol_phase_comp' or 'flow_mass_phase_comp' "
                        "state variable basis to apply the 'propogate_initial_state' method"
                    )
                if "temperature" in self[k].diluate.properties[set].define_state_vars():
                    self[k].diluate.properties[set].temperature = value(
                        self[k].diluate.properties[(0.0, 0.0)].temperature
                    )
                if "pressure" in self[k].diluate.properties[set].define_state_vars():
                    self[k].diluate.properties[set].pressure = value(
                        self[k].diluate.properties[(0.0, 0.0)].pressure
                    )
                if (
                        "flow_mol_phase_comp"
                        in self[k].diluate.properties[set].define_state_vars()
                ):
                    for ind in self[k].diluate.properties[set].flow_mol_phase_comp:
                        self[k].diluate.properties[set].flow_mol_phase_comp[ind] = (
                            value(
                                self[k]
                                .diluate.properties[(0.0, 0.0)]
                                .flow_mol_phase_comp[ind]
                            )
                        )
                if (
                        "flow_mass_phase_comp"
                        in self[k].diluate.properties[set].define_state_vars()
                ):
                    for ind in self[k].diluate.properties[set].flow_mass_phase_comp:
                        self[k].diluate.properties[set].flow_mass_phase_comp[ind] = (
                            value(
                                self[k]
                                .diluate.properties[(0.0, 0.0)]
                                .flow_mass_phase_comp[ind]
                            )
                        )
                if hasattr(self[k], "conc_mem_surf_mol_x"):
                    for mem in self[k].membrane_set:
                        for side in self[k].electrode_side:
                            for j in self[k].ion_set:
                                self[k].conc_mem_surf_mol_x[
                                    mem, side, set, j
                                ].set_value(
                                    self[k]
                                    .diluate.properties[set]
                                    .conc_mol_phase_comp["Liq", j]
                                )
                self[k].total_areal_resistance_x[set].set_value(
                    (
                            pyunits.ohm * pyunits.meter ** 2 * ((0.108 * pyunits.kg * pyunits.meter ** -3 / (
                            self.acidate.properties[set].conc_mass_phase_comp["Liq", "H_+"] +
                            self.acidate.properties[set].conc_mass_phase_comp["Liq", "Cl_-"] +
                            self.basate.properties[set].conc_mass_phase_comp["Liq", "Na_+"] +
                            self.basate.properties[set].conc_mass_phase_comp["Liq", "OH_-"]) + 0.0492) / 5)
                            + self[k].channel_height
                            * (
                                    self[k].acidate.properties[set].elec_cond_phase["Liq"]
                                    ** -1
                                    +self[k].basate.properties[set].elec_cond_phase["Liq"]
                                    ** -1
                                    + self[k].diluate.properties[set].elec_cond_phase["Liq"]
                                    ** -1
                            )
                    )
                    * self[k].cell_triplet_num
                    + self[k].electrodes_resistance
                )

        # Set the intial conditions over the 1D length from the state vars - basate
        for k in self.keys():
            for set in self[k].basate.properties:
                if ("flow_mol_phase_comp" or "flow_mass_phase_comp") not in self[
                    k
                ].basate.properties[set].define_state_vars():
                    raise ConfigurationError(
                        "Electrodialysis1D unit model requires "
                        "either a 'flow_mol_phase_comp' or 'flow_mass_phase_comp' "
                        "state variable basis to apply the 'propogate_initial_state' method"
                    )
                if "temperature" in self[k].basate.properties[set].define_state_vars():
                    self[k].basate.properties[set].temperature = value(
                        self[k].basate.properties[(0.0, 0.0)].temperature
                    )
                if "pressure" in self[k].basate.properties[set].define_state_vars():
                    self[k].basate.properties[set].pressure = value(
                        self[k].basate.properties[(0.0, 0.0)].pressure
                    )
                if (
                        "flow_mol_phase_comp"
                        in self[k].basate.properties[set].define_state_vars()
                ):
                    for ind in self[k].basate.properties[set].flow_mol_phase_comp:
                        self[k].basate.properties[set].flow_mol_phase_comp[ind] = (
                            value(
                                self[k]
                                .basate.properties[(0.0, 0.0)]
                                .flow_mol_phase_comp[ind]
                            )
                        )
                if (
                        "flow_mass_phase_comp"
                        in self[k].basate.properties[set].define_state_vars()
                ):
                    for ind in self[k].basate.properties[set].flow_mass_phase_comp:
                        self[k].basate.properties[set].flow_mass_phase_comp[ind] = (
                            value(
                                self[k]
                                .basate.properties[(0.0, 0.0)]
                                .flow_mass_phase_comp[ind]
                            )
                        )
                if hasattr(self[k], "conc_mem_surf_mol_x"):
                    for mem in self[k].membrane_set:
                        for side in self[k].electrode_side:
                            for j in self[k].ion_set:
                                self[k].conc_mem_surf_mol_x[
                                    mem, side, set, j
                                ].set_value(
                                    self[k]
                                    .basate.properties[set]
                                    .conc_mol_phase_comp["Liq", j]
                                )

                # Set the intial conditions over the 1D length from the state vars - acidate
                for k in self.keys():
                    for set in self[k].acidate.properties:
                        if ("flow_mol_phase_comp" or "flow_mass_phase_comp") not in self[
                            k
                        ].acidate.properties[set].define_state_vars():
                            raise ConfigurationError(
                                "Electrodialysis1D unit model requires "
                                "either a 'flow_mol_phase_comp' or 'flow_mass_phase_comp' "
                                "state variable basis to apply the 'propogate_initial_state' method"
                            )
                        if "temperature" in self[k].acidate.properties[set].define_state_vars():
                            self[k].acidate.properties[set].temperature = value(
                                self[k].acidate.properties[(0.0, 0.0)].temperature
                            )
                        if "pressure" in self[k].acidate.properties[set].define_state_vars():
                            self[k].acidate.properties[set].pressure = value(
                                self[k].acidate.properties[(0.0, 0.0)].pressure
                            )
                        if (
                                "flow_mol_phase_comp"
                                in self[k].acidate.properties[set].define_state_vars()
                        ):
                            for ind in self[k].acidate.properties[set].flow_mol_phase_comp:
                                self[k].acidate.properties[set].flow_mol_phase_comp[ind] = (
                                    value(
                                        self[k]
                                        .acidate.properties[(0.0, 0.0)]
                                        .flow_mol_phase_comp[ind]
                                    )
                                )
                        if (
                                "flow_mass_phase_comp"
                                in self[k].acidate.properties[set].define_state_vars()
                        ):
                            for ind in self[k].acidate.properties[set].flow_mass_phase_comp:
                                self[k].acidate.properties[set].flow_mass_phase_comp[ind] = (
                                    value(
                                        self[k]
                                        .acidate.properties[(0.0, 0.0)]
                                        .flow_mass_phase_comp[ind]
                                    )
                                )
                        if hasattr(self[k], "conc_mem_surf_mol_x"):
                            for mem in self[k].membrane_set:
                                for side in self[k].electrode_side:
                                    for j in self[k].ion_set:
                                        self[k].conc_mem_surf_mol_x[
                                            mem, side, set, j
                                        ].set_value(
                                            self[k]
                                            .acidate.properties[set]
                                            .conc_mol_phase_comp["Liq", j]
                                        )


        # ---------------------------------------------------------------------

        # Initialize diluate block
        flags_diluate = self.diluate.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        init_log.info_high("Initialization Step 1 Complete.")
        # ---------------------------------------------------------------------
        if not ignore_dof:
            check_dof(self, fail_flag=fail_on_warning, logger=init_log)
        # ---------------------------------------------------------------------
        # Initialize concentrate_basate_side block
        flags_basate = self.basate.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        init_log.info_high("Initialization Step 2 Complete.")
        # ---------------------------------------------------------------------
        # Initialize concentrate_acidate_side block
        flags_acidate = self.acidate.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,  # inlet var
            hold_state=True,
        )
        init_log.info_high("Initialization Step 3 Complete.")
        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 4 {}.".format(idaeslog.condition(res)))
        check_solve(
            res,
            logger=init_log,
            fail_flag=fail_on_warning,
            checkpoint="Initialization Step 4",
        )
        # ---------------------------------------------------------------------
        # Release state
        self.diluate.release_state(flags_diluate, outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))
        self.basate.release_state(flags_basate, outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))
        self.acidate.release_state(flags_acidate, outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(f"Unit model {self.name} failed to initialize")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        # Scaling factors that user may setup
        # The users are highly encouraged to provide scaling factors for assessable vars below.
        # Not providing these vars will give a warning.
        if (
            iscale.get_scaling_factor(self.solute_diffusivity_membrane, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.solute_diffusivity_membrane, 1e10)
        if iscale.get_scaling_factor(self.membrane_thickness, warning=True) is None:
            iscale.set_scaling_factor(self.membrane_thickness, 1e4)
        if (
            iscale.get_scaling_factor(self.water_permeability_membrane, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.water_permeability_membrane, 1e14)
        if iscale.get_scaling_factor(self.cell_triplet_num, warning=True) is None:
            iscale.set_scaling_factor(self.cell_triplet_num, 0.1)
        if iscale.get_scaling_factor(self.electrical_stage_num, warning=True) is None:
            iscale.set_scaling_factor(self.electrical_stage_num, 1)
        if iscale.get_scaling_factor(self.cell_length, warning=True) is None:
            iscale.set_scaling_factor(self.cell_length, 1e1)
        if iscale.get_scaling_factor(self.cell_width, warning=True) is None:
            iscale.set_scaling_factor(self.cell_width, 1e2)
        if iscale.get_scaling_factor(self.shadow_factor, warning=True) is None:
            iscale.set_scaling_factor(self.shadow_factor, 1e0)
        if iscale.get_scaling_factor(self.channel_height, warning=True) is None:
            iscale.set_scaling_factor(self.channel_height, 1e5)
        if iscale.get_scaling_factor(self.spacer_porosity, warning=True) is None:
            iscale.set_scaling_factor(self.spacer_porosity, 1)
        if (
            iscale.get_scaling_factor(self.membrane_areal_resistance_combined, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.membrane_areal_resistance_combined, 1e5)
        if iscale.get_scaling_factor(self.electrodes_resistance, warning=True) is None:
            iscale.set_scaling_factor(self.electrodes_resistance, 1e1)
        if hasattr(self, "voltage_applied") and (
                iscale.get_scaling_factor(self.voltage_applied, warning=True) is None
        ):
            iscale.set_scaling_factor(self.voltage_applied, 1)
        if hasattr(self, "current_applied") and (
                iscale.get_scaling_factor(self.current_applied, warning=True) is None
        ):
            iscale.set_scaling_factor(self.current_applied, 1)
        if hasattr(self, "conc_water") and (
                iscale.get_scaling_factor(self.conc_water, warning=True) is None
        ):
            iscale.set_scaling_factor(self.conc_water, 1e-4)

        if hasattr(self, "voltage_applied") and (
            iscale.get_scaling_factor(self.voltage_applied, warning=True) is None
        ):
            iscale.set_scaling_factor(self.voltage_applied, 1)
        if hasattr(self, "current_applied") and (
            iscale.get_scaling_factor(self.current_applied, warning=True) is None
        ):
            iscale.set_scaling_factor(self.current_applied, 1)

        # if hasattr(self, "acid_produced") and (
        #         iscale.get_scaling_factor(self.acid_produced, warning=True) is None
        # ):
        #     sf = 1
        #     for j in self.config.property_package.anion_set:
        #         sf = smooth_min(iscale.get_scaling_factor(self.acidate.properties[0,0].flow_mass_phase_comp["Liq", j]),
        #                                     iscale.get_scaling_factor(self.acidate.properties[0,0].flow_mass_phase_comp[
        #                                         "Liq", "H_+"]) )
        #
        #     iscale.set_scaling_factor(self.acid_produced, sf)
        #
        # if hasattr(self, "base_produced") and (
        #         iscale.get_scaling_factor(self.base_produced, warning=True) is None
        # ):
        #     sf = 1
        #     for j in self.config.property_package.cation_set:
        #         sf = smooth_min(iscale.get_scaling_factor(self.basate.properties[0,0].flow_mass_phase_comp["Liq", j]),
        #                         iscale.get_scaling_factor(self.basate.properties[0,0].flow_mass_phase_comp[
        #                                                       "Liq", "OH_-"]))
        #
        #     iscale.set_scaling_factor(self.base_produced, sf)

        if self.config.has_catalyst == True:
            if (
                    iscale.get_scaling_factor(
                        self.membrane_fixed_catalyst_cem, warning=True
                    )
                    is None
            ):
                iscale.set_scaling_factor(self.membrane_fixed_catalyst_cem, 1e-3)
            # if self.config.has_catalyst == True:
            if (
                    iscale.get_scaling_factor(
                        self.membrane_fixed_catalyst_aem, warning=True
                    )
                    is None
            ):
                iscale.set_scaling_factor(self.membrane_fixed_catalyst_aem, 1e-3)

            if iscale.get_scaling_factor(self.k_a, warning=True) is None:
                iscale.set_scaling_factor(self.k_a, 1e6)

            if iscale.get_scaling_factor(self.k_b, warning=True) is None:
                iscale.set_scaling_factor(self.k_b, 1e2)

        if (
                self.config.has_catalyst == True
                or self.config.limiting_potential_method_bpem
                == LimitingpotentialMethod.Empirical
        ) and iscale.get_scaling_factor(self.elec_field_non_dim, warning=True) is None:
            iscale.set_scaling_factor(self.elec_field_non_dim, 1e-1)

        if (
                self.config.limiting_potential_method_bpem
                == LimitingpotentialMethod.Empirical
                and iscale.get_scaling_factor(self.voltage_membrane_drop, warning=True) is None
        ):
            iscale.set_scaling_factor(self.voltage_membrane_drop, 1)

        if hasattr(self, "membrane_fixed_charge") and (
                iscale.get_scaling_factor(self.membrane_fixed_charge, warning=True) is None
        ):
            iscale.set_scaling_factor(self.membrane_fixed_charge, 1e-3)
        if hasattr(self, "diffus_mass") and (
                iscale.get_scaling_factor(self.diffus_mass, warning=True) is None
        ):
            iscale.set_scaling_factor(self.diffus_mass, 1e9)
        if hasattr(self, "salt_conc_aem_x") and (
                iscale.get_scaling_factor(self.salt_conc_aem_x, warning=True) is None
        ):
            if self.config.salt_calculation:
                sf = - smooth_min(
                    -iscale.get_scaling_factor(self.basate.properties[0, 0].conc_mol_phase_comp["Liq", "Na_+"]),
                    -iscale.get_scaling_factor(self.basate.properties[0, 0].conc_mol_phase_comp["Liq", "Cl_-"]))
            else:
                sf = value(self.salt_conc_aem_ref) ** -1
            iscale.set_scaling_factor(self.salt_conc_aem_x, sf)
        if hasattr(self, "salt_conc_cem_x") and (
                iscale.get_scaling_factor(self.salt_conc_cem_x, warning=True) is None
        ):
            if self.config.salt_calculation:
                sf = - smooth_min(
                    -iscale.get_scaling_factor(self.acidate.properties[0, 0].conc_mol_phase_comp["Liq", "Na_+"]),
                    -iscale.get_scaling_factor(self.acidate.properties[0, 0].conc_mol_phase_comp["Liq", "Cl_-"]))
                # sf = iscale.get_scaling_factor(self.acidate.properties[0,0].conc_mol_phase_comp["Liq", "Na_+"])
            else:
                sf = value(self.salt_conc_cem_ref) ** -1
            iscale.set_scaling_factor(self.salt_conc_cem_x, sf)
        if hasattr(self, "salt_conc_dilu_x") and (
                iscale.get_scaling_factor(self.salt_conc_dilu_x, warning=True) is None
        ):
            if self.config.salt_calculation:
                sf = - smooth_min(
                    -iscale.get_scaling_factor(self.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Na_+"]),
                    -iscale.get_scaling_factor(self.diluate.properties[0, 0].conc_mol_phase_comp["Liq", "Cl_-"]))
                # sf = iscale.get_scaling_factor(self.acidate.properties[0,0].conc_mol_phase_comp["Liq", "Na_+"])
            else:
                sf = value(self.salt_conc_dilu_ref) ** -1
            iscale.set_scaling_factor(self.salt_conc_dilu_x, sf)

        if (
                self.config.has_catalyst == True
                or self.config.limiting_potential_method_bpem
                == LimitingpotentialMethod.Empirical
        ):

            if hasattr(self, "relative_permittivity") and (
                    iscale.get_scaling_factor(self.relative_permittivity, warning=True)
                    is None
            ):
                iscale.set_scaling_factor(self.relative_permittivity, 1e-1)

        if iscale.get_scaling_factor(self.kr, warning=True) is None:
            iscale.set_scaling_factor(self.kr, 1e-11)
        if (
                hasattr(self, "k2_zero")
                and iscale.get_scaling_factor(self.k2_zero, warning=True) is None
        ):
            iscale.set_scaling_factor(self.k2_zero, 1e5)

        # The folloing Vars are built for constructing constraints and their sf are computed from other Vars.

        if (
            self.config.has_catalyst == True
            and iscale.get_scaling_factor(self.voltage_membrane_drop, warning=True)
            is None
        ):
            sf = (
                (
                    iscale.get_scaling_factor(self.elec_field_non_dim)
                    * iscale.get_scaling_factor(self.relative_permittivity)
                    * 293**-2
                    / 0.09636**-1
                )
                ** 2
                * value(Constants.vacuum_electric_permittivity) ** -1
                * iscale.get_scaling_factor(self.relative_permittivity)
                * value(Constants.faraday_constant) ** -1
                * iscale.get_scaling_factor(self.membrane_fixed_charge)
            )

            iscale.set_scaling_factor(self.voltage_membrane_drop, 1e0)

        if self.config.has_catalyst == True:
            if iscale.get_scaling_factor(self.flux_splitting, warning=True) is None:

                terms = 40
                sf = 0
                for indx in range(terms):
                    sf += (
                        2**indx
                        * iscale.get_scaling_factor(self.elec_field_non_dim) ** -indx
                        / (math.factorial(indx) * math.factorial(indx + 1))
                    )

                sf **= -1
                sf *= iscale.get_scaling_factor(
                    self.k2_zero
                ) * iscale.get_scaling_factor(self.conc_water)
                sf_a = (
                    sf
                    * iscale.get_scaling_factor(self.membrane_fixed_catalyst_cem)
                    / iscale.get_scaling_factor(self.k_a)
                )
                sf_b = (
                    sf
                    * iscale.get_scaling_factor(self.membrane_fixed_catalyst_aem)
                    / iscale.get_scaling_factor(self.k_b)
                )

                sf = (sf_a**-1 + sf_b**-1) ** -1 * sqrt(
                    iscale.get_scaling_factor(self.voltage_membrane_drop)
                    * value(Constants.vacuum_electric_permittivity) ** -1
                    * iscale.get_scaling_factor(self.relative_permittivity)
                    / (
                        value(Constants.faraday_constant) ** -1
                        * iscale.get_scaling_factor(self.membrane_fixed_charge)
                    )
                )

                iscale.set_scaling_factor(self.flux_splitting, sf)

        for ind in self.total_areal_resistance_x:
            if (
                iscale.get_scaling_factor(
                    self.total_areal_resistance_x[ind], warning=False
                )
                is None
            ):
                sf = (
                    iscale.get_scaling_factor(self.membrane_areal_resistance_combined) ** -1
                    + iscale.get_scaling_factor(self.channel_height)  ** -1
                    * (
                        iscale.get_scaling_factor(
                            self.diluate.properties[ind].elec_cond_phase["Liq"]
                        )
                        + iscale.get_scaling_factor(
                            self.acidate.properties[ind].elec_cond_phase["Liq"]
                        )
                        + iscale.get_scaling_factor(
                        self.basate.properties[ind].elec_cond_phase["Liq"]
                    )
                    )
                ) ** -1 * iscale.get_scaling_factor(self.cell_triplet_num)
                iscale.set_scaling_factor(self.total_areal_resistance_x[ind], sf)

        for ind in self.current_density_x:
            if (
                iscale.get_scaling_factor(self.current_density_x[ind], warning=False)
                is None
            ):
                if (
                    self.config.operation_mode
                    == ElectricalOperationMode.Constant_Current
                ):
                    sf = (
                        iscale.get_scaling_factor(self.current_applied)
                        / iscale.get_scaling_factor(self.cell_width)
                        / iscale.get_scaling_factor(self.shadow_factor)
                        / iscale.get_scaling_factor(self.cell_length)
                    )
                    iscale.set_scaling_factor(self.current_density_x[ind], sf)
                else:
                    sf = iscale.get_scaling_factor(
                        self.voltage_applied
                    ) / iscale.get_scaling_factor(self.total_areal_resistance_x[ind])
                    iscale.set_scaling_factor(self.current_density_x[ind], sf)

        for ind in self.elec_migration_mono_cem_flux:
            iscale.set_scaling_factor(
                self.elec_migration_mono_cem_flux[ind],
                iscale.get_scaling_factor(self.current_density_x[ind[0],ind[1]])
                * 1e5,
            )
        for ind in self.elec_migration_mono_aem_flux:
            iscale.set_scaling_factor(
                self.elec_migration_mono_aem_flux[ind],
                iscale.get_scaling_factor(self.current_density_x[ind[0],ind[1]])
                * 1e5,
            )
        for ind in self.elec_migration_bpem_flux:
            iscale.set_scaling_factor(
                self.elec_migration_bpem_flux[ind],
                iscale.get_scaling_factor(self.current_density_x[ind[0],ind[1]])
                * 1e5,
            )

        for ind in self.generation_cem_flux:
            if ind[3] == "H_+" or "H2O":
                if self.config.has_catalyst == True:
                    sf = 0.5 * iscale.get_scaling_factor(self.flux_splitting)
                else:
                    sf = iscale.get_scaling_factor(self.elec_migration_bpem_flux[ind])
            else:
                sf = 1

            iscale.set_scaling_factor(self.generation_cem_flux[ind], sf)



        for ind in self.generation_aem_flux:
            if ind[3] == "OH_-" or "H2O":
                if self.config.has_catalyst == True:
                    sf = iscale.get_scaling_factor(self.flux_splitting)
                else:
                    sf = iscale.get_scaling_factor(self.elec_migration_bpem_flux[ind])
            else:
                sf = 1

            iscale.set_scaling_factor(self.generation_aem_flux[ind], sf)


        for ind in self.nonelec_mono_cem_flux:
            if ind[3] == "H2O":
                sf = (
                    1e-3
                    * 0.018
                    * iscale.get_scaling_factor(self.water_permeability_membrane)
                    * iscale.get_scaling_factor(
                        self.basate.properties[ind[0],ind[1]].pressure_osm_phase[
                            ind[2]
                        ]
                    )
                )
            else:
                sf = (
                    iscale.get_scaling_factor(self.solute_diffusivity_membrane)
                    / iscale.get_scaling_factor(self.membrane_thickness)
                    * iscale.get_scaling_factor(
                        self.basate.properties[ind[0],ind[1]].conc_mol_phase_comp[
                            ind[2], ind[3]
                        ]
                    )
                )
            iscale.set_scaling_factor(self.nonelec_mono_cem_flux[ind], sf)


        for ind in self.nonelec_mono_aem_flux:
            if ind[3] == "H2O":
                sf = (
                        1e-3
                        * 0.018
                        * iscale.get_scaling_factor(self.water_permeability_membrane)
                        * iscale.get_scaling_factor(
                    self.acidate.properties[ind[0],ind[1]].pressure_osm_phase[
                        ind[2]
                    ]
                )
                )
            else:
                sf = (
                        iscale.get_scaling_factor(self.solute_diffusivity_membrane)
                        / iscale.get_scaling_factor(self.membrane_thickness)
                        * iscale.get_scaling_factor(
                    self.acidate.properties[ind[0],ind[1]].conc_mol_phase_comp[
                        ind[2], ind[3]
                    ]
                )
                )
            iscale.set_scaling_factor(self.nonelec_mono_aem_flux[ind], sf)



        for ind in self.nonelec_bpem_flux:
            if ind[3] == "H2O":
                sf = (
                    1e-3
                    * 0.018
                    * iscale.get_scaling_factor(self.water_permeability_membrane)
                    * iscale.get_scaling_factor(
                        self.acidate.properties[ind[0],ind[1]].pressure_osm_phase[ind[2]]
                    )
                )
            else:
                sf = 1
            iscale.set_scaling_factor(self.nonelec_bpem_flux[ind], sf)

        for ind in self.acidate.mass_transfer_term:
            if ind[3] == "H_+":
                sf = iscale.get_scaling_factor(self.generation_cem_flux[ind])
            else:
                if ind[3] == "H2O":
                    sf = iscale.get_scaling_factor(self.nonelec_bpem_flux[ind])
                else:
                    sf = iscale.get_scaling_factor(self.elec_migration_bpem_flux[ind])

            sf *= (
                iscale.get_scaling_factor(self.cell_width)
                * iscale.get_scaling_factor(self.shadow_factor)
                * iscale.get_scaling_factor(self.cell_length)
                * iscale.get_scaling_factor(self.cell_triplet_num)
            )
            iscale.set_scaling_factor(self.acidate.mass_transfer_term[ind], sf)
        #
        for ind in self.basate.mass_transfer_term:
            if ind[3] == "OH_-":
                sf = iscale.get_scaling_factor(self.generation_aem_flux[ind])
            else:
                if ind[3] == "H2O":
                    sf = iscale.get_scaling_factor(self.nonelec_bpem_flux[ind])
                else:
                    sf = iscale.get_scaling_factor(self.elec_migration_bpem_flux[ind])

            sf *= (
                iscale.get_scaling_factor(self.cell_width)
                * iscale.get_scaling_factor(self.shadow_factor)
                * iscale.get_scaling_factor(self.cell_length)
                * iscale.get_scaling_factor(self.cell_triplet_num)
            )
            iscale.set_scaling_factor(self.basate.mass_transfer_term[ind], sf)

        for ind in self.velocity_diluate:
            if (
                iscale.get_scaling_factor(self.velocity_diluate[ind], warning=False)
                is None
            ):
                sf = (
                    iscale.get_scaling_factor(
                        self.diluate.properties[ind].flow_vol_phase["Liq"]
                    )
                    * iscale.get_scaling_factor(self.cell_width) ** -1
                    * iscale.get_scaling_factor(self.shadow_factor) ** -1
                    * iscale.get_scaling_factor(self.channel_height) ** -1
                    * iscale.get_scaling_factor(self.spacer_porosity) ** -1
                    * iscale.get_scaling_factor(self.cell_triplet_num) ** -1
                )

                iscale.set_scaling_factor(self.velocity_diluate[ind], sf)

            for ind in self.velocity_basate:
                if (
                        iscale.get_scaling_factor(self.velocity_basate[ind], warning=False)
                        is None
                ):
                    sf = (
                            iscale.get_scaling_factor(
                                self.basate.properties[ind].flow_vol_phase["Liq"]
                            )
                            * iscale.get_scaling_factor(self.cell_width) ** -1
                            * iscale.get_scaling_factor(self.shadow_factor) ** -1
                            * iscale.get_scaling_factor(self.channel_height) ** -1
                            * iscale.get_scaling_factor(self.spacer_porosity) ** -1
                            * iscale.get_scaling_factor(self.cell_triplet_num) ** -1
                    )

                    iscale.set_scaling_factor(self.velocity_basate[ind], sf)

            for ind in self.velocity_acidate:
                if (
                        iscale.get_scaling_factor(self.velocity_diluate[ind], warning=False)
                        is None
                ):
                    sf = (
                            iscale.get_scaling_factor(
                                self.acidate.properties[ind].flow_vol_phase["Liq"]
                            )
                            * iscale.get_scaling_factor(self.cell_width) ** -1
                            * iscale.get_scaling_factor(self.shadow_factor) ** -1
                            * iscale.get_scaling_factor(self.channel_height) ** -1
                            * iscale.get_scaling_factor(self.spacer_porosity) ** -1
                            * iscale.get_scaling_factor(self.cell_triplet_num) ** -1
                    )

                    iscale.set_scaling_factor(self.velocity_acidate[ind], sf)





        for ind in self.voltage_x:
            if iscale.get_scaling_factor(self.voltage_x[ind], warning=False) is None:
                sf = iscale.get_scaling_factor(
                    self.current_density_x[ind]
                ) * iscale.get_scaling_factor(self.total_areal_resistance_x[ind])
                iscale.set_scaling_factor(self.voltage_x[ind], sf)

        if iscale.get_scaling_factor(self.spacer_porosity, warning=False) is None:
            iscale.set_scaling_factor(self.spacer_porosity, 1)


        for ind in self.diluate.power_electrical_x:
            if (
                    iscale.get_scaling_factor(
                        self.diluate.power_electrical_x[ind], warning=False
                    )
                    is None
            ):
                iscale.set_scaling_factor(
                    self.diluate.power_electrical_x[ind],
                    iscale.get_scaling_factor(self.voltage_x[ind])
                    * iscale.get_scaling_factor(self.current_density_x[ind])
                    * iscale.get_scaling_factor(self.electrical_stage_num)
                    * iscale.get_scaling_factor(self.cell_width)
                    * iscale.get_scaling_factor(self.shadow_factor)
                    * iscale.get_scaling_factor(self.cell_length),
                )
        for ind in self.diluate.Dpower_electrical_Dx:
            if (
                    iscale.get_scaling_factor(
                        self.diluate.Dpower_electrical_Dx[ind], warning=False
                    )
                    is None
            ):
                iscale.set_scaling_factor(
                    self.diluate.Dpower_electrical_Dx[ind],
                    iscale.get_scaling_factor(self.diluate.power_electrical_x[ind]),
                )
        if (
                iscale.get_scaling_factor(self.specific_power_electrical, warning=False)
                is None
        ):
            iscale.set_scaling_factor(
                self.specific_power_electrical,
                3.6e6
                * iscale.get_scaling_factor(
                    self.diluate.power_electrical_x[
                        0, self.diluate.length_domain.last()
                    ]
                )
                * (
                        iscale.get_scaling_factor(
                            self.diluate.properties[
                                0, self.diluate.length_domain.last()
                            ].flow_vol_phase["Liq"]
                        )
                        * iscale.get_scaling_factor(self.cell_triplet_num)
                )
                ** -1,
            )

        if hasattr(self, "conc_mem_surf_mol_x"):
            for ind in self.conc_mem_surf_mol_x:
                if iscale.get_scaling_factor(self.conc_mem_surf_mol_x[ind]) is None:
                    if (ind[0] == "cem" and ind[1] == "cathode_left") or (
                        ind[0] == "aem" and ind[1] == "anode_right"
                    ):
                        iscale.set_scaling_factor(
                            self.conc_mem_surf_mol_x[ind],
                            iscale.get_scaling_factor(
                                self.acidate.properties[
                                    ind[2], ind[3]
                                ].conc_mol_phase_comp["Liq", ind[4]]
                            ),
                        )
                    else:
                        iscale.set_scaling_factor(
                            self.conc_mem_surf_mol_x[ind],
                            iscale.get_scaling_factor(
                                self.diluate.properties[ind[2], ind[3]].conc_mol_phase_comp[
                                    "Liq", ind[4]
                                ]
                            ),
                        )

        if hasattr(self, "potential_nonohm_membrane_x"):
            if iscale.get_scaling_factor(self.potential_nonohm_membrane_x) is None:
                for ind in self.potential_nonohm_membrane_x:
                    sf = (
                            value(Constants.faraday_constant)
                            * value(Constants.gas_constant) ** -1
                            * 298.15 ** -1
                    )
                    iscale.set_scaling_factor(self.potential_nonohm_membrane_x[ind], sf)
        if hasattr(self, "potential_nonohm_dl_x"):
            if iscale.get_scaling_factor(self.potential_nonohm_dl_x) is None:
                for ind in self.potential_nonohm_dl_x:
                    sf = (
                            value(Constants.faraday_constant)
                            * value(Constants.gas_constant) ** -1
                            * 298.15 ** -1
                    )
                    iscale.set_scaling_factor(self.potential_nonohm_dl_x[ind], sf)
        if hasattr(self, "potential_ohm_dl_x"):
            if iscale.get_scaling_factor(self.potential_ohm_dl_x) is None:
                for ind in self.potential_nonohm_dl_x:
                    sf = (
                            96485 ** -1
                            * sum(
                        iscale.get_scaling_factor(
                            self.diluate.properties[0, 0].diffus_phase_comp[
                                "Liq", j
                            ]
                        )
                        ** -2
                        for j in self.ion_set
                    )
                            ** -0.5
                            * float(len(self.ion_set)) ** -1
                    )
                    iscale.set_scaling_factor(self.potential_ohm_dl_x[ind], sf)
        if hasattr(self, "dl_thickness_x"):
            if iscale.get_scaling_factor(self.dl_thickness_x) is None:
                for ind in self.dl_thickness_x:
                    sf = (
                            96485 ** -1
                            * sum(
                        iscale.get_scaling_factor(
                            self.diluate.properties[0, 0].diffus_phase_comp[
                                "Liq", j
                            ]
                        )
                        ** -2
                        for j in self.ion_set
                    )
                            ** -0.5
                            * len(self.ion_set) ** -1
                            * sum(
                        iscale.get_scaling_factor(self.conc_mem_surf_mol_x[ind, j])
                        ** 2
                        for j in self.cation_set
                    )
                            ** 0.5
                            * iscale.get_scaling_factor(
                        self.current_density_x[ind[2], ind[3]]
                    )
                            ** -1
                    )
                    iscale.set_scaling_factor(self.dl_thickness_x[ind], sf)
        if hasattr(self, "spacer_specific_area") and (
            iscale.get_scaling_factor(self.spacer_specific_area, warning=True) is None
        ):
            iscale.set_scaling_factor(self.spacer_specific_area, 1e-4)
        if hasattr(self, "hydraulic_diameter") and (
            iscale.get_scaling_factor(self.hydraulic_diameter, warning=True) is None
        ):
            iscale.set_scaling_factor(self.hydraulic_diameter, 1e4)
        if hasattr(self, "dens_mass") and (
            iscale.get_scaling_factor(self.dens_mass, warning=True) is None
        ):
            iscale.set_scaling_factor(self.dens_mass, 1e-3)
        if hasattr(self, "N_Re") and (
            iscale.get_scaling_factor(self.N_Re, warning=True) is None
        ):
            sf = (
                iscale.get_scaling_factor(self.dens_mass)
                * iscale.get_scaling_factor(self.velocity_diluate)
                * iscale.get_scaling_factor(self.hydraulic_diameter)
                * iscale.get_scaling_factor(self.visc_d) ** -1
            )
            iscale.set_scaling_factor(self.N_Re, sf)
        if hasattr(self, "N_Sc") and (
            iscale.get_scaling_factor(self.N_Sc, warning=True) is None
        ):
            sf = (
                iscale.get_scaling_factor(self.visc_d)
                * iscale.get_scaling_factor(self.dens_mass) ** -1
                * iscale.get_scaling_factor(self.diffus_mass) ** -1
            )
            iscale.set_scaling_factor(self.N_Sc, sf)
        if hasattr(self, "N_Sh") and (
            iscale.get_scaling_factor(self.N_Sh, warning=True) is None
        ):
            sf = (
                10
                * iscale.get_scaling_factor(self.N_Re) ** 0.5
                * iscale.get_scaling_factor(self.N_Sc) ** 0.33
            )
            iscale.set_scaling_factor(self.N_Sh, sf)
        if hasattr(self, "friction_factor") and (
            iscale.get_scaling_factor(self.friction_factor, warning=True) is None
        ):
            if self.config.friction_factor_method == FrictionFactorMethod.fixed:
                sf = 0.1
            elif self.config.friction_factor_method == FrictionFactorMethod.Gurreri:
                sf = (
                    (4 * 50.6) ** -1
                    * (iscale.get_scaling_factor(self.spacer_porosity)) ** -7.06
                    * iscale.get_scaling_factor(self.N_Re) ** -1
                )
            elif self.config.friction_factor_method == FrictionFactorMethod.Kuroda:
                sf = (4 * 9.6) ** -1 * iscale.get_scaling_factor(self.N_Re) ** -0.5
            iscale.set_scaling_factor(self.friction_factor, sf)

        if hasattr(self, "pressure_drop") and (
            iscale.get_scaling_factor(self.pressure_drop, warning=True) is None
        ):
            if self.config.pressure_drop_method == PressureDropMethod.experimental:
                sf = 1e-5
            else:
                sf = (
                    iscale.get_scaling_factor(self.dens_mass)
                    * iscale.get_scaling_factor(self.friction_factor)
                    * iscale.get_scaling_factor(self.velocity_diluate[0]) ** 2
                    * 2
                    * iscale.get_scaling_factor(self.hydraulic_diameter) ** -1
                )
            iscale.set_scaling_factor(self.pressure_drop, sf)

        if hasattr(self, "pressure_drop_total") and (
            iscale.get_scaling_factor(self.pressure_drop_total, warning=True) is None
        ):
            if self.config.pressure_drop_method == PressureDropMethod.experimental:
                sf = 1e-5 * iscale.get_scaling_factor(self.cell_length)
            else:
                sf = (
                    iscale.get_scaling_factor(self.dens_mass)
                    * iscale.get_scaling_factor(self.friction_factor)
                    * iscale.get_scaling_factor(self.velocity_diluate[0]) ** 2
                    * 2
                    * iscale.get_scaling_factor(self.hydraulic_diameter) ** -1
                    * iscale.get_scaling_factor(self.cell_length)
                )
            iscale.set_scaling_factor(self.pressure_drop_total, sf)

        if hasattr(self, "current_dens_lim_bpem"):
            if iscale.get_scaling_factor(self.current_dens_lim_bpem) is None:
                if (
                    self.config.limiting_current_density_method_bpem
                    == LimitingCurrentDensitybpemMethod.InitialValue
                ):
                    sf = self.config.limiting_current_density_data**-1
                    iscale.set_scaling_factor(self.current_dens_lim_bpem, sf)
                elif (
                    self.config.limiting_current_density_method_bpem
                    == LimitingCurrentDensitybpemMethod.Empirical
                ):
                    sf = (
                        iscale.get_scaling_factor(self.diffus_mass)
                        * value(Constants.faraday_constant) ** -1
                        * (2 * iscale.get_scaling_factor(self.salt_conc_aem_x)) ** 2
                        / (
                            iscale.get_scaling_factor(self.membrane_thickness)
                            * iscale.get_scaling_factor(self.membrane_fixed_charge)
                        )
                    )

                    iscale.set_scaling_factor(self.current_dens_lim_bpem, sf)


        if hasattr(self, "current_dens_lim_x"):
            if iscale.get_scaling_factor(self.current_dens_lim_x) is None:
                if (
                        self.config.limiting_current_density_method
                        == LimitingCurrentDensityMethod.InitialValue
                ):
                    for ind in self.current_dens_lim_x:
                        sf = (
                                self.config.limiting_current_density_data ** -1
                                * sum(
                            iscale.get_scaling_factor(
                                self.diluate.properties[
                                    ind[0], 0
                                ].conc_mol_phase_comp["Liq", j]
                            )
                            ** 2
                            for j in self.cation_set
                        )
                                ** -0.5
                                * sum(
                            iscale.get_scaling_factor(
                                self.diluate.properties[ind].conc_mol_phase_comp[
                                    "Liq", j
                                ]
                            )
                            ** 2
                            for j in self.cation_set
                        )
                                ** 0.5
                        )
                        iscale.set_scaling_factor(self.current_dens_lim_x[ind], sf)
                elif (
                        self.config.limiting_current_density_method
                        == LimitingCurrentDensityMethod.Empirical
                ):
                    for ind in self.current_dens_lim_x:
                        sf = 25 ** -1 * iscale.get_scaling_factor(self.velocity_diluate) ** 0.25 * iscale.get_scaling_factor(self.salt_conc_dilu_x)
                        iscale.set_scaling_factor(self.current_dens_lim_x[ind], sf)
                elif (
                        self.config.limiting_current_density_method
                        == LimitingCurrentDensityMethod.Theoretical
                ):
                    for ind in self.current_dens_lim_x:
                        sf = (
                                iscale.get_scaling_factor(self.N_Sh)
                                * iscale.get_scaling_factor(self.diffus_mass)
                                * iscale.get_scaling_factor(self.hydraulic_diameter) ** -1
                                * 96485 ** -1
                                * sum(
                            iscale.get_scaling_factor(
                                self.diluate.properties[ind].conc_mol_phase_comp[
                                    "Liq", j
                                ]
                            )
                            ** 2
                            for j in self.cation_set
                        )
                                ** 0.5
                        )
                        iscale.set_scaling_factor(self.current_dens_lim_x[ind], sf)
        # Constraint scaling
        # for ind, c in self.eq_current_voltage_relation.items():
        #     iscale.constraint_scaling_transform(
        #         c, iscale.get_scaling_factor(self.membrane_areal_resistance_combined)
        #     )

        for ind, c in self.eq_get_current_density.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.current_density_x[ind])
            )

        for ind, c in self.eq_power_electrical.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.diluate.power_electrical_x[ind])
            )
        for ind, c in self.eq_specific_power_electrical.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.specific_power_electrical[ind])
            )

        for ind, c in self.eq_elec_migration_mono_cem.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.elec_migration_mono_cem_flux[ind])
            )
        for ind, c in self.eq_elec_migration_mono_aem_flux.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.elec_migration_mono_aem_flux[ind])
            )
        for ind, c in self.eq_elec_migration_bpem_flux.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.elec_migration_bpem_flux[ind])
            )

        for ind, c in self.eq_nonelec_mono_cem_flux.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.nonelec_mono_cem_flux[ind])
            )

        for ind, c in self.eq_nonelec_mono_aem_flux.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.nonelec_mono_aem_flux[ind])
            )

        for ind, c in self.eq_nonelec_bpem_flux.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.nonelec_bpem_flux[ind])
            )

        for ind, c in self.eq_mass_transfer_term_diluate.items():
            iscale.constraint_scaling_transform(
                c,
                min(
                    iscale.get_scaling_factor(self.elec_migration_mono_cem_flux[ind]),
                    iscale.get_scaling_factor(
                        self.nonelec_mono_cem_flux[ind]
                    ),
                ),
            )

        for ind, c in self.eq_mass_transfer_term_basate.items():
            iscale.constraint_scaling_transform(
                c,
                min(
                    iscale.get_scaling_factor(self.generation_aem_flux[ind]),
                    iscale.get_scaling_factor(self.elec_migration_bpem_flux[ind]),
                    iscale.get_scaling_factor(
                        self.nonelec_bpem_flux[ind]
                    ),
                )
                * iscale.get_scaling_factor(self.cell_width)
                * iscale.get_scaling_factor(self.shadow_factor)
                * iscale.get_scaling_factor(self.cell_length)
                * iscale.get_scaling_factor(self.cell_triplet_num),
            )
        for ind, c in self.eq_mass_transfer_term_acidate.items():
            iscale.constraint_scaling_transform(
                c,
                min(
                    iscale.get_scaling_factor(self.generation_cem_flux[ind]),
                    iscale.get_scaling_factor(
                        self.nonelec_bpem_flux[ind]
                    ),
                )
                * iscale.get_scaling_factor(self.cell_width)
                * iscale.get_scaling_factor(self.shadow_factor)
                * iscale.get_scaling_factor(self.cell_length)
                * iscale.get_scaling_factor(self.cell_triplet_num),
            )

        if hasattr(self, "eq_flux_splitting"):
            for ind, c in self.eq_flux_splitting.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.flux_splitting[ind]),
                )

        # if hasattr(self, "eq_product_basate"):
        #     for ind, c in self.eq_product_basate.items():
        #         iscale.constraint_scaling_transform(
        #             c,
        #             iscale.get_scaling_factor(self.base_produced[ind]),
        #         )
        #
        # if hasattr(self, "eq_product_acidate"):
        #     for ind, c in self.eq_product_acidate.items():
        #         iscale.constraint_scaling_transform(
        #             c,
        #             iscale.get_scaling_factor(self.acid_produced[ind]),
        #         )

        if hasattr(self, "eq_salt_cem"):
            for ind, c in self.eq_salt_cem.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.salt_conc_cem_x[ind]),
                )

        if hasattr(self, "eq_salt_aem"):
            for ind, c in self.eq_salt_aem.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.salt_conc_aem_x[ind]),
                )
        if hasattr(self, "eq_potential_nonohm_membrane_x"):
            for ind, c in self.eq_potential_nonohm_membrane_x.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.potential_nonohm_membrane_x[ind]),
                )
        if hasattr(self, "eq_current_dens_lim_x"):
            for ind, c in self.eq_current_dens_lim_x.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.current_dens_lim_x[ind])
                )
        if hasattr(self, "eq_potential_nonohm_dl"):
            for ind, c in self.eq_potential_nonohm_dl.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.potential_nonohm_dl_x[ind])
                )
        if hasattr(self, "eq_potential_ohm_dl_x"):
            for ind, c in self.eq_potential_ohm_dl_x.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.potential_ohm_dl_x[ind])
                )
        if hasattr(self, "eq_dl_thickness_x"):
            for ind, c in self.eq_dl_thickness_x.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.dl_thickness_x[ind])
                )

        # for ind, c in self.eq_recovery_mass_H2O.items():
        #     iscale.constraint_scaling_transform(
        #         c,
        #         iscale.get_scaling_factor(
        #             self.diluate.properties[ind].flow_mass_phase_comp["Liq", "H2O"]
        #         ),
        #     )

        # for ind, c in self.eq_power_electrical.items():
        #     iscale.constraint_scaling_transform(
        #         c,
        #         iscale.get_scaling_factor(self.power_electrical[ind]),
        #     )

        # for ind, c in self.eq_specific_power_electrical.items():
        #     iscale.constraint_scaling_transform(
        #         c,
        #         iscale.get_scaling_factor(self.specific_power_electrical[ind])
        #         * iscale.get_scaling_factor(
        #             self.diluate.properties_out[ind].flow_vol_phase["Liq"]
        #         ),
        #     )
        # for ind, c in self.eq_current_efficiency.items():
        #     iscale.constraint_scaling_transform(
        #         c, iscale.get_scaling_factor(self.current[ind])
        #     )

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "Diluate Channel Inlet": self.inlet_diluate,
                "base channel of the bipolar membrane Channel Inlet": self.inlet_basate,
                "acid channel of the bipolar membrane Channel Inlet": self.inlet_acidate,
                "Diluate Channel Outlet": self.outlet_diluate,
                "base channel of the bipolar membrane Channel Outlet": self.outlet_basate,
                "acid channel of the bipolar membrane Channel Outlet": self.outlet_acidate,
            },
            time_point=time_point,
        )

    def _get_performance_contents(self, time_point=0):
        return {
            "vars": {
                "Total electrical power consumption(Watt)": self.power_electrical[
                    time_point
                ],
                "Specific electrical power consumption (kW*h/m**3)": self.specific_power_electrical[
                    time_point
                ],
                # "Current efficiency for deionzation": self.current_efficiency[
                #     time_point
                # ],
                # "Water recovery by mass": self.recovery_mass_H2O[time_point],
            },
            "exprs": {},
            "params": {},
        }

    def get_power_electrical(self, time_point=0):
        return self.diluate.power_electrical_x[
            time_point, self.diluate.length_domain.last()
        ]
    @property
    def default_costing_method(self):
        return cost_bipolar_electrodialysis
