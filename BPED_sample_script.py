import math
from idaes.core.util import DiagnosticsToolbox
import idaes.core.util.scaling as iscale
import idaes.core.util.model_statistics as istat
import idaes.logger as idaeslog

from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Expression,
    Objective,
    RangeSet,
    TransformationFactory,
    units as pyunits,
    value,
    assert_optimal_termination,
)
from idaes.core import (
    declare_process_block_class,
    FlowsheetBlock,
    ProcessBlockData,
    UnitModelCostingBlock,
)
from idaes.core.util.initialization import propagate_state
from idaes.core.util.math import smooth_min
from idaes.core.util.model_serializer import to_json
from idaes.models.unit_models import (
    Feed,
    Product,
    Separator,
    Mixer,
)
from watertap.core.solvers import get_solver
from watertap.core.util.initialization import assert_degrees_of_freedom
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
)

from parameter_sweep import (parameter_sweep,
                             LinearSample,
                             )

from watertap.unit_models.pressure_changer import Pump
from Biploar_and_Electrodialysis_1D_nmsu import (
    Bipolar_and_Electrodialysis1D,
    ElectricalOperationMode,
    LimitingCurrentDensitybpemMethod,
    PressureDropMethod,
    FrictionFactorMethod,
    HydraulicDiameterMethod,
)
from watertap.costing import WaterTAPCosting


# set up solver
solver = get_solver()
# set up logger
_log = idaeslog.getLogger(__name__)


# dummy block glass to set up flowsheet
@declare_process_block_class("IndexedBlock")
class IndexedBlockData(ProcessBlockData):
    CONFIG = ProcessBlockData.CONFIG()

    def build(self):
        super(IndexedBlockData, self).build()



def build_flowsheet_optimization(m):


    for n in m.fs.nstages:
        # degrees of freedom
        m.fs.stage[n].bpmed.cell_width.unfix()
        m.fs.stage[n].bpmed.cell_length.unfix()
        m.fs.stage[n].bpmed.cell_triplet_num.fix(163)

        m.fs.stage[n].bpmed.k_a.fix(1e1)
        #
        m.fs.stage[n].bpmed.cell_width.setlb(9e-2)
        m.fs.stage[n].bpmed.cell_length.setlb(1e-1)
        #
        m.fs.stage[n].bpmed.cell_width.setub(2e-1)
        m.fs.stage[n].bpmed.cell_length.setub(210)

        m.fs.stage[n].bpmed.cell_triplet_num.setub(900)
        m.fs.stage[n].bpmed.voltage_applied.setub(500)

        iscale.set_scaling_factor(m.fs.stage[n].bpmed.N_Re, 1e-0)
        iscale.set_scaling_factor(m.fs.stage[n].bpmed.N_Sh, 1)
        iscale.set_scaling_factor(m.fs.stage[n].bpmed.N_Sc, 1e-0)
        iscale.set_scaling_factor(m.fs.stage[n].bpmed.channel_height, 1e6)
        iscale.set_scaling_factor(m.fs.stage[n].bpmed.cell_length, 1e-0)
        m.fs.stage[n].bpmed.channel_height.fix(5e-4)

        def eq_channel_velocity_diluate_lim(blk):
            return blk.velocity_diluate[0,0] >= 0.06 * pyunits.meter * pyunits.second ** -1
        m.fs.stage[n].bpmed.eq_channel_velocity_diluate_lim = Constraint(rule=eq_channel_velocity_diluate_lim)


        m.fs.stage[n].bpmed.voltage_applied[0].unfix()

        # minimize the applied pressure at each stage
        m.fs.stage[n].pump_dilu.control_volume.properties_out[0].pressure.unfix()
        m.fs.stage[n].pump_acid.control_volume.properties_out[0].pressure.unfix()
        m.fs.stage[n].pump_base.control_volume.properties_out[0].pressure.unfix()

        #
        def eq_atm_outlet_pressure(blk):
            return blk.properties[0,1].pressure >= 101325 * pyunits.Pa

        #
        m.fs.stage[n].bpmed.diluate.eq_outlet_pressure = Constraint(rule=eq_atm_outlet_pressure)
        m.fs.stage[n].bpmed.acidate.eq_outlet_pressure = Constraint(rule=eq_atm_outlet_pressure)
        m.fs.stage[n].bpmed.basate.eq_outlet_pressure = Constraint(rule=eq_atm_outlet_pressure)


    m.fs.exit_base_conc = Var(
        initialize=100,
        bounds=(50, 5000),
        units=pyunits.mole * pyunits.meter ** -3,
    )
    iscale.set_scaling_factor(m.fs.exit_base_conc, 1e-2)

    m.fs.exit_base_conc.fix(650)

    m.fs.eq_Base_exit_conc = Constraint(
        expr=m.fs.Base_exit_conc
             == m.fs.exit_base_conc
    )

    m.fs.conc_acid_mol = Var(
        initialize=100,
        bounds=(1, 5000),
        units=pyunits.mole * pyunits.meter ** -3,
    )
    iscale.set_scaling_factor(m.fs.conc_acid_mol, 1e-1)
    m.fs.conc_acid_mol.fix(100)
    m.fs.feed_acid.properties[0].flow_mol_phase_comp['Liq', 'H_+'].unfix()
    m.fs.feed_acid.properties[0].flow_mol_phase_comp['Liq', 'Cl_-'].unfix()
    m.fs.eq_acid_H_conc = Constraint(
        expr=m.fs.feed_acid.properties[0].flow_mol_phase_comp['Liq', 'H_+'] == m.fs.conc_acid_mol * 0.0004381 * pyunits.meter**3* pyunits.second**-1)
    m.fs.eq_acid_cl_conc = Constraint(
        expr=m.fs.feed_acid.properties[0].flow_mol_phase_comp['Liq', 'Cl_-'] == m.fs.conc_acid_mol * 0.0004381 * pyunits.meter**3* pyunits.second**-1)
    #
    for ind, c in m.fs.eq_acid_H_conc.items():
        iscale.constraint_scaling_transform(
            c,
            iscale.get_scaling_factor(m.fs.conc_acid_mol)*1/0.0004381)
    #
    for ind, c in m.fs.eq_acid_cl_conc.items():
        iscale.constraint_scaling_transform(
            c,
            iscale.get_scaling_factor(m.fs.conc_acid_mol)*1/0.0004381)


    m.fs.obj = Objective(expr=m.fs.costing.LCOB)

    iscale.calculate_scaling_factors(m)


# membrane transport properties



def get_variables():
    design_variables = {
        "feed_flow_vol": 0.0004381,  # m3/s # =0.01 MGD
        "expt_salt_in_conc_g_l": 70,  # g/l
        "expt_acid_in_conc_M": 0.1 * 1e3,  # mol/m3
        "expt_base_in_conc_M": 0.1 * 1e3,  # mol/m3
        "channel_height": 0.00038,  # m
        "spacer_porosity": 0.8972,  # dimensionless
        "shadow_factor": 1,  # dimensionless
        "membrane_thickness_aem": 570e-6,  # m
        "membrane_thickness_cem": 570e-6,  # m
        "membrane_thickness_bpem": 2 * 570e-6,  # m
        "cell_triplet_num": 50,  # dimensionless
        "electrical_stage_num": 1,  # dimensionless
        "cell_length": 0.5,  # m
        "cell_width": 0.1,  # m
    }
    model_parameters = {

        "electrodes_resistance": 0.01,  # assumed
        "ion_trans_number_aem": 0.97,  # dimensionless
        "ion_trans_number_cem": 0.96,  # dimensionless
        "ion_trans_number_bpem": 1,  # dimensionless

        "ion_diffus_na_cem": 2.00E-10,  # m2/s
        "ion_diffus_na_aem": 7.50E-11,  # m2/s
        "ion_diffus_cl_cem": 1.50E-10,  # m2/s
        "ion_diffus_cl_aem": 1.90E-10,  # m2/s

        "velocity_channel": 0.08,  # m/s

        "water_trans_number_cem": 5.8,  # dimensionless
        "water_trans_number_aem": 4.3,  # dimensionless
        "water_trans_number_bpem": (5.8 + 4.3) / 2,  # dimensionless
        "water_perm_bpem": (2.16e-14 + 1.75e-14) / 2,  # m2/s
        "water_perm_cem": 2.16e-14,  # m2/s
        "water_perm_aem": 1.75e-14,  # m2/s

        "current_utilization": 1,  # dimensionless
        "current": 12,  # Ampere
        "voltage": 150,  # Volts
        "current_density": 600,  # A/m2
        "fixed_charge": 5e3,  # mol/m3
        "conc_water": 50 * 1e3,  # mol/m3
        "kr": 1.3 * 10 ** 10,  # L/Mol/s
        "k2_zero": 2 * 10 ** -6,  # /s
        "relative_permittivity": 30,  # dimensionless

        "catalyst_cem": 5e3,  # mol/m3
        "catalyst_aem": 5e3,  # mol/m3
        "k_a": 1e1,  # mol/m3
        "k_b": 5e4,  # mol/m3
        "diffus_mass": 1.6e-9,  # assumed
    }
    costing_parameters = {
        "total_investment_factor": 1,  # dimensionless
        "maintenance_labor_chemical_factor": 0.03,  # yr-1
        "utilization_factor": 0.9,  # dimensionless
        "electricity_cost": 0.07,  # USD_2018/kWh
        "electrical_carbon_intensity": 0.475,  # kg/kWh
        "plant_lifetime": 30,  # yr
        "wacc": 0.093073,  # dimensionless
        "TIC": 2,  # dimensionless
        "membrane_capital_cost": 1 * 548.7766544117648,  # USD_2018/m2  from 2022 paper. Coverted down to 2018. Average over AEM/CEM and BPEM = 742.5 USD 2022.
        "factor_membrane_replacement": 0.3,  # yr-1   from 2022 paper.
        "stack_electrode_capital_cost": 2100,  # USD_2018/m2
        "factor_stack_electrode_replacement": 0.2,  # yr-1
        "pump_cost": 889,  # USD_2018*s/l
        "pump_efficiency": 0.8,  # dimensionless
    }

    return design_variables, model_parameters, costing_parameters


def build_modular_bpmed(nstages):
    design_variables, model_parameters, costing_parameters = get_variables()

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    ion_dict = {
        "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
        "mw_data": {
            "H2O": 18e-3,
            "Na_+": 23e-3,
            "Cl_-": 35.5e-3,
            "H_+": 1e-3,
            "OH_-": 17.0e-3,
        },
        "elec_mobility_data": {
            ("Liq", "Na_+"): 5.19e-8,
            ("Liq", "Cl_-"): 7.92e-8,
            ("Liq", "H_+"): 36.23e-8,
            ("Liq", "OH_-"): 20.64e-8,
        },
        "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},
        "diffusivity_data": {
            ("Liq", "Na_+"): 1.33e-9,
            ("Liq", "Cl_-"): 2.03e-9,
            ("Liq", "H_+"): 9.31e-9,
            ("Liq", "OH_-"): 5.27e-9,
        },
    }
    m.fs.properties = MCASParameterBlock(**ion_dict)
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyunits.USD_2021

    # build unit model blocks
    m.fs.feed_dilu = Feed(property_package=m.fs.properties)
    m.fs.feed_acid = Feed(property_package=m.fs.properties)
    m.fs.feed_base = Feed(property_package=m.fs.properties)
    m.fs.dilu_out = Product(property_package=m.fs.properties)
    m.fs.acid_out = Product(property_package=m.fs.properties)
    m.fs.base_out = Product(property_package=m.fs.properties)

    # touch on-demand properties
    for block in [
        m.fs.feed_dilu.properties[0],
        m.fs.feed_acid.properties[0],
        m.fs.feed_base.properties[0],
        m.fs.dilu_out.properties[0],
        m.fs.acid_out.properties[0],
        m.fs.base_out.properties[0]
    ]:
        block.mass_frac_phase_comp
        block.conc_mass_phase_comp
        block.conc_mol_phase_comp
        block.flow_vol_phase


    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e0, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e1, index=("Liq", "Na_+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e1, index=("Liq", "Cl_-")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e1, index=("Liq", "H_+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e1, index=("Liq", "OH_-")
    )

    m.fs.feed_dilu.properties[0],
    m.fs.feed_acid.properties[0],
    m.fs.feed_base.properties[0],


    expt_salt_temperature = 298
    expt_acid_temperature = 298
    expt_base_temperature = 298

    expt_salt_pressure = 101325
    expt_acid_pressure = 101325
    expt_base_pressure = 101325

    m.fs.feed_dilu.properties[0].pressure.fix(expt_salt_pressure)
    m.fs.feed_dilu.properties[0].temperature.fix(expt_salt_temperature)

    m.fs.feed_acid.properties[0].pressure.fix(expt_acid_pressure)
    m.fs.feed_acid.properties[0].temperature.fix(expt_acid_temperature)

    m.fs.feed_base.properties[0].pressure.fix(expt_base_pressure)
    m.fs.feed_base.properties[0].temperature.fix(expt_base_temperature)

    init_arg_diluate = {
        # ("pressure", None): expt_salt_pressure,
        # ("temperature", None): expt_salt_temperature,
        ("flow_vol_phase", "Liq"): design_variables["feed_flow_vol"],
        ("conc_mass_phase_comp", ("Liq", "Na_+")): design_variables["expt_salt_in_conc_g_l"] * value(
            m.fs.feed_dilu.config.property_package.mw_comp["Na_+"] / (
                        m.fs.feed_dilu.config.property_package.mw_comp["Na_+"] +
                        m.fs.feed_dilu.config.property_package.mw_comp["Cl_-"])),
        ("conc_mass_phase_comp", ("Liq", "Cl_-")): design_variables["expt_salt_in_conc_g_l"] * value(
            m.fs.feed_dilu.config.property_package.mw_comp["Cl_-"] / (
                        m.fs.feed_dilu.config.property_package.mw_comp["Na_+"] +
                        m.fs.feed_dilu.config.property_package.mw_comp["Cl_-"])),
        ("conc_mass_phase_comp", ("Liq", "H_+")): 0,
        ("conc_mol_phase_comp", ("Liq", "OH_-")): 0,
        # ("conc_mass_phase_comp", ("Liq", "H2O")): 1e3,
    }
    init_arg_acidate = {
        # ("pressure", None): expt_acid_pressure,
        # ("temperature", None): expt_acid_temperature,
        ("flow_vol_phase", "Liq"): design_variables["feed_flow_vol"],
        ("conc_mol_phase_comp", ("Liq", "Na_+")): 0,
        ("conc_mol_phase_comp", ("Liq", "Cl_-")): design_variables["expt_acid_in_conc_M"],
        ("conc_mol_phase_comp", ("Liq", "H_+")): design_variables["expt_acid_in_conc_M"],
        ("conc_mol_phase_comp", ("Liq", "OH_-")): 0,
        # ("conc_mass_phase_comp", ("Liq", "H2O")): 1e3,
    }
    init_arg_basate = {
        # ("pressure", None): expt_base_pressure,
        # ("temperature", None): expt_base_temperature,
        ("flow_vol_phase", "Liq"): design_variables["feed_flow_vol"],
        ("conc_mol_phase_comp", ("Liq", "Na_+")): design_variables["expt_base_in_conc_M"],
        ("conc_mol_phase_comp", ("Liq", "Cl_-")): 0,
        ("conc_mol_phase_comp", ("Liq", "H_+")): 0,
        ("conc_mol_phase_comp", ("Liq", "OH_-")): design_variables["expt_base_in_conc_M"],
        # ("conc_mass_phase_comp", ("Liq", "H2O")): 1e3,
    }

    m.fs.feed_dilu.properties.calculate_state(
        init_arg_diluate,
        hold_state=True,
    )
    m.fs.feed_acid.properties.calculate_state(
        init_arg_acidate,
        hold_state=True,
    )

    m.fs.feed_base.properties.calculate_state(
        init_arg_basate,
        hold_state=True,
    )

    # construct stage objects as pyomo sets
    m.fs.nstages = RangeSet(nstages)
    m.fs.first_stage = RangeSet(1)
    m.fs.nonfirst_stages = m.fs.nstages - m.fs.first_stage
    # noinspection PyUnresolvedReferences
    m.fs.stage = IndexedBlock(m.fs.nstages)
    for n in m.fs.stage:
        factor = 10

        m.fs.stage[n].pump_dilu = Pump(dynamic=False, property_package=m.fs.properties)
        m.fs.stage[n].pump_dilu.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
            costing_method_arguments={"pump_type": "low_pressure"},
        )

        m.fs.stage[n].pump_dilu.efficiency_pump.fix(costing_parameters["pump_efficiency"])
        m.fs.stage[n].pump_dilu.control_volume.properties_out[0].pressure.fix(101325 * factor)
        iscale.set_scaling_factor(m.fs.stage[n].pump_dilu.control_volume.work, 1e-2)
        m.fs.stage[n].pump_dilu.work_mechanical.setlb(0)

        m.fs.stage[n].pump_acid = Pump(dynamic=False, property_package=m.fs.properties)
        m.fs.stage[n].pump_acid.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
            costing_method_arguments={"pump_type": "low_pressure"},
        )

        m.fs.stage[n].pump_acid.efficiency_pump.fix(costing_parameters["pump_efficiency"])
        m.fs.stage[n].pump_acid.control_volume.properties_out[0].pressure.fix(1013250 * factor)
        iscale.set_scaling_factor(m.fs.stage[n].pump_acid.control_volume.work, 1e-2)
        m.fs.stage[n].pump_acid.work_mechanical.setlb(0)

        m.fs.stage[n].pump_base = Pump(dynamic=False, property_package=m.fs.properties)
        m.fs.stage[n].pump_base.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
            costing_method_arguments={"pump_type": "low_pressure"},
        )
        m.fs.stage[n].pump_base.efficiency_pump.fix(costing_parameters["pump_efficiency"])
        m.fs.stage[n].pump_base.control_volume.properties_out[0].pressure.fix(1013250 * factor)
        iscale.set_scaling_factor(m.fs.stage[n].pump_base.control_volume.work, 1e-2)
        m.fs.stage[n].pump_base.work_mechanical.setlb(0)


        m.fs.stage[n].bpmed = Bipolar_and_Electrodialysis1D(
            property_package=m.fs.properties,
            has_pressure_change=True,
            pressure_drop_method=PressureDropMethod.Darcy_Weisbach,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            friction_factor_method=FrictionFactorMethod.Gurreri,
            hydraulic_diameter_method=HydraulicDiameterMethod.conventional,
            has_catalyst=True,
            salt_calculation=True,
            limiting_current_density_method_bpem=LimitingCurrentDensitybpemMethod.Empirical,
        )
        m.fs.stage[n].bpmed.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
            costing_method_arguments={
                "cost_electricity_flow": True,
                "has_rectifier": True,
            },
        )
        fix_modular_bpmed(m.fs.stage[n].bpmed)


    # costing model
    m.fs.costing.cost_process()
    define_costing_model(m)

    # construct arcs
    m.fs.s00a = Arc(
        source=m.fs.feed_dilu.outlet,
        destination=m.fs.stage[1].pump_dilu.inlet,
    )
    m.fs.s00b = Arc(
        source=m.fs.feed_acid.outlet,
        destination=m.fs.stage[1].pump_acid.inlet,
    )
    m.fs.s00c = Arc(
        source=m.fs.feed_base.outlet,
        destination=m.fs.stage[1].pump_base.inlet,
    )
    m.fs.s01a = Arc(
        m.fs.nstages,
        rule=lambda blk, n: {
            "source": m.fs.stage[n].pump_dilu.outlet,
            "destination": m.fs.stage[n].bpmed.inlet_diluate,
        }
    )
    m.fs.s01b = Arc(
        m.fs.nstages,
        rule=lambda blk, n: {
            "source": m.fs.stage[n].pump_acid.outlet,
            "destination": m.fs.stage[n].bpmed.inlet_acidate,
        }
    )
    m.fs.s01c = Arc(
        m.fs.nstages,
        rule=lambda blk, n: {
            "source": m.fs.stage[n].pump_base.outlet,
            "destination": m.fs.stage[n].bpmed.inlet_basate,
        }
    )

    m.fs.s02a = Arc(
        m.fs.nstages,
        rule=lambda blk, n: {
            "source": m.fs.stage[n].bpmed.outlet_diluate,
            "destination": m.fs.dilu_out.inlet,
        } if n == nstages else {
            "source": m.fs.stage[n].bpmed.outlet_diluate,
            "destination": m.fs.stage[n + 1].pump_dilu.inlet,
        }
    )

    m.fs.s02b = Arc(
        m.fs.nstages,
        rule=lambda blk, n: {
            "source": m.fs.stage[n].bpmed.outlet_acidate,
            "destination": m.fs.acid_out.inlet,
        } if n == nstages else {
            "source": m.fs.stage[n].bpmed.outlet_acidate,
            "destination": m.fs.stage[n + 1].pump_acid.inlet,
        }
    )

    m.fs.s02c = Arc(
        m.fs.nstages,
        rule=lambda blk, n: {
            "source": m.fs.stage[n].bpmed.outlet_basate,
            "destination": m.fs.base_out.inlet,
        } if n == nstages else {
            "source": m.fs.stage[n].bpmed.outlet_basate,
            "destination": m.fs.stage[n + 1].pump_base.inlet,
        }
    )
    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def initialize_model(m, solver=None):
    iscale.calculate_scaling_factors(m)

    if solver is None:
        solver = get_solver()
    optarg = solver.options

    # initialize models
    m.fs.feed_dilu.initialize()
    m.fs.feed_acid.initialize()
    m.fs.feed_base.initialize()
    propagate_state(m.fs.s00a)
    propagate_state(m.fs.s00b)
    propagate_state(m.fs.s00c)
    for n in m.fs.stage:
        m.fs.stage[n].pump_dilu.initialize(optarg=optarg)
        m.fs.stage[n].pump_acid.initialize(optarg=optarg)
        m.fs.stage[n].pump_base.initialize(optarg=optarg)
        propagate_state(m.fs.s01a[n])
        propagate_state(m.fs.s01b[n])
        propagate_state(m.fs.s01c[n])

        m.fs.stage[n].bpmed.initialize()
        m.fs.stage[n].bpmed.costing.initialize()
        propagate_state(m.fs.s02a[n])
        propagate_state(m.fs.s02b[n])
        propagate_state(m.fs.s02c[n])
    m.fs.dilu_out.initialize()
    m.fs.acid_out.initialize()
    m.fs.base_out.initialize()

    m.fs.costing.initialize()



def fix_modular_bpmed(blk_stack):
    # import membrane property data
    design_variables, model_parameters, costing_parameters = get_variables()

    # fix ed properties and design
    # pairs in system
    blk_stack.cell_triplet_num.fix(design_variables["cell_triplet_num"])
    blk_stack.electrical_stage_num.fix(design_variables["electrical_stage_num"])
    # single pair sizing
    blk_stack.cell_length.fix(design_variables["cell_length"])
    blk_stack.cell_width.fix(design_variables["cell_width"])
    blk_stack.channel_height.fix(design_variables["channel_height"])

    blk_stack.spacer_porosity.fix(design_variables["spacer_porosity"])
    blk_stack.shadow_factor.fix(design_variables["shadow_factor"])
    blk_stack.membrane_thickness["aem"].fix(design_variables["membrane_thickness_aem"])
    blk_stack.membrane_thickness["cem"].fix(design_variables["membrane_thickness_cem"])
    blk_stack.membrane_thickness["bpem"].fix(design_variables["membrane_thickness_bpem"])
    blk_stack.diffus_mass.fix(model_parameters["diffus_mass"])
    # membrane transport properties
    blk_stack.ion_trans_number_membrane["aem", "Na_+"].fix(1 - model_parameters["ion_trans_number_aem"])
    blk_stack.ion_trans_number_membrane["aem", "Cl_-"].fix(model_parameters["ion_trans_number_aem"])
    blk_stack.ion_trans_number_membrane["cem", "Na_+"].fix(model_parameters["ion_trans_number_cem"])
    blk_stack.ion_trans_number_membrane["cem", "Cl_-"].fix(1 - model_parameters["ion_trans_number_cem"])

    blk_stack.ion_trans_number_membrane["aem", "H_+"].fix(0)
    blk_stack.ion_trans_number_membrane["aem", "OH_-"].fix(0)
    blk_stack.ion_trans_number_membrane["cem", "H_+"].fix(0)
    blk_stack.ion_trans_number_membrane["cem", "OH_-"].fix(0)

    blk_stack.ion_trans_number_membrane["bpem", "Na_+"].fix(0)
    blk_stack.ion_trans_number_membrane["bpem", "Cl_-"].fix(0)
    blk_stack.ion_trans_number_membrane["bpem", "H_+"].fix(model_parameters["ion_trans_number_bpem"])
    blk_stack.ion_trans_number_membrane["bpem", "OH_-"].fix(model_parameters["ion_trans_number_bpem"])

    blk_stack.water_trans_number_membrane["cem"].fix(model_parameters["water_trans_number_cem"])
    blk_stack.water_trans_number_membrane["aem"].fix(model_parameters["water_trans_number_aem"])
    blk_stack.water_trans_number_membrane["bpem"].fix(model_parameters["water_trans_number_bpem"])

    blk_stack.solute_diffusivity_membrane["bpem", "Na_+"].fix(0)
    blk_stack.solute_diffusivity_membrane["bpem", "Cl_-"].fix(0)
    blk_stack.solute_diffusivity_membrane["bpem", "H_+"].fix(0)
    blk_stack.solute_diffusivity_membrane["bpem", "OH_-"].fix(0)

    blk_stack.solute_diffusivity_membrane["cem", "H_+"].fix(0)
    blk_stack.solute_diffusivity_membrane["aem", "H_+"].fix(0)
    blk_stack.solute_diffusivity_membrane["cem", "OH_-"].fix(0)
    blk_stack.solute_diffusivity_membrane["aem", "OH_-"].fix(0)

    blk_stack.solute_diffusivity_membrane["cem", "Na_+"].fix(model_parameters["ion_diffus_na_cem"])
    blk_stack.solute_diffusivity_membrane["aem", "Na_+"].fix(model_parameters["ion_diffus_na_aem"])
    blk_stack.solute_diffusivity_membrane["cem", "Cl_-"].fix(model_parameters["ion_diffus_cl_cem"])
    blk_stack.solute_diffusivity_membrane["aem", "Cl_-"].fix(model_parameters["ion_diffus_cl_aem"])

    blk_stack.water_permeability_membrane["bpem"].fix(model_parameters["water_perm_bpem"])
    blk_stack.water_permeability_membrane["cem"].fix(model_parameters["water_perm_cem"])
    blk_stack.water_permeability_membrane["aem"].fix(model_parameters["water_perm_aem"])

    blk_stack.current_utilization.fix(model_parameters["current_utilization"])
    blk_stack.electrodes_resistance.fix(model_parameters["electrodes_resistance"])
    blk_stack.voltage_applied[0].fix(model_parameters["voltage"])


    blk_stack.membrane_fixed_charge.fix(model_parameters["fixed_charge"])
    blk_stack.conc_water.fix(model_parameters["conc_water"])
    blk_stack.kr.fix(model_parameters["kr"])
    blk_stack.k2_zero.fix(model_parameters["k2_zero"])
    blk_stack.relative_permittivity.fix(model_parameters["relative_permittivity"])

    blk_stack.membrane_fixed_catalyst_cem.fix(model_parameters["catalyst_cem"])
    blk_stack.membrane_fixed_catalyst_aem.fix(model_parameters["catalyst_aem"])
    blk_stack.k_a.fix(model_parameters["k_a"])
    blk_stack.k_b.fix(model_parameters["k_b"])


    iscale.set_scaling_factor(blk_stack.pressure_drop, 1e-5)
    iscale.set_scaling_factor(blk_stack.pressure_drop_total, 1e-5)
    iscale.set_scaling_factor(blk_stack.N_Re, 1e-2)
    iscale.set_scaling_factor(blk_stack.N_Sh, 1)
    iscale.set_scaling_factor(blk_stack.N_Sc, 1)
    iscale.set_scaling_factor(blk_stack.velocity_diluate, 1e2)
    iscale.set_scaling_factor(blk_stack.velocity_acidate, 1e2)
    iscale.set_scaling_factor(blk_stack.velocity_basate, 1e2)
    iscale.set_scaling_factor(blk_stack.friction_factor, 1e-2)
    iscale.set_scaling_factor(blk_stack.hydraulic_diameter, 1e3)
    iscale.set_scaling_factor(blk_stack.k_a, 1e-2)
    iscale.set_scaling_factor(blk_stack.k_b, 1e-3)
    iscale.set_scaling_factor(blk_stack.flux_splitting, 1e4)
    iscale.set_scaling_factor(blk_stack.current_density_x, 1e-3)
    iscale.set_scaling_factor(blk_stack.voltage_x, 1e-2)
    iscale.set_scaling_factor(blk_stack.cell_triplet_num, 1e-2)
    iscale.set_scaling_factor(blk_stack.membrane_areal_resistance_combined, 1e2)
    iscale.set_scaling_factor(blk_stack.electrodes_resistance, 1e2)

    # custom scaling if needed
    fixed_var_sf(blk_stack)


def define_costing_model(m):
    conc_unit_mol = 1 * pyunits.mole * pyunits.meter ** -3
    #
    m.fs.Base_exit_conc = Expression(
        expr=smooth_min(m.fs.base_out.properties[0].conc_mol_phase_comp["Liq", "Na_+"] / conc_unit_mol,
                        m.fs.base_out.properties[0].conc_mol_phase_comp[
                            "Liq", "OH_-"] / conc_unit_mol) * conc_unit_mol
    )
    iscale.set_scaling_factor(m.fs.Base_exit_conc, 1e-2)

    conc_unit_flow_mole = 1e-1 * pyunits.mole * pyunits.second ** -1

    m.fs.Base_produced = Expression(
        expr=smooth_min(m.fs.base_out.properties[0].flow_mol_phase_comp["Liq", "Na_+"] / conc_unit_flow_mole,
                        m.fs.base_out.properties[0].flow_mol_phase_comp[
                            "Liq", "OH_-"] / conc_unit_flow_mole) * conc_unit_flow_mole * (
                     m.fs.feed_base.config.property_package.mw_comp["Na_+"] +
                     m.fs.feed_base.config.property_package.mw_comp["OH_-"])

             -
             smooth_min(m.fs.feed_base.properties[0].flow_mol_phase_comp["Liq", "Na_+"] / conc_unit_flow_mole,
                        m.fs.feed_base.properties[0].flow_mol_phase_comp[
                            "Liq", "OH_-"] / conc_unit_flow_mole) * conc_unit_flow_mole * (
                     m.fs.feed_base.config.property_package.mw_comp["Na_+"] +
                     m.fs.feed_base.config.property_package.mw_comp["OH_-"])

    )
    iscale.set_scaling_factor(m.fs.Base_produced, 1e3)

    m.fs.Acid_produced = Expression(
        expr=m.fs.acid_out.properties[0].flow_mass_phase_comp["Liq", "H_+"] +
                        m.fs.acid_out.properties[0].flow_mass_phase_comp[
                            "Liq", "Cl_-"]
    )
    iscale.set_scaling_factor(m.fs.Acid_produced, 1e2)


    m.fs.Acid_exit_conc = Expression(
        expr=(m.fs.acid_out.properties[0].conc_mass_phase_comp["Liq", "H_+"] +
                        m.fs.acid_out.properties[0].conc_mass_phase_comp[
                            "Liq", "Cl_-"]) / (m.fs.feed_acid.config.property_package.mw_comp["H_+"] + m.fs.feed_acid.config.property_package.mw_comp["Cl_-"])
    )
    iscale.set_scaling_factor(m.fs.Acid_exit_conc, 1e-2)




    design_variables, model_parameters, costing_parameters = get_variables()

    # defing costing parameters
    # TEA parameters
    m.fs.costing.total_investment_factor.fix(costing_parameters["total_investment_factor"])
    m.fs.costing.maintenance_labor_chemical_factor.fix(costing_parameters["maintenance_labor_chemical_factor"])
    m.fs.costing.utilization_factor.fix(costing_parameters["utilization_factor"])
    m.fs.costing.electricity_cost.fix(costing_parameters["electricity_cost"])
    m.fs.costing.electrical_carbon_intensity.fix(costing_parameters["electrical_carbon_intensity"])
    m.fs.costing.plant_lifetime.fix(costing_parameters["plant_lifetime"])
    m.fs.costing.wacc.fix(costing_parameters["wacc"])
    m.fs.costing.TIC.fix(costing_parameters["TIC"])
    # electrodialysis costing
    m.fs.costing.bipolar_electrodialysis_costing.membrane_capital_cost.fix(costing_parameters["membrane_capital_cost"])
    m.fs.costing.bipolar_electrodialysis_costing.factor_membrane_replacement.fix(
        costing_parameters["factor_membrane_replacement"])
    m.fs.costing.bipolar_electrodialysis_costing.stack_electrode_capital_cost.fix(
        costing_parameters["stack_electrode_capital_cost"])
    m.fs.costing.bipolar_electrodialysis_costing.factor_stack_electrode_replacement.fix(
        costing_parameters["factor_stack_electrode_replacement"])
    # pump costing
    m.fs.costing.low_pressure_pump.cost.fix(costing_parameters["pump_cost"])

    ref_density = 1 * pyunits.kg * pyunits.meter ** -3

    # create custom cost expressions
    m.fs.costing.add_specific_energy_consumption(pyunits.convert(m.fs.Base_produced / ref_density,
                                                                 to_units=pyunits.meter ** 3 / pyunits.hr),
                                                 name="specific_energy_consumption_base_produced")
    m.fs.costing.add_specific_energy_consumption(pyunits.convert(m.fs.Acid_produced / ref_density,
                                                                 to_units=pyunits.meter ** 3 / pyunits.hr),
                                                 name="specific_energy_consumption_acid_produced")
    m.fs.costing.LCOB = Expression(
        expr=pyunits.convert(
            m.fs.costing.total_annualized_cost * m.fs.costing.utilization_factor
            / m.fs.Base_produced,
            to_units=m.fs.costing.base_currency / pyunits.kg)
    )



def model_solve(m, dof_check=True, tee=False, output_show=False):
    # assert model is appropriately specified
    assert_units_consistent(m)
    dof = istat.degrees_of_freedom(m)
    print(f"degress of freedom for model: {dof}")
    if dof_check:
        assert_degrees_of_freedom(m, expected_dof=0)

    # solve model
    solver.options["max_iter"] = 10000
    res = solver.solve(m, tee=tee)


    print('solver termination condition:', res.solver.termination_condition)

    if not res.solver.termination_condition == "optimal" or output_show:

        target = m
        badly_scaled_var_list = iscale.badly_scaled_var_generator(target, large=1e4, small=1e-4)
        print("\n----------------   badly scaled variables   ----------------")
        for x in badly_scaled_var_list:
            try:
                print(f"{x[0].name:<80s}{value(x[0]):<20.5e}sf: {iscale.get_scaling_factor(x[0]):<20.5e}")
            except:
                continue

        print("\n---------------- variables near bounds ----------------")
        variables_near_bounds_list = istat.variables_near_bounds_generator(target, abs_tol=1e-8, rel_tol=1e-8)
        for x in variables_near_bounds_list:
            try:
                print(f"{x.name:<80s}{value(x):<20.5e}")
            except:
                continue

        print("\n---------------- violated constraints ----------------")
        total_constraints_set_list = istat.total_constraints_set(target)
        for x in total_constraints_set_list:
            try:
                residual = abs(value(x.body) - value(x.lb))
                if residual > 1e-8:
                    print(f"{x.name:<80s}{value(x.body):<20.5e}{residual:<20.5e}")
            except:
                residual = abs(value(x.body) - value(x.ub))
                if residual > 1e-8:
                    print(f"{x.name:<80s}{value(x.body):<20.5e}{residual:<20.5e}")
            finally:
                continue

    assert_optimal_termination(res)

    return res



def fixed_var_sf(block):
    for var in block.component_data_objects(ctype=Var, active=True, descend_into=False):
        if var.fixed:
            # calculate sf to the inverse fixed value
            if value(var) == 0:
                sf = 1
            else:
                sf = 1 * 10 ** -math.ceil(math.log10(abs(value(var))))
            # set sf for vars, sf is redundantly set for is_indexed vars
            if var.parent_component().is_indexed():
                iscale.set_scaling_factor(var.parent_component(), sf)
            else:
                iscale.set_scaling_factor(var, sf)


if __name__ == "__main__":
    nstages = 1


    # model build
    m = build_modular_bpmed(nstages)
    dt = DiagnosticsToolbox(m)
    # dt.report_structural_issues()
    # dt.report_numerical_issues()
    initialize_model(m)

    res = model_solve(m)

    ref_density = 1 * pyunits.kg * pyunits.meter ** -3


    print("SEC base produced  =",
          value(m.fs.costing.specific_energy_consumption_base_produced / ref_density),
          pyunits.get_units(m.fs.costing.specific_energy_consumption_base_produced / ref_density))

    print("Total electricity  =",
          value(m.fs.costing.aggregate_flow_electricity),
          pyunits.get_units(m.fs.costing.aggregate_flow_electricity))

    print("Total annual cost  =",
          value(m.fs.costing.total_annualized_cost),
          pyunits.get_units(m.fs.costing.total_annualized_cost))

    print("Levelized cost of Base produced  =",
          value(m.fs.costing.LCOB),
          pyunits.get_units(m.fs.costing.LCOB))


    # optimization build
    build_flowsheet_optimization(m)

    model_solve(m, dof_check=False, tee=False, output_show=True)

    # m.fs.stage[1].bpmed.cell_triplet_num.setub(750)
    #
    # model_solve(m, dof_check=False, tee=False, output_show=False)

    print("---After optimisation:---")


    print("SEC base produced  =",
          value(m.fs.costing.specific_energy_consumption_base_produced / ref_density),
          pyunits.get_units(m.fs.costing.specific_energy_consumption_base_produced / ref_density))
    print("Total electricity  =",
          value(m.fs.costing.aggregate_flow_electricity),
          pyunits.get_units(m.fs.costing.aggregate_flow_electricity))

    print("Total annual cost  =",
          value(m.fs.costing.total_annualized_cost),
          pyunits.get_units(m.fs.costing.total_annualized_cost))

    print("Levelized cost of Base produced  =",
          value(m.fs.costing.LCOB),
          pyunits.get_units(m.fs.costing.LCOB))

    # Parametrci sweep setup

    design_variables, model_parameters, costing_parameters = get_variables()

    def build_sweep_params(m, num_samples=2, **kwargs):
        sweep_params = dict()

        sweep_params['Inlet acid conc (mol/m3)'] = LinearSample(m.fs.conc_acid_mol, 50,200, 20)

        return sweep_params

    def build_outputs(m, **kwargs):
        outputs = dict()
        for n in m.fs.nstages:
            outputs[f'cell_triplet_num_{n}'] = m.fs.stage[n].bpmed.cell_triplet_num
            outputs[f'cell width -{n} (m)'] = m.fs.stage[n].bpmed.cell_width
            outputs[f'cell length -{n} (m)'] = m.fs.stage[n].bpmed.cell_length
            outputs[f'cell channel height -{n} (m)'] = m.fs.stage[n].bpmed.channel_height
            outputs[f'channel velocity at exit - {n}(m/s)'] = m.fs.stage[n].bpmed.velocity_diluate[0, 1]
            outputs[f'Voltage - {n}(V)'] = m.fs.stage[n].bpmed.voltage_applied[0]
            outputs[f'current density -{n} (A/m2)'] = m.fs.stage[n].bpmed.current_density_x[0, 1]
            outputs[f'current density limiting bpem -{n} (A/m2)'] = m.fs.stage[n].bpmed.current_dens_lim_bpem[0, 1]
            outputs['AC power (KW)'] = m.fs.stage[n].bpmed.costing.ac_power
            outputs['Pressure ratio diluate'] = m.fs.stage[n].bpmed.diluate.properties[0, 1].pressure / (
                    101325 * pyunits.Pa)

        outputs['LCObase ($/kg)'] = m.fs.costing.LCOB
        outputs['SECbase(kW hr/kg)'] = m.fs.costing.specific_energy_consumption_base_produced / ref_density
        outputs['Total annual cost '] = m.fs.costing.total_annualized_cost
        outputs['Total electricity (kW)'] = m.fs.costing.aggregate_flow_electricity
        outputs['Base output conc (mol/m3)'] = m.fs.Base_exit_conc
        outputs['Acid output conc (mol/m3)'] = m.fs.Acid_exit_conc

        outputs['Na Basate (mol/m3)'] = m.fs.base_out.properties[0].conc_mol_phase_comp["Liq", "Na_+"]
        outputs['OH Basate (mol/m3)'] = m.fs.base_out.properties[0].conc_mol_phase_comp["Liq", "OH_-"]
        outputs['H Basate (mol/m3)'] = m.fs.base_out.properties[0].conc_mol_phase_comp["Liq", "H_+"]
        outputs['Cl Basate (mol/m3)'] = m.fs.base_out.properties[0].conc_mol_phase_comp["Liq", "Cl_-"]

        outputs['elec migration mono cem Na'] = m.fs.stage[1].bpmed.elec_migration_mono_cem_flux[0.0, 1, 'Liq', 'Na_+']
        outputs['nonelec migration mono cem Na'] = m.fs.stage[1].bpmed.nonelec_mono_cem_flux[0.0, 1, 'Liq', 'Na_+']

        outputs['Base output flow (kg/s)'] = m.fs.Base_produced
        outputs['Acid output flow (kg/s)'] = m.fs.Acid_produced

        conc_ref = 1 * pyunits.mole * pyunits.meter ** -3
        outputs['Inlet salt (mol/m3)'] = smooth_min(
            m.fs.feed_dilu.properties[0].conc_mol_phase_comp["Liq", "Na_+"] / conc_ref,
            m.fs.feed_dilu.properties[0].conc_mol_phase_comp["Liq", "Cl_-"] / conc_ref)
        outputs['Outlet salt (mol/m3)'] = smooth_min(
            m.fs.dilu_out.properties[0].conc_mol_phase_comp["Liq", "Na_+"] / conc_ref,
            m.fs.dilu_out.properties[0].conc_mol_phase_comp["Liq", "Cl_-"] / conc_ref)

        return outputs


    parameter_sweep(m, build_sweep_params,
                    build_outputs,
                    csv_results_file_name='output/Sample_results.csv',
                    h5_results_file_name='output/Sample_results.h5',
                    h5_parent_group_name="results", )