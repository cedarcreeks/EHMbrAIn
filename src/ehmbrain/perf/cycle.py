"""CFM56-7B26 thermodynamic cycle model (pyCycle).

Two-spool, separate-flow, cooled high-bypass turbofan. Element graph, bleed
network and balance structure follow pyCycle's validated high-bypass-turbofan
example (vendored at scripts/hbtf_reference.py); design values are re-targeted
to the CFM56-7B26 per conf/cfm56_7b_targets.yaml and docs/f1-model-spec.md.

Design point (A1): M0.78 / 35 000 ft / ISA, max-cruise thrust.
"""

import openmdao.api as om
import pycycle.api as pyc


class CFM56(pyc.Cycle):
    """Single cycle point. design=True for the design point, False for off-design."""

    def initialize(self):
        self.options.declare('throttle_mode', default='T4', values=['T4', 'percent_thrust'])
        super().initialize()

    def setup(self):
        design = self.options['design']

        if self.options['thermo_method'] == 'TABULAR':
            self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
            fuel_type = 'FAR'
        else:
            self.options['thermo_data'] = pyc.species_data.janaf
            fuel_type = 'Jet-A(g)'

        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('fan', pyc.Compressor(map_data=pyc.FanMap, bleed_names=[], map_extrap=True),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('splitter', pyc.Splitter())
        self.add_subsystem('duct4', pyc.Duct())
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap, map_extrap=True),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('duct6', pyc.Duct())
        self.add_subsystem('hpc', pyc.Compressor(map_data=pyc.HPCMap,
                                                 bleed_names=['cool1', 'cool2', 'cust'], map_extrap=True),
                           promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(bleed_names=['cool3', 'cool4']))
        self.add_subsystem('burner', pyc.Combustor(fuel_type=fuel_type))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap,
                                              bleed_names=['cool3', 'cool4'], map_extrap=True),
                           promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('duct11', pyc.Duct())
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap,
                                              bleed_names=['cool1', 'cool2'], map_extrap=True),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('duct13', pyc.Duct())
        self.add_subsystem('core_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))

        self.add_subsystem('byp_bld', pyc.BleedOut(bleed_names=['bypBld']))
        self.add_subsystem('duct15', pyc.Duct())
        self.add_subsystem('byp_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))

        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=3), promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=2), promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=2, num_burners=1))

        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('core_nozz.Fg', 'perf.Fg_0')
        self.connect('byp_nozz.Fg', 'perf.Fg_1')

        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        self.connect('fc.Fl_O:stat:P', 'core_nozz.Ps_exhaust')
        self.connect('fc.Fl_O:stat:P', 'byp_nozz.Ps_exhaust')

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            # W to hit thrust; FAR to hit T4; turbine PRs to zero net shaft power.
            balance.add_balance('W', units='lbm/s', eq_units='lbf')
            self.connect('balance.W', 'fc.W')
            self.connect('perf.Fn', 'balance.lhs:W')
            self.promotes('balance', inputs=[('rhs:W', 'Fn_DES')])

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
            self.promotes('balance', inputs=[('rhs:FAR', 'T4_MAX')])

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lpt_PR')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hpt_PR')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hpt_PR')
        else:
            # W and BPR to hold the design nozzle areas; spool speeds to zero net power;
            # FAR to hold either T4 (rating) or a thrust fraction (throttle).
            if self.options['throttle_mode'] == 'T4':
                balance.add_balance('FAR', val=0.017, lower=1e-4, eq_units='degR')
                self.connect('balance.FAR', 'burner.Fl_I:FAR')
                self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
                self.promotes('balance', inputs=[('rhs:FAR', 'T4_MAX')])
            else:
                balance.add_balance('FAR', val=0.017, lower=1e-4, eq_units='lbf', use_mult=True)
                self.connect('balance.FAR', 'burner.Fl_I:FAR')
                self.connect('perf.Fn', 'balance.rhs:FAR')
                self.promotes('balance', inputs=[('mult:FAR', 'PC'), ('lhs:FAR', 'Fn_max')])

            balance.add_balance('W', units='lbm/s', lower=10., upper=1000., eq_units='inch**2')
            self.connect('balance.W', 'fc.W')
            self.connect('core_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=2., upper=10., eq_units='inch**2')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('byp_nozz.Throat:stat:area', 'balance.lhs:BPR')

            balance.add_balance('lp_Nmech', val=1.5, units='rpm', lower=500.,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lp_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lp_Nmech')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lp_Nmech')

            balance.add_balance('hp_Nmech', val=1.5, units='rpm', lower=500.,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hp_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hp_Nmech')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hp_Nmech')

        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'splitter.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O1', 'duct4.Fl_I')
        self.pyc_connect_flow('duct4.Fl_O', 'lpc.Fl_I')
        self.pyc_connect_flow('lpc.Fl_O', 'duct6.Fl_I')
        self.pyc_connect_flow('duct6.Fl_O', 'hpc.Fl_I')
        self.pyc_connect_flow('hpc.Fl_O', 'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct11.Fl_I')
        self.pyc_connect_flow('duct11.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'duct13.Fl_I')
        self.pyc_connect_flow('duct13.Fl_O', 'core_nozz.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O2', 'byp_bld.Fl_I')
        self.pyc_connect_flow('byp_bld.Fl_O', 'duct15.Fl_I')
        self.pyc_connect_flow('duct15.Fl_O', 'byp_nozz.Fl_I')

        self.pyc_connect_flow('hpc.cool1', 'lpt.cool1', connect_stat=False)
        self.pyc_connect_flow('hpc.cool2', 'lpt.cool2', connect_stat=False)
        self.pyc_connect_flow('bld3.cool3', 'hpt.cool3', connect_stat=False)
        self.pyc_connect_flow('bld3.cool4', 'hpt.cool4', connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-8
        newton.options['rtol'] = 1e-99
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 50
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 1000
        newton.options['reraise_child_analysiserror'] = False
        ls = newton.linesearch = om.ArmijoGoldsteinLS()
        ls.options['maxiter'] = 3
        ls.options['rho'] = 0.75

        self.linear_solver = om.DirectSolver()

        super().setup()


class MPCFM56(pyc.MPCycle):
    """Multi-point model: DESIGN (A1 cruise) + SLS anchors A2–A5 (WP1.2).

    Anchors are thrust-matched (percent_thrust throttle against the rated
    26 302 lbf takeoff thrust) so that fuel flow is a *prediction* to compare
    against the ICAO EEDB measurements in conf/cfm56_7b_targets.yaml.
    """

    THERMO = 'TABULAR'  # D2: tabular for speed; CEA cross-check at end of WP1.2

    FN_TAKEOFF_LBF = 26302.0            # TCDS: 11699 daN
    OD_ANCHORS = {                      # name -> fraction of takeoff thrust (EEDB settings)
        'A2_takeoff': 1.00,
        'A3_climbout': 0.85,
        'A4_approach': 0.30,
        'A5_idle': 0.07,
    }

    def initialize(self):
        self.options.declare('od', default=True, types=bool,
                             desc='include the off-design SLS anchor points')
        super().initialize()

    def setup(self):
        self.pyc_add_pnt('DESIGN', CFM56(thermo_method=self.THERMO))

        # Flow-path Mach numbers inherited from the HBTF reference (same engine class).
        self.set_input_defaults('DESIGN.inlet.MN', 0.751)
        self.set_input_defaults('DESIGN.fan.MN', 0.4578)
        self.set_input_defaults('DESIGN.splitter.BPR', 5.1)      # CFM56-7B26 target
        self.set_input_defaults('DESIGN.splitter.MN1', 0.3104)
        self.set_input_defaults('DESIGN.splitter.MN2', 0.4518)
        self.set_input_defaults('DESIGN.duct4.MN', 0.3121)
        self.set_input_defaults('DESIGN.lpc.MN', 0.3059)
        self.set_input_defaults('DESIGN.duct6.MN', 0.3563)
        self.set_input_defaults('DESIGN.hpc.MN', 0.2442)
        self.set_input_defaults('DESIGN.bld3.MN', 0.3000)
        self.set_input_defaults('DESIGN.burner.MN', 0.1025)
        self.set_input_defaults('DESIGN.hpt.MN', 0.3650)
        self.set_input_defaults('DESIGN.duct11.MN', 0.3063)
        self.set_input_defaults('DESIGN.lpt.MN', 0.4127)
        self.set_input_defaults('DESIGN.duct13.MN', 0.4463)
        self.set_input_defaults('DESIGN.byp_bld.MN', 0.4489)
        self.set_input_defaults('DESIGN.duct15.MN', 0.4589)
        # Map-scaling reference speeds. HP value chosen so that the predicted N2 at
        # rated SLS takeoff thrust stays below the TCDS redline (15 183 rpm = 105 %)
        # given the generic HPCMap speed-flow shape; with 13 940 rpm at design the
        # model gives ~14 900 rpm (103 %) at A2. Speed labels are approximate by
        # construction (generic maps) — documented limitation.
        self.set_input_defaults('DESIGN.LP_Nmech', 4666.0, units='rpm')
        self.set_input_defaults('DESIGN.HP_Nmech', 13940.0, units='rpm')

        # Pressure losses, nozzle coefficients and the secondary-air network (D5).
        self.pyc_add_cycle_param('inlet.ram_recovery', 0.9990)
        self.pyc_add_cycle_param('duct4.dPqP', 0.0048)
        self.pyc_add_cycle_param('duct6.dPqP', 0.0101)
        self.pyc_add_cycle_param('burner.dPqP', 0.0540)
        self.pyc_add_cycle_param('duct11.dPqP', 0.0051)
        self.pyc_add_cycle_param('duct13.dPqP', 0.0107)
        self.pyc_add_cycle_param('duct15.dPqP', 0.0149)
        self.pyc_add_cycle_param('core_nozz.Cv', 0.9933)
        self.pyc_add_cycle_param('byp_bld.bypBld:frac_W', 0.005)
        self.pyc_add_cycle_param('byp_nozz.Cv', 0.9939)
        self.pyc_add_cycle_param('hpc.cool1:frac_W', 0.050708)   # HPT vane cooling
        self.pyc_add_cycle_param('hpc.cool1:frac_P', 0.5)
        self.pyc_add_cycle_param('hpc.cool1:frac_work', 0.5)
        self.pyc_add_cycle_param('hpc.cool2:frac_W', 0.020274)   # HPT blade cooling
        self.pyc_add_cycle_param('hpc.cool2:frac_P', 0.55)
        self.pyc_add_cycle_param('hpc.cool2:frac_work', 0.5)
        self.pyc_add_cycle_param('bld3.cool3:frac_W', 0.067214)  # LPT cooling
        self.pyc_add_cycle_param('bld3.cool4:frac_W', 0.101256)
        self.pyc_add_cycle_param('hpc.cust:frac_P', 0.5)
        self.pyc_add_cycle_param('hpc.cust:frac_work', 0.5)
        self.pyc_add_cycle_param('hpc.cust:frac_W', 0.0445)      # customer bleed
        self.pyc_add_cycle_param('hpt.cool3:frac_P', 1.0)
        self.pyc_add_cycle_param('hpt.cool4:frac_P', 0.0)
        self.pyc_add_cycle_param('lpt.cool1:frac_P', 1.0)
        self.pyc_add_cycle_param('lpt.cool2:frac_P', 0.0)
        self.pyc_add_cycle_param('hp_shaft.HPX', 250.0, units='hp')  # accessory power extraction

        if self.options['od']:
            for name, pc in self.OD_ANCHORS.items():
                self.pyc_add_pnt(name, CFM56(design=False, thermo_method=self.THERMO,
                                             throttle_mode='percent_thrust'))
                self.set_input_defaults(f'{name}.fc.MN', 1e-3)  # SLS static
                self.set_input_defaults(f'{name}.fc.alt', 0.0, units='ft')
                self.set_input_defaults(f'{name}.fc.dTs', 0.0, units='degR')
                self.set_input_defaults(f'{name}.Fn_max', self.FN_TAKEOFF_LBF, units='lbf')
                self.set_input_defaults(f'{name}.PC', pc)

            self.od_pts = list(self.OD_ANCHORS)
            self.pyc_use_default_des_od_conns()
            self.pyc_connect_des_od('core_nozz.Throat:stat:area', 'balance.rhs:W')
            self.pyc_connect_des_od('byp_nozz.Throat:stat:area', 'balance.rhs:BPR')

        super().setup()


# CFM56-7B26 design-point inputs (A1 anchor, conf/cfm56_7b_targets.yaml).
DESIGN_INPUTS = {
    'DESIGN.fc.alt': (35000.0, 'ft'),
    'DESIGN.fc.MN': (0.78, None),
    'DESIGN.T4_MAX': (2857.0, 'degR'),
    'DESIGN.Fn_DES': (5480.0, 'lbf'),
    'DESIGN.fan.PR': (1.685, None),
    'DESIGN.fan.eff': (0.8948, None),
    'DESIGN.lpc.PR': (1.935, None),
    'DESIGN.lpc.eff': (0.9243, None),
    # Calibrated so the SLS-takeoff OPR matches the measured EEDB value (27.61);
    # the resulting cruise OPR is ~30 (design OPR = 1.685 * 1.935 * PR_hpc).
    'DESIGN.hpc.PR': (9.35, None),
    'DESIGN.hpc.eff': (0.8707, None),
    'DESIGN.hpt.eff': (0.8888, None),
    'DESIGN.lpt.eff': (0.8996, None),
}


# Off-design balance initial guesses per anchor. Idle values reflect the real
# engine's ~60 % N2 / ~20 % N1 ground idle; poor guesses here are the main
# cause of Newton divergence at low power.
OD_GUESSES = {
    'A2_takeoff':  dict(FAR=0.030, W=750.0, BPR=5.3, lp_Nmech=5100.0, hp_Nmech=14400.0),
    'A3_climbout': dict(FAR=0.026, W=680.0, BPR=5.4, lp_Nmech=4700.0, hp_Nmech=14000.0),
    'A4_approach': dict(FAR=0.016, W=380.0, BPR=5.9, lp_Nmech=2900.0, hp_Nmech=11800.0),
    'A5_idle':     dict(FAR=0.010, W=160.0, BPR=6.5, lp_Nmech=1500.0, hp_Nmech=9200.0),
}


def apply_od_guesses(prob, name, guesses=None):
    g = guesses or OD_GUESSES[name]
    prob[f'{name}.balance.FAR'] = g['FAR']
    prob[f'{name}.balance.W'] = g['W']
    prob[f'{name}.balance.BPR'] = g['BPR']
    prob[f'{name}.balance.lp_Nmech'] = g['lp_Nmech']
    prob[f'{name}.balance.hp_Nmech'] = g['hp_Nmech']
    prob[f'{name}.hpt.PR'] = 3.7
    prob[f'{name}.lpt.PR'] = 4.3
    for comp in ('fan', 'lpc', 'hpc'):
        prob[f'{name}.{comp}.map.RlineMap'] = 2.0


def build_problem(od=True, overrides=None):
    """Full problem (design + anchors), ready to run.

    `overrides` replaces DESIGN_INPUTS entries. od=False gives the fast
    design-point-only problem used by the WP1.1 tests.
    """
    prob = om.Problem()
    prob.model = MPCFM56(od=od)
    prob.setup()

    inputs = dict(DESIGN_INPUTS)
    if overrides:
        inputs.update(overrides)
    for name, (val, units) in inputs.items():
        prob.set_val(name, val, units=units)

    # Balance initial guesses (from the HBTF reference; robust for this engine class).
    prob['DESIGN.balance.FAR'] = 0.025
    prob['DESIGN.balance.W'] = 100.
    prob['DESIGN.balance.lpt_PR'] = 4.0
    prob['DESIGN.balance.hpt_PR'] = 3.0
    prob['DESIGN.fc.balance.Pt'] = 5.2
    prob['DESIGN.fc.balance.Tt'] = 440.0

    if od:
        for name in MPCFM56.OD_ANCHORS:
            apply_od_guesses(prob, name)

    return prob


def build_design_problem(overrides=None):
    """Design-point-only problem (WP1.1 gate)."""
    return build_problem(od=False, overrides=overrides)


def anchor_converged(prob, name, pc, fn_takeoff=None):
    """Converged iff the thrust balance is actually met (Newton can quit on maxiter)."""
    fn_takeoff = fn_takeoff or MPCFM56.FN_TAKEOFF_LBF
    fn = float(prob.get_val(f'{name}.perf.Fn', units='lbf')[0])
    return abs(fn - pc * fn_takeoff) < 0.01 * pc * fn_takeoff


def solve_anchors(prob):
    """Solve DESIGN + A2-A5 with the idle continuation strategy.

    Nozzle pressure ratios near 1.0 make PC=0.07 unreachable from a cold start,
    and a failed attempt leaves the point's internal states corrupted. So A5
    never starts cold at 0.07: it begins at PC=0.30 (which converges from cold,
    same as A4) and walks down, warm, to the target.

    Returns {point_name: converged_bool}.
    """
    apply_od_guesses(prob, 'A5_idle', OD_GUESSES['A4_approach'])
    for pc in (0.30, 0.20, 0.14, 0.10, 0.08, 0.07):
        prob.set_val('A5_idle.PC', pc)
        prob.run_model()
        if not anchor_converged(prob, 'A5_idle', pc):
            break
    prob.set_val('A5_idle.PC', MPCFM56.OD_ANCHORS['A5_idle'])

    return {name: anchor_converged(prob, name, pc)
            for name, pc in MPCFM56.OD_ANCHORS.items()}


def anchor_summary(prob, name):
    """Key off-design results for one SLS anchor point."""
    return {
        'Fn_lbf': float(prob.get_val(f'{name}.perf.Fn', units='lbf')[0]),
        'WF_kgps': float(prob.get_val(f'{name}.burner.Wfuel', units='kg/s')[0]),
        'OPR': float(prob.get_val(f'{name}.perf.OPR')[0]),
        'N1_rpm': float(prob.get_val(f'{name}.balance.lp_Nmech', units='rpm')[0]),
        'N2_rpm': float(prob.get_val(f'{name}.balance.hp_Nmech', units='rpm')[0]),
        'T4_degR': float(prob.get_val(f'{name}.burner.Fl_O:tot:T', units='degR')[0]),
        'T45_degR': float(prob.get_val(f'{name}.hpt.Fl_O:tot:T', units='degR')[0]),
        'BPR': float(prob.get_val(f'{name}.balance.BPR')[0]),
    }


def design_summary(prob):
    """Key design-point results as a plain dict (SI-free, pyCycle native units)."""
    cool_fracs = (0.050708, 0.020274, 0.067214, 0.101256)  # keep in sync with MPCFM56
    return {
        'Fn_lbf': float(prob.get_val('DESIGN.perf.Fn', units='lbf')[0]),
        'TSFC': float(prob.get_val('DESIGN.perf.TSFC')[0]),
        'BPR': float(prob.get_val('DESIGN.splitter.BPR')[0]),
        'OPR': float(prob.get_val('DESIGN.perf.OPR')[0]),
        'T4_degR': float(prob.get_val('DESIGN.burner.Fl_O:tot:T', units='degR')[0]),
        'W_lbm_s': float(prob.get_val('DESIGN.fc.W', units='lbm/s')[0]),
        'FAR': float(prob.get_val('DESIGN.balance.FAR')[0]),
        'LPT_PR': float(prob.get_val('DESIGN.balance.lpt_PR')[0]),
        'HPT_PR': float(prob.get_val('DESIGN.balance.hpt_PR')[0]),
        'cooling_total_frac': sum(cool_fracs),
    }
