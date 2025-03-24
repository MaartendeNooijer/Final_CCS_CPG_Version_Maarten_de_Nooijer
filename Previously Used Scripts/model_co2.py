import numpy as np

from darts.input.input_data import InputData
from darts.models.darts_model import DartsModel
from set_case import set_input_data
from darts.engines import value_vector

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData
from model_cpg import Model_CPG, fmt

from dataclasses import dataclass, field

from scipy.special import erf

# region Dataclasses
@dataclass
class Corey:
    nw: float
    ng: float
    swc: float
    sgc: float
    krwe: float
    krge: float
    labda: float
    p_entry: float
    pcmax: float
    c2: float
    def modify(self, std, mult):
        i = 0
        for attr, value in self.__dict__.items():
            if attr != 'type':
                setattr(self, attr, value * (1 + mult[i] * float(getattr(std, attr))))
            i += 1

    def random(self, std):
        for attr, value in self.__dict__.items():
            if attr != 'type':
                std_in = value * float(getattr(std, attr))
                param = np.random.normal(value, std_in)
                if param < 0:
                    param = 0
                setattr(self, attr, param)

class AlternativeContainer(PropertyContainer):
    def run_flash(self, pressure, temperature, zc):
        # Normalize fluid compositions
        zc_norm = zc if not self.ns else zc[:self.nc_fl] / (1. - np.sum(zc[self.nc_fl:]))

        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        error_output = self.flash_ev.evaluate(pressure, temperature, zc_norm)
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        try:
            self.x = np.array(flash_results.X).reshape(self.np_fl, self.nc_fl)
        except ValueError as e:
            print(e.args[0], pressure, temperature, zc)
            error_output += 1

        # Set present phase idxs
        ph = np.array([j for j in range(self.np_fl) if self.nu[j] > 0], dtype=int)

        if ph.size == 1:
            self.x[ph[0]] = zc_norm

        return ph


class ModelCCS(Model_CPG):
    def __init__(self):
        self.zero = 1e-10
        super().__init__()

    def set_physics(self): #NEW
        """Assign physical properties, including relative permeability, based on facies (SATNUM)."""
        #Code used in SPE11b

        #Code used in CCS
        # Define components and phases
        components = ["H2O", "CO2"]
        phases = ["Aq", "V"]
        comp_data = CompData(components, setprops=True)

        # Define Equations of State (EOS) for CO2 and H2O
        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, AQEoS.Ziabakhsh2012)

        flash_params = FlashParams(comp_data)
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]

        # Initialize physics model
        self.physics = Compositional(
            components=components,
            phases=phases,
            timer=self.timer,
            n_points=self.idata.obl.n_points,
            min_p=self.idata.obl.min_p,
            max_p=self.idata.obl.max_p,
            min_z=self.idata.obl.min_z,
            max_z=self.idata.obl.max_z,
            min_t=self.idata.obl.min_t,
            max_t=self.idata.obl.max_t,
            thermal=self.idata.obl.thermal,
            cache=self.idata.obl.cache
        )

        #self.physics.n_axes_points[0] = 101  # sets OBL points for pressure, From SPE11b

        temperature = None  # if None, then thermal=True

        facies_rel_perm = {  # based on fluidflower paper
            # # **Wells** ##NEW, values don't mind
            0: Corey(nw=1.5, ng=1.5, swc=0.10, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0, pcmax=0.1, c2=1.5),
            # p_entry=0.025602
            # **Channel Sand (Facies 1)**
            1: Corey(nw=1.5, ng=1.5, swc=0.10, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.025602, pcmax=300,
                     c2=1.5),  # , #p_entry=0.025602
            # **Overbank Sand (Facies 2)**
            2: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.038706, pcmax=300,
                     c2=1.5),  # p_entry=0.038706
            # **Shale (Facies 3)**
            3: Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1.935314, pcmax=300,
                     c2=1.5), }  # p_entry=1.935314

        # Assign properties per facies using `add_property_region`
        for i, (facies, corey_params) in enumerate(facies_rel_perm.items()): #NEW

            #Original
            # property_container = PropertyContainer(phases_name=phases, components_name=components,Mw=comp_data.Mw[:2], #NEW, was property_container = PropertyContainer(phases_name=phases, components_name=components,Mw=comp_data.Mw,
            #                                        temperature=temperature, min_z=self.zero/10)
            # #NEW
            property_container = AlternativeContainer(phases_name=phases, components_name=components,Mw=comp_data.Mw[:2], #NEW, was property_container = PropertyContainer(phases_name=phases, components_name=components,Mw=comp_data.Mw,
                                                   temperature=temperature, min_z=self.zero/10)

            # Assign phase behavior properties
            property_container.flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV]) #NEW, was property_container.flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])

            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw[:2])), #Other version just has Mw=comp_data.Mw
                                                  ('Aq', Garcia2001(components)), ]) # NEW, was property_container.density_ev = {'V': EoSDensity(pr, comp_data.Mw),'Aq': Garcia2001(components)}


            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(components)), ]) #NEW, was property_container.viscosity_ev = {'V': Fenghour1998(), 'Aq': Islam2012(components)}


            #Original:
            property_container.enthalpy_ev = {'V': EoSEnthalpy(pr),'Aq': EoSEnthalpy(aq)}

            property_container.conductivity_ev = {'V': ConstFunc(10.0), 'Aq': ConstFunc(180.0)}


            # Assign relative permeability and capillary pressure
            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params, 'V')),
                                                   ('Aq', ModBrooksCorey(corey_params, 'Aq'))]) #NEW


            property_container.capillary_pressure_ev = CapillaryPressure(corey_params)  # this line makes the code drastically slower

            property_container.output_props = {"satA": lambda ii=i: self.physics.property_containers[ii].sat[0],
                                               "satV": lambda ii=i: self.physics.property_containers[ii].sat[1],
                                               "xCO2": lambda ii=i: self.physics.property_containers[ii].x[0, 1],
                                               "xH2O": lambda ii=i: self.physics.property_containers[ii].x[0, 0], #NEW
                                               "yH2O": lambda ii=i: self.physics.property_containers[ii].x[1, 0],
                                               "yCO2": lambda ii=i: self.physics.property_containers[ii].x[1, 1],#} #NEW
                                               "Pc": lambda ii=i: self.physics.property_containers[ii].pc[1], #Pc Aq
                                               "enthV": lambda ii=i: self.physics.property_containers[ii].enthalpy[1]} # ,#}


            self.physics.add_property_region(property_container, i)

        return

    def get_arrays(self, ith_step):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        # Find index of properties to output
        ev_props = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        # If output_properties is None, all variables and properties from property_operators will be passed
        props_names = list(ev_props)
        props_names = props_names #+ ['pressure', 'temperature'] #this worked correctly first, but not anymore (Looks like it has to do with the regions)

        # print(props_names)#NEW

        timesteps, property_array = self.output_properties(output_properties=props_names, timestep=ith_step)

        #print(property_array) #NEW

        return property_array

    def set_input_data(self, case=''):
        self.idata = InputData(type_hydr='thermal', type_mech='none', init_type='uniform')
        set_input_data(self.idata, case)
        self.idata.faultfile =  self.idata.faultfile #NEW

        self.idata.geom.burden_layers = 0

        # well controls
        wdata = self.idata.well_data
        wells = wdata.wells  # short name
        # set default injection composition
        #wdata.inj = value_vector([self.zero])  # injection composition - water
        y2d = 365.25

        rate_kg_year = 1e9 #0.1 Mt
        rate_kg_day = rate_kg_year/y2d
        kg_to_mol = 44.01 #kg/kmol
        rate_kmol_day = rate_kg_day/kg_to_mol

        if 'wbhp' in case:
            print("The string 'wbhp' is found in case!")
            for w in wells:
                wdata.add_inj_bhp_control(name=w, bhp=200, comp_index=1, temperature=300)  # kmol/day | bars | K
                #wdata.add_prd_rate_control(time=10 * y2d, name=w, rate=0., comp_index=0, bhp_constraint=70)  # STOP WELL
        elif 'wrate' in case:
            for w in wells:
                wdata.add_inj_rate_control(name=w, rate=rate_kmol_day, comp_index=1, bhp_constraint=200, temperature=300) #rate=5e5/8 is approximately 1 Mt per year #was rate = 6e6 # kmol/day | bars | K
                #wdata.add_prd_rate_control(time=10 * y2d, name=w, rate=0., comp_index=0, bhp_constraint=70)  # STOP WELL

        self.idata.obl.n_points = 1000
        self.idata.obl.zero = 1e-11
        self.idata.obl.min_p = 1.
        self.idata.obl.max_p = 400.
        self.idata.obl.min_t = 273.15
        self.idata.obl.max_t = 373.15
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero
        self.idata.obl.cache = False
        self.idata.obl.thermal = True #This sets case to non-isothermal

    def set_initial_conditions(self):
        self.temperature_initial_ = 273.15 + 76.85  # K
        self.initial_values = {"pressure": 100.,
                            "H2O": 0.99995,
                            "temperature": self.temperature_initial_
                            }
        super().set_initial_conditions()

        mesh = self.reservoir.mesh
        depth = np.array(mesh.depth, copy=True) #- 2000
        # set initial pressure
        pressure_grad = 100
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = depth[:pressure.size] / 1000 * pressure_grad + 1
        # set initial temperature
        temperature_grad = 30
        temperature = np.array(mesh.temperature, copy=False)
        temperature[:] = depth[:pressure.size] / 1000 * temperature_grad + 273.15 + 20
        #temperature[:] = 350 #NEW #Deactivated to make sure that I have temperature gradient include

    def set_well_controls(self, time: float = 0., verbose=True):
        '''
        :param time: simulation time, [days]
        :return:
        '''
        inj_stream_base = [self.zero * 100]
        eps_time = 1e-15
        for w in self.reservoir.wells:
            # find next well control in controls list for different timesteps
            wctrl = None
            for wctrl_t in self.idata.well_data.wells[w.name].controls:
                if np.fabs(wctrl_t[0] - time) < eps_time:  # check time
                    wctrl = wctrl_t[1]
                    break
            if wctrl is None:
                continue
            if wctrl.type == 'inj':  # INJ well
                inj_stream = inj_stream_base
                if self.physics.thermal:
                    inj_stream += [wctrl.inj_bht]
                if wctrl.mode == 'rate': # rate control
                    w.control = self.physics.new_rate_inj(wctrl.rate, inj_stream, wctrl.comp_index)
                    w.constraint = self.physics.new_bhp_inj(wctrl.bhp_constraint, inj_stream)
                elif wctrl.mode == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_inj(wctrl.bhp, inj_stream)
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            elif wctrl.type == 'prod':  # PROD well
                if wctrl.mode == 'rate': # rate control
                    w.control = self.physics.new_rate_prod(wctrl.rate, wctrl.comp_index)
                    w.constraint = self.physics.new_bhp_prod(wctrl.bhp_constraint)
                elif wctrl.mode == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_prod(wctrl.bhp)
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            else:
                print('Unknown well ctrl.type', wctrl.type)
                exit(1)
            if verbose:
                print('set_well_controls: time=', time, 'well=', w.name, w.control, w.constraint)

        # check
        for w in self.reservoir.wells:
            assert w.control is not None, 'well control is not initialized for the well ' + w.name
            if verbose and w.constraint is not None and 'rate' in str(type(w.control)):
                print('A constraint for the well ' + w.name + ' is not initialized!')

    # def set_rhs_flux(self, t: float = None): #This is probably used for point injection such as FluidFlower
    #     M_CO2 = 44.01  # kg/kmol
    #     nv = self.physics.n_vars
    #     nb = self.reservoir.mesh.n_res_blocks
    #     rhs_flux = np.zeros(nb * nv)
    #     # wells
    #     enth_idx = list(self.physics.property_containers[0].output_props.keys()).index("enthV")
    #
    #     self.well_centers = np.array([self.reservoir.calc_well_index(11, 15, 20)[0]])
    #
    #     for i, well_cell in enumerate(self.well_centers): #was self.reservoir.well_cells
    #         print('well_cells = ', well_cell)
    #         # Obtain state from engine
    #         p_wellcell = self.physics.engine.X[well_cell * nv]
    #         CO2_idx = well_cell * nv + 1  # second equation
    #         temp_idx = well_cell * nv + nv - 1  # last equation
    #         state = value_vector([p_wellcell] + self.inj_stream) #need to figure out how to get my self.inj_stream
    #
    #         # calculate properties
    #         values = value_vector(np.zeros(self.physics.n_ops))
    #         self.physics.property_itor[self.op_num[well_cell]].evaluate(state, values)
    #
    #         enthV = values[enth_idx]
    #         n_CO2 = self.inj_rate[i] / M_CO2
    #         rhs_flux[CO2_idx] -= n_CO2
    #         rhs_flux[temp_idx] -= enthV * n_CO2
    #     return rhs_flux

class ModBrooksCorey:
    def __init__(self, corey, phase):

        self.phase = phase

        if self.phase == "Aq":
            self.k_rw_e = corey.krwe
            self.swc = corey.swc
            self.sgc = 0
            self.nw = corey.nw
        else:
            self.k_rg_e = corey.krge
            self.sgc = corey.sgc
            self.swc = 0
            self.ng = corey.ng

    def evaluate(self, sat):
        if self.phase == "Aq":
            Se = (sat - self.swc)/(1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rw_e * Se ** self.nw
        else:
            Se = (sat - self.sgc) / (1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rg_e * Se ** self.ng

        return k_r

class CapillaryPressure:
    def __init__(self, corey):
        self.nph = 2
        self.swc = corey.swc
        self.p_entry = corey.p_entry
        self.labda = corey.labda
        self.eps = 1e-3

    def evaluate(self, sat):
        '''
        default evaluator of capillary pressure Pc based on pow
        :param sat: saturation
        :return: Pc
        '''
        if self.nph > 1:
            Se = (sat[0] - self.swc)/(1 - self.swc) #Was Sat[1], but most likely should be sat[0]
            if Se < self.eps:
                Se = self.eps
            pc = self.p_entry * Se ** (-1/self.labda)

            Pc = np.zeros(self.nph, dtype=object)
            Pc[1] = pc #Shouldn't this be Pc[0]?
        else:
            Pc = [0.0]
        return Pc #V, Aq
