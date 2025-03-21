# encoding: utf-8
# module darts.discretizer
# from C:\Users\maartendenooijer\anaconda3\envs\thesis_old\lib\site-packages\darts\discretizer.cp39-win_amd64.pyd
# by generator 1.147
# no doc

# imports
import pybind11_builtins as __pybind11_builtins


class Discretizer(__pybind11_builtins.pybind11_object):
    # no doc
    def calc_mpfa_transmissibilities(self, arg0): # real signature unknown; restored from __doc__
        """ calc_mpfa_transmissibilities(self: darts.discretizer.Discretizer, arg0: bool) -> None """
        pass

    def calc_tpfa_transmissibilities(self, arg0, darts_discretizer_elem_loc=None, set=None, p_int=None): # real signature unknown; restored from __doc__
        """ calc_tpfa_transmissibilities(self: darts.discretizer.Discretizer, arg0: dict[darts.discretizer.elem_loc, set[int]]) -> None """
        pass

    def get_fault_xyz(self): # real signature unknown; restored from __doc__
        """ get_fault_xyz(self: darts.discretizer.Discretizer) -> darts.discretizer.value_vector """
        pass

    def get_one_way_tpfa_transmissibilities(self): # real signature unknown; restored from __doc__
        """ get_one_way_tpfa_transmissibilities(self: darts.discretizer.Discretizer) -> darts.discretizer.index_vector """
        pass

    def init(self): # real signature unknown; restored from __doc__
        """ init(self: darts.discretizer.Discretizer) -> None """
        pass

    def reconstruct_pressure_gradients_per_cell(self, arg0, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """ reconstruct_pressure_gradients_per_cell(self: darts.discretizer.Discretizer, arg0: dis::BoundaryCondition) -> None """
        pass

    def reconstruct_pressure_temperature_gradients_per_cell(self, arg0, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """ reconstruct_pressure_temperature_gradients_per_cell(self: darts.discretizer.Discretizer, arg0: dis::BoundaryCondition, arg1: dis::BoundaryCondition) -> None """
        pass

    def set_mesh(self, arg0): # real signature unknown; restored from __doc__
        """ set_mesh(self: darts.discretizer.Discretizer, arg0: darts.discretizer.Mesh) -> None """
        pass

    def set_permeability(self, arg0, arg1, arg2): # real signature unknown; restored from __doc__
        """ set_permeability(self: darts.discretizer.Discretizer, arg0: darts.discretizer.value_vector, arg1: darts.discretizer.value_vector, arg2: darts.discretizer.value_vector) -> None """
        pass

    def set_porosity(self, arg0): # real signature unknown; restored from __doc__
        """ set_porosity(self: darts.discretizer.Discretizer, arg0: darts.discretizer.value_vector) -> None """
        pass

    #NEW
    def set_satnum(self, arg0): # real signature unknown; restored from __doc__
        """ set_satnum(self: darts.discretizer.Discretizer, arg0: darts.discretizer.value_vector) -> None """
        pass

    #NEW
    def set_opnum(self, arg0): # real signature unknown; restored from __doc__
        """ set_opnum(self: darts.discretizer.Discretizer, arg0: darts.discretizer.value_vector) -> None """
        pass

    def write_tran_cube(self, arg0, arg1): # real signature unknown; restored from __doc__
        """ write_tran_cube(self: darts.discretizer.Discretizer, arg0: str, arg1: str) -> None """
        pass

    def write_tran_list(self, arg0): # real signature unknown; restored from __doc__
        """ write_tran_list(self: darts.discretizer.Discretizer, arg0: str) -> None """
        pass

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: darts.discretizer.Discretizer) -> None """
        pass

    cell_m = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    cell_p = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    fluxes_matrix = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    flux_offset = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    flux_rhs = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    flux_stencil = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    flux_vals = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    flux_vals_homo = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    flux_vals_thermal = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    grav_vec = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    heat_conductions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    perms = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    permx = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    permy = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    permz = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    poro = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    satnum = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default #NEW

    opnum = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default #NEW

    p_grads = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    t_grads = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default


    __pybind11_module_local_v5_msvc_mscver1941__ = None # (!) real value is '<capsule object NULL at 0x0000027BFCD02180>'


