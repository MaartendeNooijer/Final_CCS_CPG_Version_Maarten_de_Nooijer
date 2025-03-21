# encoding: utf-8
# module darts.discretizer
# from C:\Users\maartendenooijer\anaconda3\envs\thesis_old\lib\site-packages\darts\discretizer.cp39-win_amd64.pyd
# by generator 1.147
# no doc

# imports
import pybind11_builtins as __pybind11_builtins


class Mesh(__pybind11_builtins.pybind11_object):
    # no doc
    def calc_cell_nodes(self, arg0, arg1, arg2, arg3, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """ calc_cell_nodes(self: darts.discretizer.Mesh, arg0: int, arg1: int, arg2: int, arg3: Annotated[list[float], FixedSize(8)], arg4: Annotated[list[float], FixedSize(8)], arg5: Annotated[list[float], FixedSize(8)]) -> None """
        pass

    def calc_cell_sizes(self, arg0, arg1, arg2): # real signature unknown; restored from __doc__
        """ calc_cell_sizes(self: darts.discretizer.Mesh, arg0: int, arg1: int, arg2: int) -> Annotated[list[float], FixedSize(3)] """
        pass

    def construct_local_global(self, arg0): # real signature unknown; restored from __doc__
        """ construct_local_global(self: darts.discretizer.Mesh, arg0: darts.discretizer.index_vector) -> None """
        pass

    def cpg_cell_props(self, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9): # real signature unknown; restored from __doc__
        """ cpg_cell_props(self: darts.discretizer.Mesh, arg0: int, arg1: int, arg2: int, arg3: darts.discretizer.value_vector, arg4: darts.discretizer.value_vector, arg5: darts.discretizer.index_vector, arg6: darts.discretizer.value_vector, arg7: darts.discretizer.value_vector, arg8: int, arg9: darts.discretizer.index_vector) -> None """
        pass

    def cpg_connections(self, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, darts_discretizer_elem_loc=None, set=None, p_int=None): # real signature unknown; restored from __doc__
        """ cpg_connections(self: darts.discretizer.Mesh, arg0: int, arg1: int, arg2: darts.discretizer.value_vector, arg3: darts.discretizer.index_vector, arg4: darts.discretizer.index_vector, arg5: darts.discretizer.index_vector, arg6: darts.discretizer.index_vector, arg7: darts.discretizer.index_vector, arg8: darts.discretizer.value_vector, arg9: darts.discretizer.value_vector, arg10: darts.discretizer.value_vector, arg11: darts.discretizer.index_vector, arg12: dict[darts.discretizer.elem_loc, set[int]]) -> None """
        pass

    def cpg_elems_nodes(self, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10): # real signature unknown; restored from __doc__
        """ cpg_elems_nodes(self: darts.discretizer.Mesh, arg0: int, arg1: int, arg2: int, arg3: darts.discretizer.value_vector, arg4: darts.discretizer.index_vector, arg5: darts.discretizer.index_vector, arg6: darts.discretizer.index_vector, arg7: darts.discretizer.index_vector, arg8: darts.discretizer.index_vector, arg9: darts.discretizer.value_vector, arg10: darts.discretizer.index_vector) -> darts.discretizer.index_vector """
        pass

    def generate_adjacency_matrix(self): # real signature unknown; restored from __doc__
        """ generate_adjacency_matrix(self: darts.discretizer.Mesh) -> None """
        pass

    def get_centers(self): # real signature unknown; restored from __doc__
        """ get_centers(self: darts.discretizer.Mesh) -> darts.discretizer.value_vector """
        pass

    def get_global_index(self, arg0, arg1, arg2): # real signature unknown; restored from __doc__
        """ get_global_index(self: darts.discretizer.Mesh, arg0: int, arg1: int, arg2: int) -> int """
        return 0

    def get_ijk(self, arg0, arg1, arg2, arg3, arg4): # real signature unknown; restored from __doc__
        """ get_ijk(self: darts.discretizer.Mesh, arg0: int, arg1: int, arg2: int, arg3: int, arg4: bool) -> None """
        pass

    def get_ijk_as_str(self, arg0, arg1): # real signature unknown; restored from __doc__
        """ get_ijk_as_str(self: darts.discretizer.Mesh, arg0: int, arg1: bool) -> str """
        return ""

    def get_nodes_array(self): # real signature unknown; restored from __doc__
        """ get_nodes_array(self: darts.discretizer.Mesh) -> darts.discretizer.value_vector """
        pass

    def get_prisms(self): # real signature unknown; restored from __doc__
        """ get_prisms(self: darts.discretizer.Mesh) -> darts.discretizer.value_vector """
        pass

    def gmsh_mesh_processing(self, arg0, arg1, darts_discretizer_elem_loc=None, set=None, p_int=None): # real signature unknown; restored from __doc__
        """ gmsh_mesh_processing(self: darts.discretizer.Mesh, arg0: str, arg1: dict[darts.discretizer.elem_loc, set[int]]) -> None """
        pass

    def write_float_array_to_file(self, arg0, arg1, arg2, arg3, arg4, arg5, arg6): # real signature unknown; restored from __doc__
        """ write_float_array_to_file(self: darts.discretizer.Mesh, arg0: str, arg1: str, arg2: darts.discretizer.value_vector, arg3: darts.discretizer.index_vector, arg4: int, arg5: float, arg6: bool) -> None """
        pass

    def write_int_array_to_file(self, arg0, arg1, arg2, arg3, arg4, arg5, arg6): # real signature unknown; restored from __doc__
        """ write_int_array_to_file(self: darts.discretizer.Mesh, arg0: str, arg1: str, arg2: darts.discretizer.index_vector, arg3: darts.discretizer.index_vector, arg4: int, arg5: float, arg6: bool) -> None """
        pass

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: darts.discretizer.Mesh) -> None """
        pass

    actnum = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    op_num = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default #NEW

    adj_matrix = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    adj_matrix_cols = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    adj_matrix_offset = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    centroids = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    conns = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    conn_type_map = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    coord = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    depths = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    elems = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    elem_nodes = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    elem_nodes_sorted = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    elem_type_map = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    global_to_local = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    init_apertures = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    local_to_global = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    nodes = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    num_of_elements = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    num_of_nodes = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    nx = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    ny = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    nz = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    n_cells = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    poro = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    region_elems_num = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    region_ranges = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    tags = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    volumes = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    zcorn = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default


    __pybind11_module_local_v5_msvc_mscver1941__ = None # (!) real value is '<capsule object NULL at 0x0000027BFCD27CF0>'


