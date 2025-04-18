import os
from darts.tools.plot_darts import *
from darts.tools.logging import redirect_all_output

from model_co2_spe11b import ModelCCS #NEW
#from model_co2_spe11b_Ilshat_test_environment import ModelCCS #NEW

from darts.engines import redirect_darts_output

def run(physics_type : str, case: str, out_dir: str, export_vtk=True, redirect_log=True, platform='cpu'):
    '''
    :param physics_type: "geothermal" or "dead_oil"
    :param case: input grid name
    :param out_dir: directory name for outpult files
    :param export_vtk:
    :return:
    '''
    import os
    from darts.engines import set_num_threads

    NT = int(os.getenv("OMP_NUM_THREADS", 5))  # Default to 6 if not set
    set_num_threads(NT)

    print('Test started', 'physics_type:', physics_type, 'case:', case, 'platform=', platform)
    # base_results_dir = "results"
    # out_dir = os.path.join(base_results_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_filename = os.path.join(out_dir, 'run.log')
    if redirect_log:
        log_stream = redirect_all_output(log_filename)

    # Assuming out_dir is already defined
    log_path = os.path.join(out_dir, "run_n.log")
    redirect_darts_output(log_path)

    #Assing CCS Model
    m = ModelCCS()

    #Assign CCS Physics
    m.physics_type = physics_type

    #Load input data
    m.set_input_data(case=case)

    #Initialize reservoir and properties
    m.init_reservoir()

    m.init(output_folder=out_dir, platform=platform)
    #m.reservoir.mesh.init_grav_coef(0)
    m.save_data_to_h5(kind = 'solution')
    m.set_well_controls()

    ret = m.run_simulation(out_dir)
    if ret != 0:
        exit(1)

    m.reservoir.centers_to_vtk(out_dir)

    m.print_timers()

    if export_vtk:
        # read h5 file and write vtk
        m.reservoir.create_vtk_wells(output_directory=out_dir)
        for ith_step in range(len(m.idata.sim.time_steps)):
            m.output_to_vtk(ith_step=ith_step) #Go to this function in darts_model.py and add props_names = props_names + ['pressure', 'temperature'] after line 863 to add pressure and temperature

    if export_vtk:  # for the first, last and every tenth
        # Determine the indices for the VTK output steps
        m.reservoir.create_vtk_wells(output_directory=out_dir)
        steps_to_export = [0]  # Add the first step (0-indexed)

        # Add every fifth step in between
        for i in range(9, len(m.idata.sim.time_steps), 10):
            steps_to_export.append(i)

        steps_to_export.append(len(m.idata.sim.time_steps) - 1)  # Add the last step

        # Now generate VTK files for the selected steps
        for ith_step in steps_to_export:
            m.output_to_vtk(ith_step=ith_step)

    def add_columns_time_data(time_data):
        molar_mass_co2 = 44.01 #kg/kmol
        time_data['Time (years)'] = time_data['time'] / 365.25
        for k in time_data.keys():
            if physics_type == 'ccs' and 'V rate (m3/day)' in k:
                time_data[k.replace('V rate (m3/day)', 'V rate (kmol/day)')] = time_data[k]
                time_data[k.replace('V rate (m3/day)', 'V rate (ton/day)')] = time_data[k] * molar_mass_co2 / 1000 #tons
                time_data.drop(columns=k, inplace=True)

            if physics_type == 'ccs' and 'V  volume (m3)' in k:
                time_data[k.replace('V  volume (m3)', 'V volume (kmol)')] = time_data[k]
                time_data[k.replace('V  volume (m3)', 'V volume (Mt/year)')] = time_data[k] * molar_mass_co2 / 1000 / 1000000 #Mt per year
                time_data.drop(columns=k, inplace=True)

    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    add_columns_time_data(time_data)
    time_data.to_pickle(os.path.join(out_dir, 'time_data.pkl'))

    time_data_report = pd.DataFrame.from_dict(m.physics.engine.time_data_report)
    add_columns_time_data(time_data_report)
    time_data_report.to_pickle(os.path.join(out_dir, 'time_data_report.pkl'))

    writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data.xlsx'))
    time_data.to_excel(writer, sheet_name='time_data')
    writer.close()

    # filter time_data_report and write to xlsx
    # list the column names that should be removed
    press_gridcells = time_data_report.filter(like='reservoir').columns.tolist()
    chem_cols = time_data_report.filter(like='Kmol').columns.tolist()
    #remove columns from data
    time_data_report.drop(columns=press_gridcells + chem_cols, inplace=True)
    # add time in years
    time_data_report['Time (years)'] = time_data_report['time'] / 365.25
    writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data_report.xlsx'))
    time_data_report.to_excel(writer, sheet_name='time_data_report')
    writer.close()

    return  time_data, time_data_report, m.idata.well_data.wells.keys(), m.well_is_inj #NEW, originalaly also failed, sim_time included

##########################################################################################################
def plot_results(wells, well_is_inj, time_data_list, time_data_report_list, label_list, physics_type, out_dir):
    plt.rc('font', size=12)

    def plot_total_inj_gas_rate_darts_volume(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
        acc_df = pd.DataFrame()
        acc_df['time'] = darts_df['time']
        acc_df['total'] = 0
        search_str = ' : V rate (ton/day)'
        for col in darts_df.columns:
            if search_str in col:
                # if sum(darts_df[col]) > 0:
                if 'I' in col:
                    #     acc_df['total'] += darts_df[col]
                    for i in range(0, len(darts_df[col])):
                        if darts_df[col][i] >= 0:
                            # acc_df['total'][i] += darts_df[col][i]
                            acc_df.loc[i, 'total'] += darts_df[col][i]

        ax = acc_df.plot(x='time', y='total', style=style, color=color,
                         ax=ax, alpha=alpha)

        ymin, ymax = ax.get_ylim()
        if ymax < 0:
            ax.set_ylim(ymin * 1.1, 0)

        if ymin > 0:
            ax.set_ylim(0, ymax * 1.1)

        return ax

    for well_name in wells:

        ax = None
        for time_data, label in zip(time_data_list, label_list):
            ax = plot_total_inj_gas_rate_darts_volume(time_data, ax=ax)  # , label=label) #NEW was ax = plot_total_inj_gas_rate_darts(time_data, ax=ax)
        ax.set(xlabel="Days", ylabel="Total Inj Gas Rate [Ton/Day]")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'total_inj_gas_rate_' + well_name + '_' + case + '.png'))
        plt.close()

    for well_name in wells:
        ax = None
        for time_data_report, label in zip(time_data_list, label_list): #NEW was time_data_report_list to check per year
            ax = plot_bhp_darts(well_name, time_data_report, ax=ax)#, label=label)
        ax.set(xlabel="Days", ylabel="BHP [bar]")
        plt.savefig(os.path.join(out_dir, 'well_' + well_name + '_bhp_' + case + '.png'))
        plt.tight_layout()
        plt.close()

##########################################################################################################
# for CI/CD
def check_performance_local(m, case, physics_type):
    import platform

    os.makedirs('ref', exist_ok=True)

    pkl_suffix = ''
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
        pkl_suffix = '_gpu'
    elif os.getenv('ODLS') != None and os.getenv('ODLS') == '-a':
        pkl_suffix = '_iter'
    else:
        pkl_suffix = '_odls'
    print('pkl_suffix=', pkl_suffix)

    file_name = os.path.join('ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix +
                             '_' + case + '_' + physics_type + '.pkl')
    overwrite = 0
    if os.getenv('UPLOAD_PKL') == '1':
        overwrite = 1

    is_plk_exist = os.path.isfile(file_name)

    failed = m.check_performance(perf_file=file_name, overwrite=overwrite, pkl_suffix=pkl_suffix)

    if not is_plk_exist or overwrite == '1':
        m.save_performance_data(file_name=file_name, pkl_suffix=pkl_suffix)
        return False, 0.0

    if is_plk_exist:
        return (failed > 0), -1.0 #data[-1]['simulation time']
    else:
        return False, -1.0

def run_test(args: list = [], platform='cpu'):
    if len(args) > 1:
        case = args[0]
        physics_type = args[1]

        out_dir = 'results_' + physics_type + '_' + case
        ret = run(case=case, physics_type=physics_type, out_dir=out_dir, platform=platform)
        return ret[0], ret[1] #failed_flag, sim_time
    else:
        print('Not enough arguments provided')
        return True, 0.0
##########################################################################################################

if __name__ == '__main__':
    platform = 'cpu'
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
            platform = 'gpu'

    physics_list = []
    #physics_list += ['geothermal']
    #physics_list += ['deadoil']
    physics_list += ['ccs']

    cases_list = []

    #cases_list += ['grid_CCS_maarten'] #Heterogeneous, contains different regions
    cases_list += ['grid_CCS_maarten_homogeneous'] #Homogeneous, one region assignment

    #cases_list += ["case_1_50x50x40"]
    #cases_list += ["case_1_100x100x80"]
    # cases_list += ["case_1_125x125x80"]
    # cases_list += ["case_1_250x250x80"]
    #cases_list += ["case_2_50x50x40"]
    # cases_list += ["case_2_100x100x80"]
    # cases_list += ["case_2_125x125x80"]
    #cases_list += ["case_2_250x250x80"]
    #cases_list += ["case_3_50x50x40"]
    #cases_list += ["case_3_100x100x80"]
    #cases_list += ["case_3_125x125x80"]
    #cases_list += ["case_3_250x250x80"]

    #Continue here!!
    #Change to rate, to see if that works better instead of injecting 15 mt
    #Checking if it works if I adjust the property container with only wells, and one sand facies properties region

    #cases_list += ['case_40_actnum']

    #cases_list += ['grid_CCS_maarten']
    #cases_list += ['grid_CCS_maarten_homogeneous']
    #cases_list += ['30x30x40_dxdydz_5m'] #C:\Users\maartendenooijer\PycharmProjects\Old_CCS_CPG_Version\meshes\

    well_controls = []
    #well_controls += ['wrate']
    well_controls += ['wbhp'] #slower, but this is the one we are going to use

    for physics_type in physics_list:
        for case_geom in cases_list:
            for wctrl in well_controls:
                if physics_type == 'deadoil' and wctrl == 'wrate':
                    continue
                case = case_geom + '_' + wctrl
                out_dir = 'results_' + physics_type + '_' + case
                base_results_dir = "../results"
                out_dir = os.path.join(base_results_dir, out_dir)
                time_data, time_data_report, wells, well_is_inj = run(physics_type=physics_type, #NEW, originally failed, sim_time, were included
                                                                                        case=case, out_dir=out_dir,
                                                                                        redirect_log=False,
                                                                                        export_vtk= True,
                                                                                        platform=platform)

                # one can read well results from pkl file to add/change well plots without re-running the model
                pkl1_dir = '..'
                pkl_fname = 'time_data.pkl'
                pkl_report_fname = 'time_data_report.pkl'
                time_data_list = [time_data]
                time_data_report_list = [time_data_report]
                label_list = [None]

                # compare the current results with another run
                #pkl1_dir = r'../../../open-darts_dev/models/cpg_sloping_fault/results_' + physics_type + '_' + case_geom
                #time_data_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_fname))
                #time_data_report_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_report_fname))
                #time_data_list = [time_data_1, time_data]
                #time_data_report_list = [time_data_report_1, time_data_report]
                #label_list = ['1', 'current']

                plot_results(wells=wells, well_is_inj=well_is_inj,
                             time_data_list=time_data_list, time_data_report_list=time_data_report_list, label_list=label_list,
                             physics_type=physics_type, out_dir=out_dir)
