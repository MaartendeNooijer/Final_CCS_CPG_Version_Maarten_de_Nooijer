import os
from darts.tools.plot_darts import *
from darts.tools.logging import redirect_all_output

from model_co2_spe11b import ModelCCS #NEW
#from model_co2_spe11b_Ilshat_test_environment import ModelCCS #NEW

from darts.engines import redirect_darts_output

def run(physics_type : str, case: str, out_dir: str, export_vtk=True, redirect_log=True, platform='cpu'):
    import os
    from darts.engines import set_num_threads

    NT = int(os.getenv("OMP_NUM_THREADS", 5))  # Default to 6 if not set
    set_num_threads(NT)

    print('Test started', 'physics_type:', physics_type, 'case:', case, 'platform=', platform)
    os.makedirs(out_dir, exist_ok=True)
    log_filename = os.path.join(out_dir, 'run.log')
    if redirect_log:
        log_stream = redirect_all_output(log_filename)

    log_path = os.path.join(out_dir, "run_n.log")
    redirect_darts_output(log_path)

    m = ModelCCS()
    m.physics_type = physics_type
    m.set_input_data(case=case)
    m.init_reservoir()
    m.init(output_folder=out_dir, platform=platform)
    m.save_data_to_h5(kind = 'solution')
    m.set_well_controls()

    ret = m.run_simulation(out_dir)
    if ret != 0:
        exit(1)

    m.reservoir.centers_to_vtk(out_dir)
    m.print_timers()

    if export_vtk:
        m.reservoir.create_vtk_wells(output_directory=out_dir)
        for ith_step in range(len(m.idata.sim.time_steps)):
            m.output_to_vtk(ith_step=ith_step)

    if export_vtk:
        m.reservoir.create_vtk_wells(output_directory=out_dir)
        steps_to_export = [0]
        for i in range(9, len(m.idata.sim.time_steps), 10):
            steps_to_export.append(i)
        steps_to_export.append(len(m.idata.sim.time_steps) - 1)
        for ith_step in steps_to_export:
            m.output_to_vtk(ith_step=ith_step)

    def add_columns_time_data(time_data):
        molar_mass_co2 = 44.01 #kg/kmol
        time_data['Time (years)'] = time_data['time'] / 365.25
        for k in time_data.keys():
            if physics_type == 'ccs' and 'V rate (m3/day)' in k:
                time_data[k.replace('V rate (m3/day)', 'V rate (kmol/day)')] = time_data[k]
                time_data[k.replace('V rate (m3/day)', 'V rate (ton/day)')] = time_data[k] * molar_mass_co2 / 1000
                time_data.drop(columns=k, inplace=True)
            if physics_type == 'ccs' and 'V  volume (m3)' in k:
                time_data[k.replace('V  volume (m3)', 'V volume (kmol)')] = time_data[k]
                time_data[k.replace('V  volume (m3)', 'V volume (Mt/year)')] = time_data[k] * molar_mass_co2 / 1000 / 1e6
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

    press_gridcells = time_data_report.filter(like='reservoir').columns.tolist()
    chem_cols = time_data_report.filter(like='Kmol').columns.tolist()
    time_data_report.drop(columns=press_gridcells + chem_cols, inplace=True)
    time_data_report['Time (years)'] = time_data_report['time'] / 365.25
    writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data_report.xlsx'))
    time_data_report.to_excel(writer, sheet_name='time_data_report')
    writer.close()

    # Delete solution.h5 if it exists
    if os.path.exists(solution_h5_path):
        os.remove(solution_h5_path)
        print(f"Deleted {solution_h5_path}")

    # Delete well_data.h5 if it exists
    if os.path.exists(well_data_h5_path):
        os.remove(well_data_h5_path)
        print(f"Deleted {well_data_h5_path}")

    extract_elapsed_time(log_path, os.path.join(out_dir, "elapsed_time.txt"))

    return  time_data, time_data_report, m.idata.well_data.wells.keys(), m.well_is_inj

def plot_results(wells, well_is_inj, time_data_list, time_data_report_list, label_list, physics_type, out_dir):
    plt.rc('font', size=12)

    def plot_total_inj_gas_rate_darts_volume(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
        acc_df = pd.DataFrame()
        acc_df['time'] = darts_df['time']
        acc_df['total'] = 0
        search_str = ' : V rate (ton/day)'
        for col in darts_df.columns:
            if search_str in col and 'I' in col:
                for i in range(len(darts_df[col])):
                    if darts_df[col][i] >= 0:
                        acc_df.loc[i, 'total'] += darts_df[col][i]

        ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)
        ymin, ymax = ax.get_ylim()
        if ymax < 0:
            ax.set_ylim(ymin * 1.1, 0)
        if ymin > 0:
            ax.set_ylim(0, ymax * 1.1)
        return ax

    for well_name in wells:
        ax = None
        for time_data, label in zip(time_data_list, label_list):
            ax = plot_total_inj_gas_rate_darts_volume(time_data, ax=ax)
        ax.set(xlabel="Days", ylabel="Total Inj Gas Rate [Ton/Day]")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'total_inj_gas_rate_' + well_name + '_' + case + '.png'))
        plt.close()

    for well_name in wells:
        ax = None
        for time_data_report, label in zip(time_data_list, label_list):
            ax = plot_bhp_darts(well_name, time_data_report, ax=ax)
        ax.set(xlabel="Days", ylabel="BHP [bar]")
        plt.savefig(os.path.join(out_dir, 'well_' + well_name + '_bhp_' + case + '.png'))
        plt.tight_layout()
        plt.close()

def check_performance_local(m, case, physics_type):
    import platform
    os.makedirs('ref', exist_ok=True)
    pkl_suffix = ''
    if os.getenv('TEST_GPU') == '1':
        pkl_suffix = '_gpu'
    elif os.getenv('ODLS') == '-a':
        pkl_suffix = '_iter'
    else:
        pkl_suffix = '_odls'

    print('pkl_suffix=', pkl_suffix)
    file_name = os.path.join('ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix + '_' + case + '_' + physics_type + '.pkl')
    overwrite = 1 if os.getenv('UPLOAD_PKL') == '1' else 0
    is_plk_exist = os.path.isfile(file_name)
    failed = m.check_performance(perf_file=file_name, overwrite=overwrite, pkl_suffix=pkl_suffix)

    if not is_plk_exist or overwrite == 1:
        m.save_performance_data(file_name=file_name, pkl_suffix=pkl_suffix)
        return False, 0.0
    if is_plk_exist:
        return (failed > 0), -1.0

def run_test(args: list = [], platform='cpu', run_tag=""):
    if len(args) > 1:
        case = args[0]
        physics_type = args[1]
        out_dir = 'results_' + physics_type + '_' + case + run_tag
        ret = run(case=case, physics_type=physics_type, out_dir=out_dir, platform=platform)
        return ret[0], ret[1]
    else:
        print('Not enough arguments provided')
        return True, 0.0

if __name__ == '__main__':
    platform = 'cpu'
    if os.getenv('TEST_GPU') == '1':
        platform = 'gpu'

    physics_list = ['ccs']

    cases_list = []
    #cases_list += ['grid_CCS_maarten'] #Heterogeneous, contains different regions
    cases_list += ['grid_CCS_maarten_homogeneous'] #Homogeneous, one region assignment

    well_controls = ['wbhp']

    run_tag = "_v2"  # <-- Change this to make each run unique

    for physics_type in physics_list:
        for case_geom in cases_list:
            for wctrl in well_controls:
                if physics_type == 'deadoil' and wctrl == 'wrate':
                    continue
                case = case_geom + '_' + wctrl
                out_dir = 'results_' + physics_type + '_' + case + run_tag
                base_results_dir = "../results"
                out_dir = os.path.join(base_results_dir, out_dir)

                time_data, time_data_report, wells, well_is_inj = run(
                    physics_type=physics_type,
                    case=case,
                    out_dir=out_dir,
                    redirect_log=False,
                    export_vtk=True,
                    platform=platform)

                pkl1_dir = '..'
                pkl_fname = 'time_data.pkl'
                pkl_report_fname = 'time_data_report.pkl'
                time_data_list = [time_data]
                time_data_report_list = [time_data_report]
                label_list = [None]

                plot_results(
                    wells=wells,
                    well_is_inj=well_is_inj,
                    time_data_list=time_data_list,
                    time_data_report_list=time_data_report_list,
                    label_list=label_list,
                    physics_type=physics_type,
                    out_dir=out_dir)


def extract_elapsed_time(log_path, output_path):
    """
    Extracts the last occurrence of ELAPSED time from a log file and writes it to a text file.

    Parameters:
        log_path (str): Path to the log file to parse.
        output_path (str): Path to the text file to write the elapsed time.
    """
    elapsed_time = None

    try:
        with open(log_path, "r") as file:
            lines = file.readlines()

        # Find last line with 'ELAPSED'
        for line in reversed(lines):
            if "ELAPSED" in line:
                parts = line.split("ELAPSED")
                if len(parts) > 1:
                    time_part = parts[1].strip().strip("()")
                    if time_part.count(":") == 2:
                        elapsed_time = time_part
                        break

        # Save the result
        if elapsed_time:
            with open(output_path, "w") as out:
                out.write(elapsed_time)
            print(f"Elapsed time '{elapsed_time}' written to {output_path}")
        else:
            print("No elapsed time found in the log file.")

    except Exception as e:
        print(f"Error: {e}")
