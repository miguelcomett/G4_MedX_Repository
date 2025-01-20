# Python modules: 
import os, sys, time, subprocess, shutil, platform, numpy as np, pandas as pd, matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr; from pathlib import Path

# 0.1. ========================================================================================================================================================

def Install_Libraries():

    libraries = {
        "numpy"           : None,
        "matplotlib"      : None,
        "dask"            : "2024.10.0",  
        "tqdm"            : None,
        "send2trash"      : None,
        "pygame"          : None,
        "ipywidgets"      : None,
        "uproot"          : None,
        "tqdm"            : None,
        "plotly"          : None,
        "scipy"           : None,
        "pydicom"         : None,
        "PIL"             : None,
        "scikit-image"    : None,
    }

    def install_and_import(package, version=None):

        try:
            
            if package in sys.modules: return  
            if version: __import__(package)
            else: __import__(package)
        
        except ImportError: 
            
            print(f"Installing {package}...")
            try:
                if version: subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
                else: subprocess.check_call([sys.executable, "-m", "pip", "install", package])
           
            except Exception as e: print(f"Error installing {package}: {e}")
            else: print(f"{package} installed successfully.")
        
        except Exception as e: print(f"Unexpected error with {package}: {e}")

    for lib, version in libraries.items(): install_and_import(lib, version)

    print("All libraries are installed and ready to use.")

# 0.2. ========================================================================================================================================================

def PlayAlarm():

    import pygame

    alarm_path = 'Alarm.mp3'

    volume_level = 30
    os.system(f"osascript -e 'set volume output volume {volume_level}'")

    pygame.mixer.init()
    pygame.mixer.music.load(alarm_path)

    # print("Script completed. Playing alarm...")
    pygame.mixer.music.play(loops = -1) 

    time.sleep(8)
    # input("Press Enter to stop the alarm...")
    pygame.mixer.music.stop()

# 1.1. ========================================================================================================================================================

def Trash_Folder(trash_folder):

    from send2trash import send2trash
                 
    try: send2trash(trash_folder)
    except Exception as e: print(f"Error deleting trash folder: {e}")


def Simulation_Setup():
        
    from send2trash import send2trash
    
    mac_filename = 'radiography.mac'
    
    if platform.system() == "Darwin":
        directory = Path('BUILD')
        #  directory = 'BUILD/'
        executable_file = "Sim"
        run_sim = f"./{executable_file} {mac_filename} . . ."

    elif platform.system() == "Windows":
        directory = Path('build') / 'Release'
        # directory = 'build\\Release\\'
        executable_file = "Sim.exe"
        run_sim = fr".\{executable_file} .\{mac_filename} . . ."
        print(run_sim)

    elif platform.system() == "Linux":
        directory = Path('build') / 'Release'
        # directory = 'build\\Release\\'
        executable_file = "Sim.exe"
        run_sim = fr"./{executable_file} {mac_filename} . . ."
        print(run_sim)

    else: raise EnvironmentError("Unsupported operating system")

    root_folder  = directory / "ROOT/"
    mac_filepath = directory / mac_filename
    
    rad_folder = directory / "ROOT/" / "Rad_temp/"
    try: send2trash(rad_folder)
    except: pass
    os.makedirs(rad_folder, exist_ok = True)

    return directory, run_sim, root_folder, mac_filepath, rad_folder


def Run_Calibration(directory, run_sim):
    
    start_time = time.perf_counter()
    try: subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
    except Exception as e: print(f"Error running simulation: {e}")
    end_time = time.perf_counter()
    calibration_time = end_time - start_time
    print("Calibration run completed.")

    return calibration_time


def Generate_MAC_Template(
    simulation_mode,              # Obligarory parameter: 'single (1)' or 'DEXA (2)'
    threads              = None,  # Optional parameter
    spectra_mode         = None,  # Optional parameter:   'mono (1)'   or 'poly (2)'
    detector_parameters  = None,  # Optional parameter
    gun_parameters       = None,  # Optional parameter
):

    if spectra_mode        is None: spectra_mode = 'mono'
    if detector_parameters is None: detector_parameters = {'nColumns': 1, 'nRows': 1}
    if gun_parameters      is None: gun_parameters = {'X': 0, 'Y': 0, 'gaussX': 'true', 'SpanX': 230, 'SpanY': 240}

    mac_template = []

    if threads: mac_template.append(f"/run/numberOfThreads {'{Threads}'}")
    mac_template.append("/run/initialize")

    mac_template.extend([
        f"/myDetector/nColumns {detector_parameters['nColumns']}",
        f"/myDetector/nRows {detector_parameters['nRows']}"
    ])

    mac_template.extend([
        f"/Pgun/X {gun_parameters['X']} mm",
        f"/Pgun/Y {gun_parameters['Y']} mm",
        f"/Pgun/gaussX {gun_parameters['gaussX']}",
        f"/Pgun/SpanX {gun_parameters['SpanX']} mm",
        f"/Pgun/SpanY {gun_parameters['SpanY']} mm"
    ])

    if simulation_mode == 'single' or simulation_mode == 0:

        if spectra_mode == 'mono' or spectra_mode == 0:
            mac_template.extend([
                f"/gun/energy {'{Energy}'} keV",
                f"/run/beamOn {'{Beams}'}"
            ])

        if spectra_mode == '80kvp' or spectra_mode == 1:
            mac_template.extend([
                f"/Pgun/Mode 1",
                f"/run/beamOn {'{Beams}'}"
            ])

        if spectra_mode == '140kvp' or spectra_mode == 2:
            mac_template.extend([
                f"/Pgun/Mode 2",
                f"/run/beamOn {'{Beams}'}"
            ])

    if simulation_mode == 'DEXA' or simulation_mode == 1:
        
        if spectra_mode == 'mono' or spectra_mode == 0:
            mac_template.extend([
                f"/gun/energy 40 keV",
                f"/run/beamOn {'{Beams40}'}",

                f"/gun/energy 80 keV",
                f"/run/beamOn {'{Beams80}'}",
            ])

        if spectra_mode == 'poly' or spectra_mode == 1:
            mac_template.extend([
                f"/Pgun/Mode 1",
                f"/run/beamOn {'{Beams40}'}",

                f"/Pgun/Mode 2",
                f"/run/beamOn {'{Beams80}'}",
            ])

    return "\n".join(mac_template)  


def RunRadiography(threads, energy, sim_time, iteration_time, spectra_mode, detector_parameters, gun_parameters, alarm):

    from tqdm.notebook import tqdm

    if iteration_time == 0 or iteration_time > sim_time: iteration_time = sim_time

    sim_time = sim_time * 60 # s
    iteration_time = iteration_time * 60 # s 

    energy_name = f"{str(energy)}{'kev'}"

    directory, run_sim, root_folder, mac_filepath, rad_folder = Simulation_Setup()

    simulation_mode = 'single'
    mac_template = Generate_MAC_Template(simulation_mode, threads, spectra_mode, detector_parameters, gun_parameters)
    
    Beams_calibration = 2500000

    filled_template = mac_template.format(Threads = threads, Energy = energy, Beams = Beams_calibration)
    with open(mac_filepath, 'w') as template_file: template_file.write(filled_template)
    
    calibration_time = Run_Calibration(directory, run_sim)

    root_base_name= 'CT'

    if spectra_mode == 'mono' or spectra_mode == 0: 
        new_base_name = 'Rad'

    if spectra_mode == '80kvp' or spectra_mode == 1: 
        new_base_name = 'Poly'
        energy_name = spectra_mode
    if spectra_mode == '140kvp' or spectra_mode == 2: 
        new_base_name = 'Poly'
        energy_name = spectra_mode

    old_root_name = root_folder/f"{root_base_name}{'_00.root'}"
    new_root_name = root_folder/f"{new_base_name}{'_0.root'}"

    try: os.rename(old_root_name, new_root_name)
    except FileNotFoundError: print("The file does not exist.")
    except PermissionError: print("You do not have permission to rename this file.")

    if os.path.exists(new_root_name): shutil.move(new_root_name, rad_folder)

    iterations = int(sim_time / iteration_time)
    
    Beams = int((sim_time * Beams_calibration) / (calibration_time * iterations))
    print('Beams to simulate:', round(Beams * iterations / 1000000, 2), 'M')

    filled_template = mac_template.format(Threads = threads, Energy = energy, Beams = Beams)
    with open(mac_filepath, 'w') as template_file: template_file.write(filled_template)

    exit_requested = False

    for iteration in tqdm(range(iterations), desc = "Running Simulations", unit = " Iterations", leave = True):
        
        try: 
            subprocess.run(run_sim, cwd = directory,check = True, shell = True, stdout = subprocess.DEVNULL)
    
            new_root_name = root_folder / f"{new_base_name}{'_'}{str(iteration + 1)}{'.root'}"
            try: os.rename(old_root_name, new_root_name)
            except FileNotFoundError: print("The file does not exist.")
            except PermissionError: print("You do not have permission to rename this file.")
            if os.path.exists(new_root_name): shutil.move(new_root_name, rad_folder)

            if exit_requested: break
        
        except subprocess.CalledProcessError as e: print(f"Error al ejecutar la simulación: {e}")
        except KeyboardInterrupt: 
            if not exit_requested: print("\nKeyboardInterrupt detected! Exiting after this iteration."); exit_requested = True
            else: print("Forcing immediate termination."); raise

    total_beams = int(np.ceil(Beams * iterations / 1000000))
    merged_name = f"{new_base_name}{'_'}{energy_name}{'_'}{str(total_beams)}{'M'}"

    if os.path.exists(root_folder / f"{merged_name}{'.root'}"):
        counter = 1
        while os.path.exists(root_folder / f"{merged_name}{'_'}{str(counter)}{'.root'}"): counter = counter + 1
        merged_name = f"{merged_name}{'_'}{str(counter)}"
    merged_name = f"{merged_name}{'.root'}"

    with open(os.devnull, "w") as fnull: 
        with redirect_stdout(fnull), redirect_stderr(fnull):
            Merge_Roots_HADD(rad_folder, new_base_name, merged_name, trim_coords = None)

    shutil.move(rad_folder/merged_name, root_folder)
    Trash_Folder(rad_folder)

    print('-> Simulation completed. Files:', merged_name, 'written in', root_folder)
    if alarm == True or alarm == 1: PlayAlarm()

# 1.2. ========================================================================================================================================================

def Rename_and_Move(root_folder, rad_folder, iteration, spectra_mode):

    if spectra_mode == 'mono' or spectra_mode == 0:
        base_name_40 = 'Rad_40kev'
        base_name_80 = 'Rad_80kev'
    if spectra_mode == 'poly' or spectra_mode == 1:
        base_name_40 = 'Poly_80kvp'
        base_name_80 = 'Poly_140kvp'
    
    old_root_name = 'CT'

    file_40 = root_folder / f"{old_root_name}{'_00.root'}"
    file_80 = root_folder / f"{old_root_name}{'_01.root'}"
    new_name_40 = root_folder / f"{base_name_40}{'_'}{str(iteration)}{'.root'}"
    new_name_80 = root_folder / f"{base_name_80}{'_'}{str(iteration)}{'.root'}"

    try: os.rename(file_40, new_name_40)
    except FileNotFoundError: 
        print("The file does not exist.")
        sys.exit()
    except PermissionError: 
        print("You do not have permission to rename this file.")
        sys.exit()

    try: os.rename(file_80, new_name_80)
    except FileNotFoundError: 
        print("The file does not exist.")
        sys.exit()
    except PermissionError: 
        print("You do not have permission to rename this file.")
        sys.exit()

    if os.path.exists(new_name_40): shutil.move(new_name_40, rad_folder)
    if os.path.exists(new_name_80): shutil.move(new_name_80, rad_folder)

    return base_name_40, base_name_80
    

def RunDEXA(threads, sim_time, iteration_time, spectra_mode, detector_parameters, gun_parameters, alarm):

    from tqdm.notebook import tqdm

    if iteration_time == 0 or iteration_time > sim_time: iteration_time = sim_time

    sim_time = sim_time * 60 # s
    iteration_time = iteration_time * 60 # s 

    directory, run_sim, root_folder, mac_filepath, rad_folder = Simulation_Setup()

    simulation_mode = 'DEXA'
    mac_template = Generate_MAC_Template(simulation_mode, threads, spectra_mode, detector_parameters, gun_parameters)

    if spectra_mode == 'mono' or spectra_mode == 0:
        Beams40_calibration = 2000000
        Beams80_calibration = int(Beams40_calibration / 1.61)

    if spectra_mode == 'poly' or spectra_mode == 1:
        Beams40_calibration = 2000000
        Beams80_calibration = int(Beams40_calibration * 1.05)

    filled_template = mac_template.format(Threads = threads, Beams40 = Beams40_calibration, Beams80 = Beams80_calibration)
    with open(mac_filepath, 'w') as template_file: template_file.write(filled_template)

    calibration_time = Run_Calibration(directory, run_sim)
    base_name_40, base_name_80  = Rename_and_Move(root_folder, rad_folder, 0, spectra_mode)
    iterations = int(sim_time / iteration_time)

    Beams40 = int((sim_time * Beams40_calibration) / (calibration_time * iterations))
    Beams80 = int((sim_time * Beams80_calibration) / (calibration_time * iterations))

    print('Beams to simulate:', round(Beams40 * iterations / 1000000, 2), 'M', round(Beams80 * iterations / 1000000, 2), 'M')

    filled_template = mac_template.format(Threads = threads, Beams40 = Beams40, Beams80 = Beams80)
    with open(mac_filepath, 'w') as template_file: template_file.write(filled_template)

    exit_requested = False
    
    for iteration in tqdm(range(iterations), desc = "Running Simulations", unit = " Iterations", leave = True):

        try: 
            subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
            Rename_and_Move(root_folder, rad_folder, iteration + 1, spectra_mode)
            if exit_requested: break
        
        except subprocess.CalledProcessError as e: print(f"Error al ejecutar la simulación: {e}")
        except KeyboardInterrupt: 
            if not exit_requested: print("\nKeyboardInterrupt detected! Exiting after this iteration."); exit_requested = True
            else: print("Forcing immediate termination."); raise

    total_beams_40 = int(np.ceil(Beams40 * iterations / 1_000_000))
    total_beams_80 = int(np.ceil(Beams80 * iterations / 1_000_000))

    merged_40 = f"{base_name_40}{'_'}{str(total_beams_40)}{'M'}"
    merged_80 = f"{base_name_80}{'_'}{str(total_beams_80)}{'M'}"

    if os.path.exists(root_folder / f"{merged_40}{'.root'}"):
        counter = 1
        while os.path.exists(root_folder / f"{merged_40}{'_'}{str(counter)}{'.root'}"): counter = counter + 1
        merged_40 = f"{merged_40}{'_'}{str(counter)}"
    merged_40 = f"{merged_40}{'.root'}"

    if os.path.exists(root_folder / f"{merged_80}{'.root'}"):
        counter = 1
        while os.path.exists(root_folder / f"{merged_80}{'_'}{str(counter)}{'.root'}"): counter = counter + 1
        merged_80 = f"{merged_80}{'_'}{str(counter)}"
    merged_80 = f"{merged_80}{'.root'}"

    fnull = open(os.devnull, "w")
    with redirect_stdout(fnull), redirect_stderr(fnull): Merge_Roots_HADD(rad_folder, base_name_40, merged_40, trim_coords = None)
    with redirect_stdout(fnull), redirect_stderr(fnull): Merge_Roots_HADD(rad_folder, base_name_80, merged_80, trim_coords = None)
    fnull.close()

    shutil.move(rad_folder/merged_40, root_folder)
    shutil.move(rad_folder/merged_80, root_folder)

    Trash_Folder(rad_folder)

    print('Files:', merged_40, 'and', merged_80, 'written in', root_folder)
    print("Simulation completed.")
    if alarm == True or alarm == 1: PlayAlarm()


def UI_RunDEXA():

    import ipywidgets as widgets; from IPython.display import display, HTML

    style = """
    <style>
    .widget-label {font-size: 18px !important;}
    .widget-button {font-size: 18px !important;}
    .widget-dropdown > select {font-size: 16px !important;}
    .widget-text {font-size: 18px !important;}
    </style>
    """
    display(HTML(style))

    labels_width = '200px'
    custom_layout   = widgets.Layout(width = '350px')

    threads_slider  = widgets.Dropdown(
                        options = [('None', None), ('4', 4), ('9', 9), ('10', 10)],
                        value = 10,
                        description = 'Number of CPU Cores',
                        layout = custom_layout,
                        style = {'description_width': labels_width})
    
    sim_time_slider = widgets.BoundedFloatText(
                        value = 30, min = 0, max = 100000, step = 1, 
                        description = 'Simulation Time (min)', 
                        layout = custom_layout, 
                        style = {'description_width': labels_width})
    
    iteration_time  = widgets.BoundedFloatText(
                        value = 30, min = 0, max = 300, step = 1, 
                        description = 'Iteration Time (min)',
                        layout = custom_layout,
                        style = {'description_width': labels_width})
    
    spectra_mode   = widgets.Dropdown(
                        options = [('Mono', 'mono'), ('Poly', 'poly')],
                        value = 'poly',
                        description = 'Spectra Mode',
                        layout = custom_layout,
                        style = {'description_width': labels_width})
    
    alarm_toggle    = widgets.ToggleButton(
                        value = True, 
                        description = 'Alarm', 
                        button_style = 'success',
                        layout = widgets.Layout(width='350px', height='30px'))

    run_button      = widgets.Button(
                        description = 'Run Simulation', 
                        button_style = 'primary', 
                        layout = widgets.Layout(width='350px', height='50px'))
    
    output = widgets.Output()

    def toggle_alarm_state(change):
        alarm_toggle.description  = 'Alarm On' if change.new else 'Alarm Off'
        alarm_toggle.button_style = 'success' if change.new else 'danger'
    
    alarm_toggle.observe(toggle_alarm_state, 'value')
    
    def on_run_clicked(change):
        with output:
            output.clear_output()
            print('Simulation Started')
            RunDEXA(threads         = threads_slider.value,
                    sim_time        = sim_time_slider.value,
                    iteration_time  = iteration_time.value,
                    spectra_mode    = spectra_mode.value,
                    detector_parameters = None, gun_parameters = None,
                    alarm           = alarm_toggle.value)

    run_button.on_click(on_run_clicked)

    ui = widgets.VBox([
            widgets.HTML(value="<h3 style = 'color:blue; font-size: 20px;'> DEXA Simulation Parameters </h3>"),
            threads_slider, sim_time_slider, iteration_time, spectra_mode, alarm_toggle, run_button, output,])
    
    display(ui)

# 1.3. ========================================================================================================================================================

def Merge_Roots_HADD(directory, starts_with, output_name, trim_coords):
    
    if trim_coords == None: 

        trash_folder, file_list, merged_file = Manage_Files(directory, starts_with, output_name)
        hadd_command = ["hadd", '-f', merged_file] + file_list

        try:
            with open(os.devnull, 'wb') as devnull: success = subprocess.run(hadd_command, stdout = devnull, stderr = devnull, check = True)
            Trash_Folder(trash_folder)
            success = success.returncode
            if success == 0: print(f"Merged data written to: {merged_file}")
        
        except subprocess.CalledProcessError as e:
            print(f"Error: The merge process failed with return code {e.returncode}.")
            print(f"Command: {' '.join(hadd_command)}")
            print("Retriying with Merge_Roots_Dask Function")
            if 'directory' in locals() and 'starts_with' in locals() and 'output_name' in locals() and 'trim_coords' in locals():
                Merge_Roots_Dask(directory, starts_with, output_name, trim_coords)
            else: print("Error: One or more arguments for Merge_Roots_Dask are missing or undefined.")
        
        except FileNotFoundError:
            print("Error: 'hadd' command not found. Make sure ROOT is installed and configured.")

    if trim_coords: 

        print("Using Merge_Roots_Dask Function")
        Merge_Roots_Dask(directory, starts_with, output_name, trim_coords)


def Merge_Roots_Dask(directory, starts_with, output_name, trim_coords):

    import uproot, dask.array as da

    trash_folder, file_list, merged_file = Manage_Files(directory, starts_with, output_name)

    merged_trees_data = {}

    for file_path in file_list:

        opened_file = uproot.open(file_path)

        for tree_name in opened_file.keys():

            if not tree_name.endswith(";1"): continue

            tree_key = tree_name.rstrip(";1")
            tree = uproot.dask(opened_file[tree_key], library="np")

            if tree_key not in merged_trees_data: merged_trees_data[tree_key] = {}

            branches = tree.keys()
            for branch_name in branches:

                branch_data = tree[branch_name]
                if branch_name not in merged_trees_data[tree_key]: merged_trees_data[tree_key][branch_name] = [branch_data]
                else: merged_trees_data[tree_key][branch_name].append(branch_data)

    for tree_name, branches_data in merged_trees_data.items():
        for branch_name, data_list in branches_data.items():
            merged_trees_data[tree_name][branch_name] = da.concatenate(data_list)

    if trim_coords:
        x_min, x_max, y_min, y_max = trim_coords
        for tree_name in list(merged_trees_data.keys()):  # Use list() to avoid runtime error while modifying dict
            if "x_ax" in merged_trees_data[tree_name] and "y_ax" in merged_trees_data[tree_name]:
                x_data = merged_trees_data[tree_name]["x_ax"]
                y_data = merged_trees_data[tree_name]["y_ax"]

                mask = ((x_data >= x_min) & (x_data <= x_max) & (y_data >= y_min) & (y_data <= y_max))
                if mask.sum().compute() == 0:
                    print(f"No data after filtering in tree '{tree_name}'. Skipping tree.")
                    del merged_trees_data[tree_name]
                else: merged_trees_data[tree_name] = {key: value[mask] for key, value in merged_trees_data[tree_name].items()}

    with uproot.recreate(merged_file) as new_root_file:
        for tree_name, branches_data in merged_trees_data.items():
            new_root_file[tree_name] = branches_data

    Trash_Folder(trash_folder)
    print(f"Merged data written to: {merged_file}")


def Manage_Files(directory, starts_with, output_name):

    directory = os.path.join(directory, '')

    trash_folder = directory + 'Trash_' + output_name + '/'
    os.makedirs(trash_folder, exist_ok = True)

    file_list = []
    for file in os.listdir(directory):
        if file.endswith('.root') and file.startswith(starts_with): 
            file_path  = os.path.join(directory, file)
            shutil.move(file_path, trash_folder)
            file_list.append(os.path.join(trash_folder, file))

    if file_list == []: 
        print("No files found. Please check your inputs.")
        sys.exit(1)

    if output_name.endswith('.root'): output_name = output_name.rstrip('.root')
    merged_file = directory + output_name 
    if not os.path.exists(merged_file + ".root"): merged_file = merged_file + ".root"
    if os.path.exists(merged_file + ".root"):
        counter = 0
        while os.path.exists(f"{merged_file}_{counter}.root"): counter += 1
        merged_file = f"{merged_file}_{counter}.root"

    return trash_folder, file_list, merged_file

# 1.4. ========================================================================================================================================================

def Summary_Data(directory, root_file, data_tree, data_branch, summary_tree, summary_branches):

    import uproot

    directory = os.path.join(directory, '')
    if not root_file.endswith('.root'): root_file = root_file + '.root'
    file_path = directory + root_file

    opened_file = uproot.open(file_path)
        
    Hits_tree = uproot.dask(opened_file[data_tree], library='np', step_size = '50 MB')
    if data_branch not in Hits_tree.keys(): raise ValueError(f"Branch: '{data_branch}', not found in tree: '{data_tree}'.")
    Hits = Hits_tree[data_branch]
    NumberofHits = len(Hits)

    branches_num = (len(summary_branches))
    summary_tree = opened_file[summary_tree]
    for i in range(branches_num): 
        if summary_branches[i] not in summary_tree.keys(): raise ValueError(f"Branch: '{summary_branches[i]}', not found in tree: '{summary_tree}'.")

    Photons_Energy   = int(summary_tree[summary_branches[0]].array(library="np").mean())
    NumberofPhotons  = int(summary_tree[summary_branches[1]].array(library="np").sum())
    EnergyDeposition = summary_tree[summary_branches[2]].array(library="np").sum()
    RadiationDose    = summary_tree[summary_branches[3]].array(library="np").sum()

    print('-> Photons energy:', Photons_Energy, 'keV')
    print('-> Initial photons in simulation:', f"{NumberofPhotons:,}")
    print('-> Total photon hits in detector:', f"{NumberofHits:,}")
    print('-> Total energy deposited in tissue:', f"{EnergyDeposition:,.5f}", 'TeV')
    print('-> Dose of radiation received:', f"{RadiationDose:,.5f}", 'µSv')
    
    return NumberofHits, NumberofPhotons, EnergyDeposition, RadiationDose


def XY_1D_Histogram(directory, root_file, tree_name, x_branch, y_branch, range_x, range_y):

    import uproot, dask.array as dask_da

    directory = os.path.join(directory, '')
    if not root_file.endswith('.root'): root_file = root_file + '.root'
    file_path = directory + root_file
    opened_file = uproot.open(file_path)

    dataframe = uproot.dask(opened_file[tree_name], library='np', step_size = '50 MB')
    x_values = dataframe[x_branch]
    y_values = dataframe[y_branch]

    range_min_x = range_x[0]
    range_max_x = range_x[1]
    bins_x = range_x[2]

    range_min_y = range_y[0]
    range_max_y = range_y[1]
    bins_y = range_y[2]

    hist_x, bin_edges = dask_da.histogram(x_values, bins = bins_x, range = (range_min_x, range_max_x))
    hist_y, bin_edges = dask_da.histogram(y_values, bins = bins_y, range = (range_min_y, range_max_y))

    hist_x = hist_x.compute()
    hist_y = hist_y.compute()

    plt.figure(figsize = (14, 4)); plt.tight_layout()

    plt.subplot(1, 2, 1)
    plt.bar(bin_edges[:-1], hist_x, width = (bin_edges[1] - bin_edges[0]), align = 'edge', edgecolor = 'gray', linewidth = 0.8)
    plt.xlabel('Distance in X (mm)'); plt.ylabel('Frequency'); #plt.title('1D Histogram with Dask')

    plt.subplot(1, 2, 2)
    plt.bar(bin_edges[:-1], hist_y, width = (bin_edges[1] - bin_edges[0]), align = 'edge', edgecolor = 'gray', linewidth = 0.8)
    plt.xlabel('Distance in Y (mm)'); plt.ylabel('Frequency'); # plt.title('')

# 2.0. ========================================================================================================================================================

def Root_to_Heatmap(directory, root_file, tree_name, x_branch, y_branch, size, pixel_size):

    import uproot, dask.array as dask_da

    directory = os.path.join(directory, '')

    xlim = size[0]
    ylim = size[1]
    x_shift = size[2]
    y_shift = size[3]

    if not root_file.endswith('.root'): root_file = root_file + '.root'
    file_path = directory + root_file

    opened_file = uproot.open(file_path)
    tree = opened_file[tree_name]
    if tree is None: print(f"Tree '{tree_name}' not found in {root_file}"); return
    if x_branch not in tree or y_branch not in tree: print(f"Branches '{x_branch}' or '{y_branch}' not found in the tree"); return

    file_size = os.path.getsize(file_path)
    file_size_MB = file_size / (1000000)

    if file_size_MB < 1000: 
        chunk_size = int(file_size_MB / 10)
        if chunk_size < 10: chunk_size = '10 MB'
        else: chunk_size = f"{chunk_size} MB"
    else: chunk_size = '50 MB'

    dataframe = uproot.dask(opened_file[tree_name], library='np', step_size = chunk_size)
    x_values = dataframe[x_branch]
    y_values = dataframe[y_branch]

    x_data_shifted = x_values + x_shift            
    y_data_shifted = y_values + y_shift

    bins_x0 = np.arange(-xlim, xlim + pixel_size, pixel_size)
    bins_y0 = np.arange(-ylim, ylim + pixel_size, pixel_size)

    heatmap = dask_da.histogram2d(x_data_shifted, y_data_shifted, bins=[bins_x0, bins_y0])[0].compute()
    heatmap = np.rot90(heatmap.T, 2)

    return heatmap, bins_x0, bins_y0

def Logarithmic_Transform(heatmap):

    max_values = np.max(heatmap, axis = 0, keepdims = True)
    
    heatmap[heatmap <= 0] = np.nan

    with np.errstate(divide = 'warn', invalid = 'warn'): heatmap = np.log(max_values / heatmap)

    # heatmap = np.where(np.isnan(heatmap), 1, 0) # for debugging

    return heatmap

def Plot_Heatmap(heatmap, save_as):

    rows = heatmap.shape[0]
    cols = heatmap.shape[1]

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1); plt.imshow(heatmap, cmap="gray"); plt.colorbar()
    if save_as: plt.savefig(save_as + ".png", bbox_inches = "tight", dpi = 900)
    plt.subplot(1, 3, 2); plt.plot(heatmap[rows//2, :])
    plt.subplot(1, 3, 3); plt.plot(heatmap[:, cols//2])

def Plot_Plotly(heatmap, xlim, ylim):

    import plotly.graph_objects as go

    fig = go.Figure(go.Heatmap(z = heatmap, x = xlim, y = ylim, colorscale = [[0, 'black'], [1, 'white']], showscale = True))
    fig.update_layout(width = 800, height = 800, yaxis = dict(autorange = 'reversed'))    
    fig.show()

def Save_Heatmap_to_CSV(heatmap, save_folder, save_as):

    save_as = save_folder + save_as + ".csv"
    np.savetxt(save_as, heatmap, delimiter=',', fmt='%.4f')

def Read_Heatmap_from_CSV(save_folder, csv_name):

    csv_path = save_folder + csv_name + ".csv"
    heatmap = np.genfromtxt(csv_path, delimiter = ',')
    return heatmap

# 3.0. ========================================================================================================================================================

def IsolateTissues(low_energy_img, high_energy_img, sigma1, sigma2, wn, save_in, save_as, save_all):

    from scipy.ndimage import gaussian_filter

    save_as_1 = save_as[0]; save_as_2 = save_as[1]; save_as_3 = save_as[2]; save_as_4 = save_as[3]
    save_as_5 = save_as[4]; save_as_6 = save_as[5]; save_as_7 = save_as[6]; save_as_8 = save_as[7]

    U_b_l = 0.7519 # mu1
    U_b_h = 0.3012 # mu2
    U_t_l = 0.26 # mu3
    U_t_h = 0.18 # mu4

    SLS_Bone = ( (U_t_h/U_t_l) * low_energy_img ) - high_energy_img
    SLS_Tissue = high_energy_img - ( low_energy_img * (U_b_h/U_b_l) )

    SSH_Bone = ( (U_t_h/U_t_l) * low_energy_img) - gaussian_filter(high_energy_img, sigma = sigma1)
    SSH_Tissue = gaussian_filter( high_energy_img, sigma = sigma1) - ( low_energy_img * (U_b_h/U_b_l) )

    ACNR_Bone     = SLS_Bone + (gaussian_filter(SLS_Tissue, sigma = sigma1) * wn) - 1
    ACNR_SSH_Bone = SSH_Bone + (gaussian_filter(SSH_Tissue, sigma = sigma2) * wn) - 1
    ACNR_Tissue = SLS_Tissue + (gaussian_filter(SLS_Bone,   sigma = sigma1) * wn) - 1

    plt.imshow(low_energy_img, cmap='gray'); plt.axis('off')
    if save_as_1 != '': plt.savefig(save_in + save_as_1, bbox_inches = 'tight', dpi = 600); plt.close()
    plt.imshow(high_energy_img, cmap='gray'); plt.axis('off')
    if save_as_2 != '': plt.savefig(save_in + save_as_2, bbox_inches = 'tight', dpi = 600); plt.close()
    plt.imshow(SLS_Bone, cmap='gray'); plt.axis('off')
    if save_as_3 != '': plt.savefig(save_in + save_as_3, bbox_inches = 'tight', dpi = 600); plt.close()
    plt.imshow(SLS_Tissue, cmap='gray'); plt.axis('off')
    if save_as_4 != '': plt.savefig(save_in + save_as_4, bbox_inches = 'tight', dpi = 600); plt.close()
    plt.imshow(SSH_Bone, cmap='gray'); plt.axis('off')
    if save_as_5 != '': plt.savefig(save_in + save_as_5, bbox_inches = 'tight', dpi = 600); plt.close()
    plt.imshow(SSH_Tissue, cmap='gray'); plt.axis('off')
    if save_as_6 != '': plt.savefig(save_in + save_as_6, bbox_inches = 'tight', dpi = 600); plt.close()
    plt.imshow(ACNR_Bone, cmap='gray'); plt.axis('off')
    if save_as_7 != '': plt.savefig(save_in + save_as_7, bbox_inches = 'tight', dpi = 600); plt.close()
    plt.imshow(ACNR_Tissue, cmap='gray'); plt.axis('off')
    if save_as_8 != '': plt.savefig(save_in + save_as_8, bbox_inches = 'tight', dpi = 600); 
    plt.close()

    plt.figure(figsize = (18, 10)); plt.tight_layout()
    plt.subplot(2, 4, 1); plt.imshow(low_energy_img,    cmap='gray'); plt.axis('off');  plt.title("Low Energy")
    plt.subplot(2, 4, 2); plt.imshow(high_energy_img,   cmap='gray'); plt.axis('off');  plt.title("High Energy")
    plt.subplot(2, 4, 3); plt.imshow(SLS_Bone,          cmap='gray'); plt.axis('off');  plt.title("Bone [SLS]")
    plt.subplot(2, 4, 4); plt.imshow(SLS_Tissue,        cmap='gray'); plt.axis('off');  plt.title("Tissue [SLS]")
    plt.subplot(2, 4, 5); plt.imshow(SSH_Bone,          cmap='gray'); plt.axis('off');  plt.title("Bone [SSH]")
    plt.subplot(2, 4, 6); plt.imshow(SSH_Tissue,        cmap='gray'); plt.axis('off');  plt.title("Tissue [SSH]")
    plt.subplot(2, 4, 7); plt.imshow(ACNR_Bone,         cmap='gray'); plt.axis('off');  plt.title("Bone [ACNR]")
    plt.subplot(2, 4, 8); plt.imshow(ACNR_SSH_Bone,     cmap='gray'); plt.axis('off');  plt.title("Bone [ACNR + SSH]")
    # plt.subplot(2, 4, 8); plt.imshow(ACNR_Tissue,       cmap='gray'); plt.axis('off');  plt.title("Tissue [ACNR]")
    if save_all != '': plt.savefig(save_in + save_all, bbox_inches = 'tight', dpi = 600)
   
    return SLS_Bone, SLS_Tissue, SSH_Bone, SSH_Tissue, ACNR_Bone, ACNR_Tissue

# 4.0. ========================================================================================================================================================

def BMO(SLS_Bone, SLS_Tissue):

    U_b_l = 0.7519 # mu1
    U_b_h = 0.3012 # mu2
    U_t_l = 0.281 # mu3
    U_t_h = 0.192 # mu4

    Thick_cons_bone = (U_t_l) / ( (U_t_h * U_b_l) - (U_t_l * U_b_h) )
    thickness_bone = Thick_cons_bone * SLS_Bone
    Thick_cons_tissue = (U_t_l) / ( (U_t_l * U_b_h) - (U_t_h * U_b_l) )
    thickness_tissue = Thick_cons_tissue * SLS_Tissue

    return thickness_bone, thickness_tissue

# 5.1 ========================================================================================================================================================

def Interactive_CNR(cropped_image):

    data = np.array(cropped_image)
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, cmap='gray')

    rectangles = []
    start_pos = [None]  # Using a list to store coordinates
    signal_avg = [0]
    background_avg = [0]
    background_std = [0]

    def on_press(event):
        if event.inaxes != ax: return
        start_pos[0] = (event.xdata, event.ydata)
        rect = plt.Rectangle(start_pos[0], 1, 1, fill=False, color='blue', lw=1)
        ax.add_patch(rect)
        rectangles.append(rect)

        if len(rectangles) > 2:
            first_rect = rectangles.pop(0)
            second_rect = rectangles.pop(0)
            first_rect.remove()
            second_rect.remove()

        fig.canvas.draw()

    def on_motion(event):
        if start_pos[0] is None or event.inaxes != ax: return
        width = event.xdata - start_pos[0][0]
        height = event.ydata - start_pos[0][1]
        rect = rectangles[-1]
        rect.set_width(width)
        rect.set_height(height)
        fig.canvas.draw()

    def on_release(event):
        if start_pos[0] is None or event.inaxes != ax: return
        end_pos = (event.xdata, event.ydata)

        x1 = start_pos[0][0]
        y1 = start_pos[0][1]
        x2 = end_pos[0]
        y2 = end_pos[1]

        if len(rectangles) == 1:
            if x2 > x1:
                if y2 > y1: signal = data[round(y1):round(y2), round(x1):round(x2)]
                else:       signal = data[round(y2):round(y1), round(x1):round(x2)]
            else:
                if y2 > y1: signal = data[round(y1):round(y2), round(x2):round(x1)]
                else:       signal = data[round(y2):round(y1), round(x2):round(x1)]

            signal_avg[0] = np.average(signal)
            print("Signal avg: "+str(signal_avg[0]))
        else:
            if x2 > x1:
                if y2 > y1: background = data[round(y1):round(y2), round(x1):round(x2)]
                else:       background = data[round(y2):round(y1), round(x1):round(x2)]
            else:
                if y2 > y1: background = data[round(y1):round(y2), round(x2):round(x1)]
                else:       background = data[round(y2):round(y1), round(x2):round(x1)]

            background_avg[0] = np.average(background)
            background_std[0] = np.std(background)
            print("Background avg: "+str(background_avg[0]))
            print("Background std dev: "+str(background_std[0]))
            cnr = (signal_avg[0] - background_avg[0]) / background_std[0]
            print("CNR: " + str(cnr) + '\n')

        start_pos[0] = None

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.show()

# 5.2 ========================================================================================================================================================

def Fixed_CNR(image_path, save_as, coords_signal, coords_bckgrnd):
    
    from PIL import Image

    image = Image.open(image_path)
    image = image.convert('L')
    cropped_image = image
    # cropped_image = image.crop((520, 450, image.width - 580, image.width - 440))
    data = np.array(cropped_image)

    plt.imshow(data, cmap = 'gray')
    plt.axis('off')

    signal_avg = 0
    background_avg = 0
    background_std = 0

    x1_signal = coords_signal[0]
    y1_signal = coords_signal[1]
    x2_signal = coords_signal[2]
    y2_signal = coords_signal[3]

    plt.gca().add_patch(plt.Rectangle((x1_signal, y1_signal), x2_signal - x1_signal, y2_signal - y1_signal, linewidth=2, edgecolor='yellow', facecolor='none'))

    if x2_signal > x1_signal:
        if y2_signal > y1_signal:
            signal = data[round(y1_signal):round(y2_signal), round(x1_signal):round(x2_signal)]
        else:
            signal = data[round(y2_signal):round(y1_signal), round(x1_signal):round(x2_signal)]
    else:
        if y2_signal > y1_signal:
            signal = data[round(y1_signal):round(y2_signal), round(x2_signal):round(x1_signal)]
        else:
            signal = data[round(y2_signal):round(y1_signal), round(x2_signal):round(x1_signal)]

    signal_avg = np.average(signal)
    # signal_std = np.std(signal)
    print("Signal avg: ", round(signal_avg, 3))

    x1_background = coords_bckgrnd[0]
    y1_background = coords_bckgrnd[1]
    x2_background = coords_bckgrnd[2]
    y2_background = coords_bckgrnd[3]

    plt.gca().add_patch(plt.Rectangle((x1_background, y1_background), x2_background - x1_background, y2_background - y1_background, linewidth=2, edgecolor='red', facecolor='none'))

    if x2_background > x1_background:
        if y2_background > y1_background:
            background = data[round(y1_background):round(y2_background), round(x1_background):round(x2_background)]
        else:
            background = data[round(y2_background):round(y1_background), round(x1_background):round(x2_background)]
    else:
        if y2_background > y1_background:
            background = data[round(y1_background):round(y2_background), round(x2_background):round(x1_background)]
        else:
            background = data[round(y2_background):round(y1_background), round(x2_background):round(x1_background)]

    background_avg = np.average(background)
    background_std = np.std(background)

    print("Background avg: ", round(background_avg, 3))
    print("Background std dev: ", round(background_std, 3))

    cnr = (signal_avg - background_avg) / background_std
    # cnr = (background_avg - signal_avg) / signal_std
    print("CNR: ", round(cnr, 1))

    if save_as != '': plt.savefig('RESULTS/' + save_as + '.png', bbox_inches = 'tight', dpi = 900)

# 6.1 ========================================================================================================================================================

def Denoise_EdgeDetection(path, isArray, sigma_color, sigma_spatial):

    from skimage.restoration import denoise_bilateral; from PIL import Image
    
    if isArray == True:
        original_image = np.array(path)
    else:
        original_image = Image.open(path)
        
    denoised_image = denoise_bilateral(original_image, sigma_color = sigma_color, sigma_spatial = sigma_spatial, channel_axis = None)

    save_as = ''

    plt.figure(figsize = (10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(denoised_image, cmap = 'gray')
    plt.title('Denoised Image')
    plt.axis('off')
    if save_as != '': plt.savefig('RESULTS/' + save_as + '.png', bbox_inches = 'tight', dpi = 900)

    plt.subplot(1, 2, 2)
    plt.imshow(original_image, cmap = 'gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.show()

    return denoised_image

# 6.2 ========================================================================================================================================================

def Denoise(array, isHann, alpha, save_as, isCrossSection):
    
    from scipy import signal; from scipy.fft import fft2, fftshift, ifft2

    image = array

    fft_image = fft2(image)
    fft_image = fftshift(fft_image)

    rows, cols = image.shape

    if isHann == True:
    
        l = rows * alpha
        a = np.hanning(l)
        b = np.hanning(l)

        padding_size = rows - len(a)
        left_padding = padding_size // 2
        right_padding = padding_size - left_padding
        a = np.pad(a, (left_padding, right_padding), mode='constant')

        padding_size = cols - len(b)
        left_padding = padding_size // 2
        right_padding = padding_size - left_padding
        b = np.pad(b, (left_padding, right_padding), mode='constant')

        window = np.outer(a, b)

    else:

        a = signal.windows.tukey(rows, alpha)
        b = signal.windows.tukey(rows, alpha)
        window = np.outer(a, b)

    fft_image_2 = fft_image * (window)
    fft_image = fftshift(fft_image_2)
    fft_image = (ifft2(fft_image))
    fft_image = (np.abs(fft_image))

    if isCrossSection == True:
        
        plt.figure(figsize = (7, 3))
        plt.subplot(1, 2, 1); plt.plot(a); plt.title('Window')
        plt.subplot(1, 2, 2); plt.plot(np.abs((fft_image_2[:][rows//2]))); plt.title('F. Transform Slice')

        plt.figure(figsize = (7, 3))
        plt.subplot(1, 2, 1); plt.plot(image[:][rows//2]); plt.title('Original Slice')
        plt.subplot(1, 2, 2); plt.plot(np.abs(fft_image[:][rows//2])); plt.title('Denoised Slice')

    plt.figure(figsize = (8, 4))
    plt.subplot(1, 2, 1); plt.imshow(image, cmap = 'gray'); plt.title('Original Image'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(fft_image, cmap = 'gray'); plt.title('Filtered Image'); plt.axis('off')
    if save_as != '': plt.savefig('Results/' + save_as + '.png', dpi = 900)
    plt.show()

    return fft_image

# 7.0 ========================================================================================================================================================

def Plotly_Heatmap_1(array, xlim, ylim, title, x_label, y_label, width, height, save_as):

    import plotly.io as pio, plotly.graph_objects as go

    font_family = 'Merriweather'
    family_2    = 'Optima'
    font_small  = 16
    font_medium = 20
    font_large  = 18

    fig = go.Figure(go.Heatmap(z = array, x = xlim, y = ylim,
                                colorscale = [[0, 'black'], [1, 'white']], 
                                colorbar = dict(title = "Density", tickfont = dict(family = family_2, size = 15, color = 'Black'))))
    
    fig.update_layout(
                    title = dict(text = title, font = dict(family = font_family, size = font_large, color = "Black"), 
                                 x = 0.51, y = 0.93, yanchor = 'middle', xanchor = 'center'),
                    xaxis_title = dict(text = x_label, font = dict(family = font_family, size = font_medium, color = "Black")),
                    yaxis_title = dict(text = y_label, font = dict(family = font_family, size = font_medium, color = "Black")),
                    xaxis = dict(tickfont = dict(family = family_2, size = font_small, color = "Black"), title_standoff = 25),
                    yaxis = dict(tickfont = dict(family = family_2, size = font_small, color = "Black"), title_standoff = 10, range=[max(xlim), min(xlim)]),
                    width = width, height = height, margin = dict(l = 105, r = 90, t = 90, b = 90)
    )
   
    if save_as != '': pio.write_image(fig, save_as + '.png', width = width, height = height, scale = 5)
    fig.show()


def Plotly_Heatmap_2(array, xlim, ylim, title, x_label, y_label, sqr_1_coords, sqr_2_coords, annotation, width, height, save_as):

    import plotly.graph_objects as go, plotly.io as pio

    font_family = 'Merriweather'
    family_2    = 'Optima'
    font_small  = 18
    font_medium = 20
    font_large  = 18

    fig = go.Figure(go.Heatmap(z = array, x = xlim, y = ylim,
                                colorscale = [[0, 'black'], [1, 'white']], showscale = False,
                                colorbar = dict(title = "Density", tickfont = dict(family = family_2, size = 15, color = 'Black'))))
    
    fig.update_layout(
                    title = dict(text = title, font = dict(family = font_family, size = font_large, color = "Black"), 
                                 x = 0.51, y = 0.93, yanchor = 'middle', xanchor = 'center'),
                    xaxis_title = dict(text = x_label, font = dict(family = font_family, size = font_medium, color = "Black")),
                    yaxis_title = dict(text = y_label, font = dict(family = font_family, size = font_medium, color = "Black")),
                    xaxis = dict(tickfont = dict(family = family_2, size = font_small, color = "Black"), title_standoff = 25, range=[max(xlim), min(xlim)]),
                    yaxis = dict(tickfont = dict(family = family_2, size = font_small, color = "Black"), title_standoff = 10, range=[max(xlim), min(xlim)],
                                 showticklabels = False
                                ),
                    width = width, height = height, margin = dict(l = 105, r = 90, t = 90, b = 90),
                    annotations = [dict(x = 0.95, y = 0.15,  xref = 'paper', yref = 'paper', showarrow = False,
                                        font = dict(family = family_2, size = 18, color = "White"),
                                        bgcolor = "rgba(255, 255, 255, 0.1)", borderpad = 8, bordercolor = "White", borderwidth = 0.2,
                                        text = annotation)])

    fig.add_shape(type = "rect", line = dict(color = "blue", width = 2), fillcolor = "rgba(0, 0, 0, 0)",
                  x0 = sqr_1_coords[0], y0 = sqr_1_coords[1], x1 = sqr_1_coords[2], y1 = sqr_1_coords[3]) 
    
    fig.add_shape(type = "rect", line = dict(color = "red", width = 2), fillcolor = "rgba(0, 0, 0, 0)",
                  x0 = sqr_1_coords[0], y0 = sqr_1_coords[1], x1 = sqr_1_coords[2], y1 = sqr_1_coords[3]) 
   
    if save_as != '': pio.write_image(fig, save_as + '.png', width = width, height = height, scale = 5)
    fig.show()

# 8.0 ========================================================================================================================================================

def ClearFolder(directory):

    from send2trash import send2trash
        
    for file_name in os.listdir(directory):

        if file_name.startswith('CT_') and file_name.endswith('.root'):

            file_path = os.path.join(directory, file_name)
            # if os.path.isfile(file_path):
            try: send2trash(file_path)
            except Exception as e: print(f"Error deleting file {file_path}: {e}")


def Generate_CT_MAC_Template(
    threads              = None,  # Optional parameter
    spectra_mode         = None,  # Optional parameter:   'mono (1)'   or 'poly (2)'
    detector_parameters  = None,  # Optional parameter
    gun_parameters       = None,  # Optional parameter
):

    if spectra_mode        is None: spectra_mode = 'mono'
    if detector_parameters is None: detector_parameters = {'nColumns': 1, 'nRows': 1}
    if gun_parameters      is None: gun_parameters = {'X': 0, 'Y': 0, 'gaussX': 'true', 'SpanX': 230, 'SpanY': 0.01}

    mac_template = []

    mac_template.extend([
        f"/myDetector/Rotation {'{angle}'}",
        f"/myDetector/nColumns {detector_parameters['nColumns']}",
        f"/myDetector/nRows {detector_parameters['nRows']}",
        f"/run/reinitializeGeometry"
    ])

    if threads: mac_template.append(f"/run/numberOfThreads {'{Threads}'}")
    mac_template.append("/run/initialize")

    mac_template.extend([
        f"/Pgun/X {gun_parameters['X']} mm",
        f"/Pgun/gaussX {gun_parameters['gaussX']}",
        f"/Pgun/SpanX {gun_parameters['SpanX']} mm",
        f"/Pgun/Xcos true",
        f"/Pgun/Y {gun_parameters['Y']} mm",
        f"/Pgun/SpanY {gun_parameters['SpanY']} mm"
    ])

    if spectra_mode == 'mono' or spectra_mode == 0:
        mac_template.append(f"/gun/energy {'{Energy}'} keV")

    if spectra_mode == '80kvp' or spectra_mode == 1:
        mac_template.append(f"/Pgun/Mode 1")

    if spectra_mode == '140kvp' or spectra_mode == 2:
        mac_template.extend(f"/Pgun/Mode 2")

    mac_template.append(f"{'{beam_lines}'}")

    return "\n".join(mac_template)  


def CT_Loop(threads, starts_with, angles, slices, alarm):

    from tqdm.notebook import tqdm

    mac_filename = 'CT.mac'
    
    if platform.system() == "Darwin":
        directory = Path('BUILD')
        executable_file = "Sim"
        run_sim = f"./{executable_file} {mac_filename} . . ."

    elif platform.system() == "Windows":
        directory = Path('build') / 'Release'
        executable_file = "Sim.exe"
        run_sim = fr".\{executable_file} .\{mac_filename} . . ."
        print(run_sim)

    elif platform.system() == "Linux":
        directory = Path('build') / 'Release'
        executable_file = "Sim.exe"
        run_sim = fr"./{executable_file} {mac_filename} . . ."
        print(run_sim)

    else: raise EnvironmentError("Unsupported operating system")

    root_folder = directory / "ROOT/"
    mac_filepath = directory / mac_filename
    ct_folder = directory / f"ROOT/CT/"
    os.makedirs(ct_folder, exist_ok = True)

    y_start = slices[0]
    y_end = slices[1]
    step = slices[2]

    exit_requested = False
    
    for angle in tqdm(range(angles[0], angles[1]), desc = "Creating CT", unit = "Angles", leave = True):
        
        # if exit_requested == True: print('breaking incorrectly'); break
        
        # try:
            ClearFolder(root_folder) # deletes residual files but doesn't delete subfolders

            mac_template = Generate_CT_MAC_Template(threads, spectra_mode='mono', detector_parameters=None, gun_parameters=None)
            
            beam_lines = ""
            for y in range(y_start, y_end + 1, step): 
                beam_lines += f"""
                /Pgun/Y {y} mm
                /run/beamOn 150000
                """

            energy = 80

            filled_template = mac_template.format(angle = angle, Threads = threads, Energy = energy, beam_lines = beam_lines)
            with open(mac_filepath, 'w') as mac_file: mac_file.write(filled_template)

            try: subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e: print(f"Error during simulation: {e}"); continue  # Skip to the next angle
        
            output_name = f"Aang_{angle}"
            if os.path.exists(ct_folder / f"{output_name}.root"):
                counter = 0
                while os.path.exists(ct_folder / f"{output_name}_{counter}.root"): counter = counter + 1
                output_name = root_folder / f"{output_name}_{counter}"

            with open(os.devnull, "w") as fnull: 
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    Merge_Roots_HADD(root_folder, starts_with, output_name, trim_coords = None)

            merged_file_path = root_folder / f"{output_name}.root"

            if os.path.exists(merged_file_path): shutil.move(merged_file_path, ct_folder)

        # except KeyboardInterrupt:
        #     if exit_requested == True: print("Second interrupt detected, stopping execution."); ClearFolder(root_folder); break
        #     if exit_requested == False: print('Exit request detected, exiting after current iteration.'); exit_requested = True; continue
    
    print("Finished Simulating CT")
    if alarm == True: PlayAlarm()


def CT_Summary_Data(directory, tree, branches):

    import uproot

    NumberofPhotons = 0
    EnergyDeposition = 0
    RadiationDose = 0

    num_of_files = 0

    for file in os.listdir(directory):

        file_path = os.path.join(directory, file)
        if not file_path.endswith('.root'): continue

        with uproot.open(file_path) as root_file:

            tree_data = root_file[tree]
            if branches[0] not in tree_data.keys(): raise ValueError(f"Branch: '{branches[0]}', not found in tree: '{tree}'.")
            if branches[1] not in tree_data.keys(): raise ValueError(f"Branch: '{branches[1]}', not found in tree: '{tree}'.")
            if branches[2] not in tree_data.keys(): raise ValueError(f"Branch: '{branches[2]}', not found in tree: '{tree}'.")

            NumberofPhotons  = NumberofPhotons  + tree_data[branches[0]].array(library="np").sum()
            EnergyDeposition = EnergyDeposition + tree_data[branches[1]].array(library="np").sum()
            RadiationDose    = RadiationDose    + tree_data[branches[2]].array(library="np").sum()
        
        num_of_files = num_of_files + 1

    print('Files processed: ', num_of_files)

    print('Initial photons in simulation:', NumberofPhotons)
    print('Total energy deposited in tissue (TeV):', round(EnergyDeposition, 5))
    print('Dose of radiation received (uSv):', round(RadiationDose, 5))
        
    return NumberofPhotons, EnergyDeposition, RadiationDose


def Calculate_Projections(directory, filename, degrees, root_structure, dimensions, pixel_size, csv_folder):
    
    from tqdm import tqdm; import dask; from dask import delayed; from dask.diagnostics import ProgressBar

    os.makedirs(csv_folder, exist_ok = True)

    start = degrees[0]
    end = degrees[1]
    deg = degrees[2]
    projections = np.arange(start, end+1, deg)

    tree_name = root_structure[0]
    x_branch = root_structure[1]
    y_branch = root_structure[2]

    @delayed
    def calculate_heatmaps(i, directory, tree_name, x_branch, y_branch, dimensions, pixel_size, csv_folder):
        root_name = f"{filename}_{i}.root"
        heatmap, xlim, ylim = Root_to_Heatmap(directory, root_name, tree_name, x_branch, y_branch, dimensions, pixel_size)

        write_name = csv_folder + f"{'CT_raw_'}{i}.csv"
        np.savetxt(write_name, heatmap, delimiter=',', fmt='%.2f')

        return heatmap

    heatmap_tasks = []
    for i in projections: heatmap_tasks += [calculate_heatmaps(i, directory, tree_name, x_branch, y_branch, dimensions, pixel_size, csv_folder)]
    print('Calculating Heatmaps for Every Angle in CT:')
    with ProgressBar(): dask.compute(*heatmap_tasks, scheduler='processes')

    read_name = csv_folder + f"{'CT_raw_'}{0}.csv"
    raw_heatmap = np.genfromtxt(read_name, delimiter=',')

    lower = np.percentile(raw_heatmap, 0)
    upper = np.percentile(raw_heatmap, 98)
    clipped_htmp = np.clip(raw_heatmap, lower, upper)
    Plot_Heatmap(clipped_htmp, save_as='')

    return raw_heatmap

def old_RadonReconstruction(csv_read, csv_write, degrees, layers, sigma):

    import plotly.graph_objects as go; from skimage.transform import iradon; from scipy import ndimage
    import dask; from dask.diagnostics import ProgressBar; from dask import delayed
    from tqdm import tqdm

    start = degrees[0]
    end = degrees[1]
    deg = degrees[2]
    projections = np.arange(start, end+1, deg)

    initial = layers[0]
    final = layers[1]
    spacing = layers[2]
    slices = np.round(np.arange(initial, final, spacing))
    
    heatmap_matrix = np.zeros(len(projections), dtype = object)
    sinogram_matrix = np.zeros(len(slices), dtype = object)
    slices_matrix = np.zeros(len(slices), dtype = object)

    for i in tqdm(projections, desc = 'Performing Logarithmic Transformation', unit = ' Heatmaps', leave = True):

        read_name = csv_read + f"{'CT_raw_'}{i}.csv"
        raw_heatmap = pd.read_csv(read_name, delimiter = ',')
        
        raw_heatmap = ndimage.gaussian_filter(raw_heatmap, sigma)
        heatmap = Logarithmic_Transform(raw_heatmap)
        heatmap_matrix[i] = heatmap

    heatmap_matrix = np.stack(heatmap_matrix, axis=0)
    print(heatmap_matrix.shape)
    

    for i, y in enumerate(tqdm(slices, desc = 'Reconstructing slices', unit = ' Slices', leave = True)):

        sinogram = []
        for heatmap in heatmap_matrix: sinogram.append(heatmap[y])
        sinogram = np.array(sinogram).T
        sinogram_matrix[i] = sinogram

        reconstructed_slice = iradon(sinogram, theta = projections)
        slices_matrix[i] = reconstructed_slice

    fig = go.Figure(go.Heatmap(z = sinogram_matrix[30], colorscale = [[0, 'black'], [1, 'white']], showscale = True))
    fig.update_layout(width = 500, height = 500, yaxis = dict(autorange = 'reversed'))
    fig.show()

    Plot_Heatmap(sinogram_matrix[30], save_as='')

    fig = go.Figure(go.Heatmap(z = slices_matrix[30], colorscale = [[0, 'black'], [1, 'white']], showscale = True))
    fig.update_layout(width = 500, height = 500, yaxis = dict(autorange = 'reversed'))
    fig.show()


def RadonReconstruction(csv_read, csv_write, degrees, layers, sigma):

    import plotly.graph_objects as go; from skimage.transform import iradon; from scipy import ndimage
    import dask; from dask.diagnostics import ProgressBar; from dask import delayed

    start = degrees[0]
    end = degrees[1]
    deg = degrees[2]
    projections = np.arange(start, end+1, deg)

    initial = layers[0]
    final = layers[1]
    spacing = layers[2]
    slices_vector = np.round(np.arange(initial, final, spacing))
    
    heatmap_matrix = np.zeros(len(projections), dtype = object)
    sinogram_matrix = np.zeros(len(slices_vector), dtype = object)
    slices_matrix = np.zeros(len(slices_vector), dtype = object)

    @delayed
    def process_heatmap(i, csv_read, sigma):
        
        read_name = csv_read + f"{'CT_raw_'}{i}.csv"
        raw_heatmap = pd.read_csv(read_name, delimiter=',', header=None)
        raw_heatmap = raw_heatmap.to_numpy()        
        raw_heatmap = ndimage.gaussian_filter(raw_heatmap, sigma)
        heatmap = Logarithmic_Transform(raw_heatmap)

        return heatmap

    @delayed
    def compute_sinogram(y, heatmap_matrix):
        
        sinogram = []
        for heatmap in heatmap_matrix: sinogram.append(heatmap[y])
        sinogram = np.array(sinogram).T

        return sinogram

    @delayed
    def reconstruct_slice(i, sinogram_matrix, projections):
        
        sinogram = sinogram_matrix[i]
        reconstructed_slice = iradon(sinogram, theta=projections)

        # if i < 10: Plot_Heatmap(sinogram_matrix[i], save_as='')
        suma = np.sum(reconstructed_slice)
        if suma > 0: print('Suma:', suma)
        
        return reconstructed_slice
    
    heatmap_tasks = []
    for i in projections: heatmap_tasks = heatmap_tasks + [process_heatmap(i, csv_read, sigma)]
    print('Reading and Performing Logarithmic Transform:')
    with ProgressBar(): heatmaps = dask.compute(*heatmap_tasks, scheduler='processes')
    heatmap_matrix = np.stack(heatmaps, axis=0)
    print('Heatmap Matrix Shape:', heatmap_matrix.shape, '\n')

    sinogram_tasks = []
    for y in slices_vector: sinogram_tasks = sinogram_tasks + [compute_sinogram(y, heatmap_matrix)]
    print('Computing Sinograms:')
    with ProgressBar(): sinograms = dask.compute(*sinogram_tasks)    
    sinogram_matrix = np.stack(sinograms, axis=0)
    print('Sinogram Matrix Shape:', sinogram_matrix.shape, '\n')

    slices_tasks = []
    for i in range(len(slices_vector)): slices_tasks = slices_tasks + [reconstruct_slice(i, sinogram_matrix, projections)]
    print('Reconstruction slices:')
    with ProgressBar(): slices = dask.compute(*slices_tasks) 
    slices_matrix = np.stack(slices, axis=0)
    print('Slices Matrix Shape:', slices_matrix.shape, '\n')

    # os.makedirs(csv_write, exist_ok = True)
    # for i, y in enumerate(slices_vector): 
    #     slice = slices_matrix[i]
    #     slice[np.isnan(slice)] = 0
    #     write_name = csv_write + f"{'CT_slice_'}{y}mm.csv"
    #     np.savetxt(write_name, slice, delimiter=',', fmt='%d') #.2f

    for i in range(10):
        Plot_Heatmap(heatmap_matrix[i], save_as='')
        Plot_Heatmap(sinogram_matrix[i], save_as='')
        Plot_Heatmap(slices_matrix[i], save_as='')

    # fig = go.Figure(go.Heatmap(z = sinogram_matrix[30], colorscale = [[0, 'black'], [1, 'white']], showscale = True))
    # fig.update_layout(width = 500, height = 500, yaxis = dict(autorange = 'reversed'))
    # fig.show()

    # fig = go.Figure(go.Heatmap(z = slices_matrix[10], colorscale = [[0, 'black'], [1, 'white']], showscale = True))
    # fig.update_layout(width = 500, height = 500, yaxis = dict(autorange = 'reversed'))
    # fig.show()


def CoefficientstoHU(csv_slices, mu_water, air_parameter):

    import plotly.graph_objects as go

    # initial = slices[0]
    # final = slices[1]
    # spacing = slices[2]
    # slices = np.round(np.arange(initial, final, spacing))

    slices = os.listdir(csv_slices)
    HU_images = np.zeros(len(slices), dtype="object")

    for i in range(len(HU_images)):

        HU_images[i] = np.round(1000 * ((slices[i] - mu_water) / mu_water)).astype(int)
        HU_images[i][HU_images[i] < air_parameter] = -1000

    fig = go.Figure(go.Heatmap(z = HU_images[0], colorscale = [[0, 'black'], [1, 'white']],))
    fig.update_layout(width = 600, height = 600, xaxis = dict(autorange = 'reversed'), yaxis = dict(autorange = 'reversed'))
    fig.show()

    return HU_images


def Export_to_Dicom(HU_images, size_y, directory, compressed):

    import pydicom; from pydicom.uid import RLELossless; from pydicom.encaps import encapsulate; from pydicom.dataset import Dataset; 
    # from pydicom.uid import ExplicitVRLittleEndian; from pydicom.pixels import compress; from pydicom.dataset import FileDataset

    image2d = HU_images[0].astype('int16')
    print("Setting file meta information...")
    print("Setting pixel data...")
    
    # instanceUID =  pydicom.uid.generate_uid()
    seriesUID = pydicom.uid.generate_uid()
    studyInstance = pydicom.uid.generate_uid()
    frameOfReference = pydicom.uid.generate_uid()

    if compressed:
        for i, image in enumerate(HU_images):
            meta = pydicom.Dataset()
            meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
            instanceUID_var = pydicom.uid.generate_uid()
            meta.MediaStorageSOPInstanceUID = instanceUID_var
            # meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.70' #LOSSLESS
            meta.TransferSyntaxUID= pydicom.uid.ImplicitVRLittleEndian
            # instanceUID_var = instanceUID[:-7]+f".000{i+1}.0"
            ds = Dataset()
            ds.file_meta = meta

            # ds.is_little_endian = True
            # ds.is_implicit_VR = False

            ds.SOPInstanceUID = instanceUID_var
            ds.SOPClassUID = pydicom.uid.CTImageStorage 
            ds.PatientName = "NAME^NONE"
            ds.PatientID = "NOID"

            ds.Modality = "CT"
            ds.SeriesInstanceUID = seriesUID
            ds.StudyInstanceUID = studyInstance
            ds.FrameOfReferenceUID = frameOfReference
            ds.SeriesNumber = 3

            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.SamplesPerPixel = 1
            ds.HighBit = 15
            ds.WindowCenter = 30
            ds.WindowWidth = 100

            ds.Rows = image2d.shape[0]
            ds.Columns = image2d.shape[1]
            ds.AcquisitionNumber = 1

            ds.ImageOrientationPatient = r"1\0\0\0\1\0"
            ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

            ds.RescaleIntercept = "0"
            ds.RescaleSlope = "1"
            ds.PixelSpacing = r"0.5\0.5"
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 1
            # ds.RescaleIntercept = "-1024"
            ds.RescaleType = 'HU'
            name = directory + f"/I{i}"
            # pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
            thickness = (size_y * 2)/len(HU_images)
            ds.SliceThickness = str(thickness)
            ds.SpacingBetweenSlices = str(thickness)
            ds.ImagePositionPatient = f"0\\0\\{thickness * i}"
            ds.SliceLocation = str(thickness * i)+'00'
            ds.InstanceNumber = i+1
            # instanceUID_var = instanceUID[:-7]+f".000{i+1}.0"
            # meta.MediaStorageSOPInstanceUID = instanceUID_var
            # ds.SOPInstanceUID = instanceUID_var
            image2d = image.astype('int16')
            # ds.PixelData = image2d.tobytes()
            ds.PixelData = encapsulate([image2d.tobytes()])
            ds.compress(RLELossless)
            pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
            ds.save_as(name + '.dcm')
        
    else:
        for i, image in enumerate(HU_images):
            
            meta = pydicom.Dataset()
            meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
            instanceUID_var = pydicom.uid.generate_uid()
            # instanceUID_var = instanceUID[:-7]+f".000{i+1}.0"
            meta.MediaStorageSOPInstanceUID = instanceUID_var
            # meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.70' #LOSSLESS
            meta.TransferSyntaxUID= pydicom.uid.ImplicitVRLittleEndian
            ds = Dataset()
            ds.file_meta = meta

            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.SOPInstanceUID = instanceUID_var
            ds.SOPClassUID = pydicom.uid.CTImageStorage 
            ds.PatientName = "NAME^NONE"
            ds.PatientID = "NOID"

            ds.Modality = "CT"
            ds.SeriesInstanceUID = seriesUID
            ds.StudyInstanceUID = studyInstance
            ds.FrameOfReferenceUID = frameOfReference
            ds.SeriesNumber = 3

            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.SamplesPerPixel = 1
            ds.HighBit = 15
            ds.WindowCenter = 30
            ds.WindowWidth = 100

            ds.Rows = image2d.shape[0]
            ds.Columns = image2d.shape[1]
            ds.AcquisitionNumber = 1

            ds.ImageOrientationPatient = r"1\0\0\0\1\0"
            ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

            ds.RescaleIntercept = "0"
            ds.RescaleSlope = "1"
            ds.PixelSpacing = r"0.5\0.5"
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 1
            # ds.RescaleIntercept = "-1024"
            ds.RescaleType = 'HU'
            name = directory + f"/I{i}"
            pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
            thickness = (size_y * 2)/len(HU_images)
            ds.SliceThickness = str(thickness)
            ds.SpacingBetweenSlices = str(thickness)
            ds.ImagePositionPatient = f"0\\0\\{thickness * i}"
            ds.SliceLocation = str(thickness * i)+'00'
            ds.InstanceNumber = i+1
            image2d = image.astype('int16')
            ds.PixelData = image2d.tobytes()
            ds.save_as(name + '.dcm')



















# Deprecated ========================================================================================================================================================

def ModifyRoot(directory, root_name, tree_name, branch_names, output_name, new_tree_name, new_branch_names):

    import uproot; import uproot.writing; import os

    input_file = directory + root_name + '.root'
    with uproot.open(input_file) as file:       
        tree = file[tree_name]
        branches = tree.arrays(branch_names, library="np")
        
    output_file = directory + output_name
    counter = 1
    while True:
        if not os.path.exists(f"{output_file}{counter}.root"):
            output_file = f"{output_file}{counter}.root"
            break
        counter = counter + 1

    with uproot.recreate(output_file) as new_file:
        new_file[new_tree_name] = {new_branch_names[0]: branches[branch_names[0]],
                                   new_branch_names[1]: branches[branch_names[1]]}

def LogaritmicTransformation(radiographs, pixel_size, sigma):
    
    import matplotlib.pyplot as plt; from scipy import ndimage; from tqdm import tqdm

    htmps = np.zeros(len(radiographs), dtype = 'object')

    for i, radiograph in tqdm(enumerate(radiographs), desc = 'Computing logarithmic transformation', unit = ' Heatmaps', leave = True):
        radiograph = ndimage.gaussian_filter(radiograph, sigma)
        maxi = np.max(radiograph)
        htmps[i][htmps[i] == 0] = np.nan
        htmps[i] = np.log(maxi/radiograph) / (pixel_size * 0.1)

    plt.imshow(htmps[-1]); plt.colorbar(); plt.show()

    return htmps

def MergeRoots(directory, starts_with, output_name):

    import uproot; from tqdm import tqdm

    trash_folder, file_list, merged_file = Manage_Files(directory, starts_with, output_name)

    with uproot.recreate(merged_file) as f_out:
        
        for file in tqdm(file_list, desc="Merging ROOT files", unit="file"):
            
            with uproot.open(file) as root_in:
                
                for key in root_in.keys():
                    
                    base_key = key.split(';')[0]  # Obtener el nombre base sin número de ciclo
                    obj = root_in[key]

                    if isinstance(obj, uproot.TTree):

                        for new_data in obj.iterate(library="np", step_size="10 MB"):

                            if base_key in f_out: f_out[base_key].extend(new_data)
                            else: f_out[base_key] = new_data

    print("Archivo final creado en:", merged_file)

def MergeRootsParallel(directory, starts_with, output_name, trim_coords):
    
    import uproot; from tqdm import tqdm; from concurrent.futures import ThreadPoolExecutor; import threading

    max_workers = 100
    
    trash_folder, file_list, merged_file = Manage_Files(directory, starts_with, output_name)

    lock = threading.Lock() # Crear un lock para el acceso a f_out

    with uproot.recreate(merged_file) as f_out:
        
        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            
            futures = [executor.submit(ProcessMerging, file, f_out, lock, trim_coords = trim_coords) for file in file_list]
            for future in tqdm(futures, desc = "Merging ROOT files", unit = "file"): future.result()

    Trash_Folder(trash_folder)
    print("Archivo final creado en:", merged_file)

def ProcessMerging(file, root_out, lock, trim_coords):

    import uproot

    step_size = "50 MB"
    
    with uproot.open(file) as root_in:
        
        for key in root_in.keys():
            base_key = key.split(';')[0]
            obj = root_in[key]

            if base_key == "Hits" and isinstance(obj, uproot.TTree):
                
                for new_data in obj.iterate(["x_ax", "y_ax"], library="np", step_size=step_size):
                    
                    if trim_coords:
                        x_min, x_max, y_min, y_max = trim_coords
                        mask = ((new_data['x_ax'] >= x_min) & (new_data['x_ax'] <= x_max) & (new_data['y_ax'] >= y_min) & (new_data['y_ax'] <= y_max))
                        if mask.sum() == 0: print("No data after filtering. Skipping chunk."); continue
                        new_data = {key: value[mask] for key, value in new_data.items()}
                    
                    with lock: # Lock para asegurar que la escritura en f_out sea thread-safe
                        
                        if base_key in root_out: root_out[base_key].extend(new_data)
                        else: root_out[base_key] = new_data

            elif base_key == "Run Summary" and isinstance(obj, uproot.TTree):
                
                for summary_data in obj.iterate(library="np", step_size=step_size):
                    
                    with lock:
                        
                        if base_key in root_out: root_out[base_key].extend(summary_data)
                        else: root_out[base_key] = summary_data
            
            else: print(f"Skipping unrecognized tree or object: {base_key}")

# ===========================================================================================================================================================================