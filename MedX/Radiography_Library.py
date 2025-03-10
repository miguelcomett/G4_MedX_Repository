# 0.0. ========================================================================================================================================================

import os, sys, time, subprocess, shutil, platform, threading, signal, math
from contextlib import redirect_stdout; from pathlib import Path

try: import numpy as np
except ImportError: print("NumPy is not installed.")

try: import pandas as pd
except ImportError: print("Pandas is not installed.")

try: import matplotlib.pyplot as plt
except ImportError: print("Matplotlib is not installed.")

def Install_Libraries():

    libraries = {
        "numpy"           : None,
        "pandas"          : None,
        "matplotlib"      : None,
        "dask"            : "2024.10.0",  
        "tqdm"            : None,
        "send2trash"      : None,
        "pygame"          : None,
        "ipywidgets"      : None,
        "uproot"          : None,
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

def PlayAlarm():

    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame

    alarm_path = 'Alarm.mp3'

    volume_level = 30
    os.system(f"osascript -e 'set volume output volume {volume_level}'")

    pygame.mixer.init()
    pygame.mixer.music.load(alarm_path)
    pygame.mixer.music.play(loops = -1) 
    time.sleep(5)
    pygame.mixer.music.stop()

def Formatted_Time(elapsed_time):

    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:     formatted_time = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
    elif minutes > 0: formatted_time = f"{int(minutes)}m {seconds:.1f}s"
    else:             formatted_time = f"{seconds:.1f}s"

    return formatted_time

# 0.1 ========================================================================================================================================================

def Compile_Geant4():

    Build_Path = Path('BUILD')
    if not os.path.exists(Build_Path): os.makedirs(Build_Path)
    
    if platform.system() == "Darwin":
        
        cmake = "cmake .."
        make = "make -j8"

    elif platform.system() == "Linux":

        cmake = "cmake .."
        make = "make -j8"
    
    elif platform.system() == "Windows":

        cmake = "cmake .."
        make = "cmake --build . --config Release"

    else: raise EnvironmentError("Unsupported operating system")
    
    print("-> Building Geant4... ", end = "", flush = True)

    if not os.listdir(Build_Path):
        print("Empty directory. Running Cmake... ", end = "", flush = True)
        try: subprocess.run(cmake, cwd = Build_Path, check = True, shell = True, stdout = subprocess.DEVNULL)
        except subprocess.CalledProcessError as error: print(f"Error Running Cmake: {error}"); raise

    try: subprocess.run(make, cwd = Build_Path, check = True, shell = True, stdout = subprocess.DEVNULL)
    except subprocess.CalledProcessError as error: print(f"Error During Compilation: {error}"); raise
    
    print("Built Successfully.")

def Simulation_Setup(executable_file, mac_filename, temp_folder):
        
    from send2trash import send2trash

    Compile_Geant4()
    
    if platform.system() == "Darwin":
        directory = Path('BUILD')
        run_sim = f"./{executable_file} {mac_filename} . . ."

    elif platform.system() == "Linux":
        directory = Path('BUILD')
        run_sim = f"./{executable_file} {mac_filename} . . ."

    elif platform.system() == "Windows":
        directory = Path('BUILD') / 'Release'
        executable_file = f"{executable_file}.exe"
        run_sim = fr".\{executable_file} .\{mac_filename} . . ."

    else: raise EnvironmentError("Unsupported operating system")
    
    root_folder  = directory / "ROOT/"
    mac_filepath = directory / mac_filename

    if temp_folder:
        temp_folder = directory / "ROOT/" / temp_folder
        try: send2trash(temp_folder)
        except: pass
        os.makedirs(temp_folder, exist_ok = True)

    return directory, run_sim, root_folder, mac_filepath, temp_folder

def Button_Main():

    global Button_Action

    def Toggle_Pause(change):
        
        with output_message: 
            output_message.clear_output(wait = False)
            if pause_flag.is_set(): 
                print("Simulation Paused on Next Iteration.")
                pause_flag.clear()
            else: 
                pause_flag.set()

    def Stop_Execution(change):
        
        with output_message:
            output_message.clear_output(wait = False)
            print("Stopping Execution on Next Iteration...")
        
        stop_flag.set()

    def Elapsed_Time():

        elapsed_time = 0

        while True:
            if finished_flag: break
            if pause_flag.is_set():  
                elapsed_time += 1
                formatted_Time = Formatted_Time(elapsed_time)
                elapsed_time_label.value = f"Time Elapsed: {formatted_Time}"
            time.sleep(1)

    def Button_Action():

        from IPython.display import display
        import ipywidgets as widgets

        global pause_flag, stop_flag, output_message, elapsed_time_label

        pause_flag = threading.Event()
        pause_flag.set() 
        stop_flag = threading.Event()  
        output_message = widgets.Output()
        elapsed_time_label = widgets.Label(value="Time Elapsed: 0s")

        button_layout = widgets.Layout(width='170px', height='40px')
        button_style = {'font_weight': 'bold'}
        
        pause_button = widgets.Button(description = "Pause / Resume", button_style = 'info', layout=button_layout, style=button_style)
        pause_button.on_click(Toggle_Pause)

        stop_button = widgets.Button(description = "Stop Execution", button_style = "danger", layout=button_layout, style=button_style)
        stop_button.on_click(Stop_Execution)

        button_box = widgets.HBox([pause_button, stop_button])
        gui_container = widgets.VBox([button_box, output_message], layout=widgets.Layout(margin='10px 0px 0px 0px'))

        display(gui_container, elapsed_time_label)

        timer_thread = threading.Thread(target=Elapsed_Time, daemon=True)
        timer_thread.start()

        stop_flag.clear()

# 0.2 ========================================================================================================================================================

def MAC_Template_AttCoeff(threads):

    mac_template = []

    if threads: mac_template.append(f"/run/numberOfThreads {'{Threads}'}")

    mac_template.extend([
        f"/run/numberOfThreads {'{Threads}'}",
        f"/run/initialize",
        f"/gun/energy {'{energy}'} eV",
        f"/myDetector/ThicknessTarget {'{thickness}'}",
        f"/run/reinitializeGeometry",
        f"/run/beamOn {'{beam_count}'}"
    ])

    return mac_template

def Loop_for_Bisection(threads, root_path, output_file, tolerance, directory, mac_filepath, run_sim, tree_name, branch_1, branch_2, energies_vector):
    
    import tqdm as tqdm, uproot
    
    results = []
    counter_3 = 0
    
    for energy in tqdm(energies_vector, desc = "Mappping Energies", unit = "Energies", leave = True): 

        ratio = 0
        counter_1 = 1
        counter_2 = 0
        counter_4 = 1
        beam_count = 200

        if counter_3 == 1:
            if (energy / previous_energy) < 5: thickness_1 = thickness_1 * 5           
            elif (energy / previous_energy) < 10: thickness_1 = thickness_1 * 10
            else: counter_3 = 0
        
        previous_energy = energy

        kev = energy / 1000
        if counter_3 == 0:
            if kev <= 0.1:                  thickness = 0.0001 * kev
            if kev > 0.1 and kev <= 1:      thickness = 0.0005 * kev
            if kev > 1   and kev <= 10:     thickness = 0.001 * kev
            if kev > 10  and kev <= 100:    thickness = .01 * kev
            if kev > 100:                   thickness = 0.01 * kev

            thickness_0 = thickness / 100
            thickness_1 = thickness * 100

        while True: 
            
            if counter_4 == 1:
                thickness = math.sqrt(thickness_0 * thickness_1)
                counter_4 = 2 
            
            if counter_4 == 2:
                thickness = (thickness_0 + thickness_1) / 2
                counter_4 = 1 

            mac_template = MAC_Template_AttCoeff(threads)

            mac_content = mac_template.format(energy = energy, thickness = thickness, beam_count = beam_count)
            with open(mac_filepath, 'w') as f: f.write(mac_content)

            try: subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
            except subprocess.CalledProcessError as e: print(f"Error al ejecutar la simulación: {e}")

            if not os.path.isfile(root_path): print("Error: El archivo ROOT no existe."); break          
            
            try:
                
                root_file = uproot.open(root_path)
                tree = root_file[tree_name]
                if branch_1 not in tree.keys(): print(f"Branch '{branch_1}' not found in tree '{tree_name}' in {root_path}"); continue

                hits_count = tree[branch_1].array(library="np")[0]  # Assuming you want the first entry

            except Exception as e: print(f"Error al procesar el archivo ROOT: {e}"); continue
            
            ratio = hits_count / beam_count * 100

            if counter_3 == 1:
                if ratio == 0:      thickness_0 = thickness_0 / 10
                elif ratio < 10:    thickness_0 = thickness_0 / 5
                elif ratio == 100:  thickness_1 = thickness_1 * 10
                elif ratio > 90:    thickness_1 = thickness_1 * 5
                
                counter_3 = 0

            if   ratio > (50 + tolerance / 2): thickness_0 = thickness
            elif ratio < (50 - tolerance / 2): thickness_1 = thickness 
            else:
                
                if counter_2 > 0:
                    try:
                        
                        branch2_array = tree[branch_2].array(library="np")
                        
                        if len(branch2_array) > 0:
                            coeficient = branch2_array[0]
                            results.append({'Energy': energy / 1000, 'Optimal_Thickness': thickness, 'AtCoefficient': coeficient})
                            counter_3 = 1
                            break
                        else: print(f"No data in branch '{branch_2}' in tree '{tree_name}' in {root_path}"); break
                    
                    except Exception as e: print(f"Error al procesar el branch '{branch_2}': {e}"); break

                if counter_2 == 0:
                    beam_count = 100000
                    counter_2 = 1

            counter_1 += 1
            if counter_1 == 30:
                print("No se encontró una solución en el rango especificado.")
                print('Thickness:', thickness, 'mm')
                print('Ratio:', ratio, '%')
                break

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)

def BisectionEnergiesNIST(threads, outputcsv_name, root_structure, input_csv, tolerance):

    executable_file = 'Sim'
    root_base_name = 'AttCoeff.root'
    mac_filename = 'Bisection.mac'
    temp_folder = None

    directory, run_sim, root_folder, mac_filepath, temp_folder = Simulation_Setup(executable_file, mac_filename, temp_folder)

    tree_name = root_structure[0]
    branch_1 = root_structure[1]
    branch_2 = root_structure[2]
    
    root_path = os.path.join(directory + 'ROOT/' + root_base_name)
    output_csv = os.path.join(directory + 'ROOT/' + outputcsv_name)
    input_file  = os.path.join(directory + 'ROOT/' + input_csv)
    energies_table = pd.read_csv(input_file)

    energies_vector = energies_table['Energy']

    Loop_for_Bisection(threads, root_path, output_csv, tolerance, directory, mac_filepath, run_sim, tree_name, branch_1, branch_2, energies_vector)
    
    print('Finished Bisection')

def BisectionFixedEnergyStep(threads, output_csv, root_structure, energies, tolerance):

    executable_file = 'Sim'
    root_base_name = 'AttCoeff.root'
    mac_filename = 'Bisection.mac'
    temp_folder = None

    directory, run_sim, root_folder, mac_filepath, temp_folder = Simulation_Setup(executable_file, mac_filename, temp_folder)

    root_path = os.path.join(directory + 'ROOT/' + root_base_name)
    output_csv = os.path.join(directory + 'ROOT/' + output_csv)

    tree_name = root_structure[0]
    branch_1 = root_structure[1]
    branch_2 = root_structure[2]

    initial_energy = energies[0]
    final_energy = energies[1]
    energy_step = energies[2]

    energies_vector = np.arange(initial_energy, final_energy, energy_step)

    Loop_for_Bisection(threads,root_path, output_csv, tolerance, directory, mac_filename, run_sim, tree_name, branch_1, branch_2, energies_vector)

    print('Finished Bisection')

def Plot_Att_Coeff(directory, DATA, title, x_label, y_label, X_axis_log, Y_axis_log, Figure_Text, Font_Size_Normal, Font_Size_Large, save_as):

    plt.figure(figsize = (14, 8))

    plt.rc("font",  family = 'Century Expanded')  
    plt.rc("font",  weight = "normal")  
    plt.rc("axes",  titlesize = Font_Size_Large)  
    plt.rc("axes",  labelsize = Font_Size_Large)  
    plt.rc("font",  size      = Font_Size_Normal)  
    plt.rc("xtick", labelsize = Font_Size_Normal)  
    plt.rc("ytick", labelsize = Font_Size_Normal)  

    for CSV_File in DATA:
        
        file_path = directory + CSV_File["CSV"]
        DataFrame = pd.read_csv(file_path)
        
        plt.plot(DataFrame[CSV_File["X"]], DataFrame[CSV_File["Y"]], label = CSV_File["LABEL"], marker = CSV_File["MARKER"], 
                 markersize = CSV_File["MARKERSIZE"], color = CSV_File["COLOR"], alpha = CSV_File["ALPHA"])

    if X_axis_log == True: plt.xscale("log")
    if Y_axis_log == True: plt.yscale("log")

    plt.title(title, pad = 12); plt.xlabel(x_label, labelpad = 7); plt.ylabel(y_label, labelpad = 8)

    plt.figtext(Figure_Text, fontsize = Font_Size_Normal, bbox = dict(facecolor = 'white', alpha = 0.5))

    plt.legend(); plt.grid(True)
    
    if save_as != None: plt.savefig(f"{directory}/{save_as}", dpi = 600)
    plt.show()

def Merge_CSVs(directory, output_file):
    
    first_file = True
    sorted_filenames = sorted(os.listdir(directory))
    
    with open(output_file, 'w') as outfile:
        
        for filename in sorted_filenames:
            if filename.endswith('.csv') and filename != output_file:
                
                with open(os.path.join(directory, filename), 'r') as file:
                    
                    if first_file:
                        outfile.write(file.read())
                        first_file = False
                    else:
                        next(file)
                        outfile.write(file.read()) 
                    
                    outfile.write('\n')

# 1.1. ========================================================================================================================================================

def Trash_Folder(trash_folder):

    from send2trash import send2trash
                 
    try: send2trash(trash_folder)
    except Exception as e: print(f"Error deleting trash folder: {e}")

def MAC_Template_Radiography(
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

    mac_template.extend([
        f"/myDetector/nColumns {detector_parameters['nColumns']}",
        f"/myDetector/nRows {detector_parameters['nRows']}",
        f"/run/reinitializeGeometry",
        f""
    ])

    if threads: mac_template.append(f"/run/numberOfThreads {'{Threads}'}")
    mac_template.extend([
        f"/run/initialize",
        f""
    ])

    mac_template.extend([
        f"/Pgun/X {gun_parameters['X']} mm",
        f"/Pgun/Y {gun_parameters['Y']} mm",
        f"/Pgun/gaussX {gun_parameters['gaussX']}",
        f"/Pgun/SpanX {gun_parameters['SpanX']} mm",
        f"/Pgun/SpanY {gun_parameters['SpanY']} mm",
        f""
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

def Run_Calibration(directory, run_sim):
    
    print("-> Running Calibration... ", end = "", flush = True)
    start_time = time.perf_counter()
    try: subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
    except subprocess.CalledProcessError as error: print(f"Error running the simulation: {error}"); raise
    
    end_time = time.perf_counter()
    calibration_time = end_time - start_time
    print(f"Finished ", end = '', flush = True)

    return calibration_time

def RunRadiography(threads, energy, sim_time, iteration_time, spectra_mode, detector_parameters, gun_parameters, alarm):

    from tqdm import tqdm

    global finished_flag, start_time
    finished_flag = False

    start_time = time.perf_counter()

    if iteration_time == 0 or iteration_time > sim_time: iteration_time = sim_time

    sim_time = sim_time * 60 # s
    iteration_time = iteration_time * 60 # s 

    executable_file = 'Sim'
    root_base_name = 'CT'
    mac_filename = 'radiography.mac'
    temp_folder = 'Rad_temp'

    directory, run_sim, root_folder, mac_filepath, temp_folder = Simulation_Setup(executable_file, mac_filename, temp_folder)

    simulation_mode = 'single'
    mac_template = MAC_Template_Radiography(simulation_mode, threads, spectra_mode, detector_parameters, gun_parameters)
    
    Beams_calibration = 2_500_000

    filled_template = mac_template.format(Threads = threads, Energy = energy, Beams = Beams_calibration)
    with open(mac_filepath, 'w') as template_file: template_file.write(filled_template)
    
    calibration_geant4_time = Run_Calibration(directory, run_sim)

    if spectra_mode == 'mono' or spectra_mode == 0: 
        new_base_name = 'Rad'
        energy_name = f"{str(energy)}{'kev'}"

    if spectra_mode == '80kvp' or spectra_mode == 1: 
        new_base_name = 'Poly'
        energy_name = spectra_mode

    if spectra_mode == '140kvp' or spectra_mode == 2: 
        new_base_name = 'Poly'
        energy_name = spectra_mode

    old_root_name = root_folder/f"{root_base_name}{'_00.root'}"
    new_root_name = root_folder/f"{new_base_name}{'_0.root'}"

    try: os.rename(old_root_name, new_root_name)
    except OSError as error: print(f"Error renaming {old_root_name}. {error}"); raise
    
    try: shutil.move(new_root_name, temp_folder)
    except OSError as error: print(f"Error moving {new_root_name}. {error}"); raise

    iterations = int(sim_time / iteration_time)
    
    Beams = int((sim_time * Beams_calibration) / (calibration_geant4_time * iterations))
    
    end_time = time.perf_counter()
    calibration_python_time = end_time - start_time
    formatted_time = Formatted_Time(calibration_python_time)

    print(f"({formatted_time}). Beams to simulate: \033[1m{round(Beams * iterations / 1000000, 2)}M.")

    filled_template = mac_template.format(Threads = threads, Energy = energy, Beams = Beams)
    with open(mac_filepath, 'w') as template_file: template_file.write(filled_template)

    Button_Main(); Button_Action(); start_time = time.perf_counter()

    def Radiography_Loop():

        global finished_flag
        
        for iteration in tqdm(range(iterations), desc = "Running Radiography", unit = " Iterations", leave = True):

            if stop_flag.is_set(): 
                finished_flag = True
                break
            while not pause_flag.is_set():
                if stop_flag.is_set(): return
                time.sleep(0.1) 

            try: subprocess.run(run_sim, cwd = directory,check = True, shell = True, stdout = subprocess.DEVNULL)
            except subprocess.CalledProcessError as error: print(f"Error running the simulation: {error}"); raise
            
            new_root_name = root_folder / f"{new_base_name}{'_'}{str(iteration + 1)}{'.root'}"
            
            try: os.rename(old_root_name, new_root_name)
            except OSError as error: print(f"Error renaming {old_root_name}. {error}"); raise
            
            try: shutil.move(new_root_name, temp_folder)
            except OSError as error: print(f"Error moving the file: {error}"); raise

    def Finally():

        global finished_flag

        Simulation_Thread.join()

        total_beams = int(np.ceil(Beams * iterations / 1000000))
        merged_name = f"{new_base_name}{'_'}{energy_name}{'_'}{str(total_beams)}{'M'}"

        if os.path.exists(root_folder / f"{merged_name}{'.root'}"):
            counter = 1
            while os.path.exists(root_folder / f"{merged_name}{'_'}{str(counter)}{'.root'}"): counter = counter + 1
            merged_name = f"{merged_name}{'_'}{str(counter)}"
        merged_name = f"{merged_name}{'.root'}"

        with open(os.devnull, "w") as fnull: 
            with redirect_stdout(fnull): Merge_Roots_HADD(temp_folder, new_base_name, merged_name, trim_coords = None)

        try: shutil.move(temp_folder/merged_name, root_folder)
        except OSError as error: print(f"Error moving the file: {error}"); raise
        Trash_Folder(temp_folder)

        end_time = time.perf_counter()
        loop_time = end_time - start_time

        formatted_time = Formatted_Time(calibration_python_time + loop_time)

        print(f"-> Simulation Completed. Files: \033[1m{merged_name}\033[0m written in \033[1m{root_folder}\033[0m.")
        print(f"   Total Time: {formatted_time}")
        if alarm == True: PlayAlarm()

        finished_flag = True

    Simulation_Thread = threading.Thread(target = Radiography_Loop, daemon = True)
    Simulation_Thread.start()

    finally_thread = threading.Thread(target = Finally, daemon = True)
    finally_thread.start()

# 1.2. ========================================================================================================================================================

def Rename_and_Move(root_folder, temp_folder, iteration, spectra_mode):

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
    except OSError as error: print(f"Error renaming {file_40}. {error}"); raise

    try: os.rename(file_80, new_name_80)
    except OSError as error: print(f"Error renaming {file_80}. {error}"); raise
        
    try: shutil.move(new_name_40, temp_folder)
    except OSError as error: print(f"Error moving the file: {error}"); raise

    try: shutil.move(new_name_80, temp_folder)
    except OSError as error: print(f"Error moving the file: {error}"); raise

    return base_name_40, base_name_80
    
def RunDEXA(threads, sim_time, iteration_time, spectra_mode, detector_parameters, gun_parameters, alarm):

    from tqdm import tqdm

    global finished_flag, start_time
    finished_flag = False
    
    start_time = time.perf_counter()

    if iteration_time == 0 or iteration_time > sim_time: iteration_time = sim_time

    sim_time = sim_time * 60 # s
    iteration_time = iteration_time * 60 # s 

    executable_file = 'Sim'
    mac_filename = 'DEXA.mac'
    temp_folder = 'DEXA_temp'

    directory, run_sim, root_folder, mac_filepath, temp_folder = Simulation_Setup(executable_file, mac_filename, temp_folder)

    simulation_mode = 'DEXA'
    mac_template = MAC_Template_Radiography(simulation_mode, threads, spectra_mode, detector_parameters, gun_parameters)

    if spectra_mode == 'mono' or spectra_mode == 0:
        Beams40_calibration = 2_000_000
        Beams80_calibration = int(Beams40_calibration / 1.61)

    if spectra_mode == 'poly' or spectra_mode == 1:
        Beams40_calibration = 2_000_000
        Beams80_calibration = round(Beams40_calibration / 1.30)

    filled_template = mac_template.format(Threads = threads, Beams40 = Beams40_calibration, Beams80 = Beams80_calibration)
    with open(mac_filepath, 'w') as template_file: template_file.write(filled_template)

    calibration_geant4_time = Run_Calibration(directory, run_sim)
    base_name_40, base_name_80  = Rename_and_Move(root_folder, temp_folder, 0, spectra_mode)
    iterations = int(sim_time / iteration_time)

    Beams40 = int((sim_time * Beams40_calibration) / (calibration_geant4_time * iterations))
    Beams80 = int((sim_time * Beams80_calibration) / (calibration_geant4_time * iterations))

    Beams40_str = f"{round(Beams40 * iterations / 1_000_000, 2)}M"
    Beams80_str = f"{round(Beams80 * iterations / 1_000_000, 2)}M"

    end_time = time.perf_counter()
    calibration_python_time = end_time - start_time
    formatted_time = Formatted_Time(calibration_python_time)
    
    print(f"({formatted_time}) Beams to Simulate: \033[1m{Beams40_str}, {Beams80_str}.")

    filled_template = mac_template.format(Threads = threads, Beams40 = Beams40, Beams80 = Beams80)
    with open(mac_filepath, 'w') as template_file: template_file.write(filled_template)
    
    def DEXA_Loop():

        global finished_flag

        for iteration in tqdm(range(iterations), desc = "Running DEXA", unit = " Iterations", leave = True):

            if stop_flag.is_set(): 
                finished_flag = True
                break
            while not pause_flag.is_set():
                if stop_flag.is_set(): return
                time.sleep(0.1) 

            try: subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
            except subprocess.CalledProcessError as error: print(f"Error running the simulation: {error}"); raise

            Rename_and_Move(root_folder, temp_folder, iteration + 1, spectra_mode)

    def Finally():

        global finished_flag

        Simulation_Thread.join()

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
        with redirect_stdout(fnull): Merge_Roots_HADD(temp_folder, base_name_40, merged_40, trim_coords = None)
        with redirect_stdout(fnull): Merge_Roots_HADD(temp_folder, base_name_80, merged_80, trim_coords = None)
        fnull.close()

        shutil.move(temp_folder/merged_40, root_folder)
        shutil.move(temp_folder/merged_80, root_folder)

        Trash_Folder(temp_folder)

        finished_flag = True

        end_time = time.perf_counter()
        loop_time = end_time - start_time
        formatted_time = Formatted_Time(calibration_python_time + loop_time)

        print(f"-> Simulation Completed. Files: \033[1m{merged_40}\033[0m and \033[1m{merged_80}\033[0m written in \033[1m{root_folder}\033[0m.")
        print(f"   Total Time: {formatted_time}")
        if alarm == True: PlayAlarm()

    Button_Main(); Button_Action(); 
    start_time = time.perf_counter()

    Simulation_Thread = threading.Thread(target = DEXA_Loop, daemon = True)
    Simulation_Thread.start()

    finally_thread = threading.Thread(target = Finally, daemon = True)
    finally_thread.start()

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

def Manage_Merge_Files(directory, starts_with, output_name):

    directory = os.path.join(directory, '')

    trash_folder = f"{directory}Trash_{output_name}"
    os.makedirs(trash_folder, exist_ok = True)

    file_list = []
    for file in os.listdir(directory):
        if file.endswith('.root') and file.startswith(starts_with): 
            file_path  = os.path.join(directory, file)
            shutil.move(file_path, trash_folder)
            file_list.append(os.path.join(trash_folder, file))

    if file_list == []: raise RuntimeError("No files found. Please check your inputs.")

    if output_name.endswith('.root'): output_name = output_name.rstrip('.root')
    merged_file = directory + output_name 
    if not os.path.exists(merged_file + ".root"): merged_file = merged_file + ".root"
    if os.path.exists(merged_file + ".root"):
        counter = 0
        while os.path.exists(f"{merged_file}_{counter}.root"): counter += 1
        merged_file = f"{merged_file}_{counter}.root"

    return trash_folder, file_list, merged_file

def Merge_Roots_HADD(directory, starts_with, output_name, trim_coords):
    
    if trim_coords == None: 

        trash_folder, file_list, merged_file = Manage_Merge_Files(directory, starts_with, output_name)
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

    trash_folder, file_list, merged_file = Manage_Merge_Files(directory, starts_with, output_name)

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

# 1.4. ========================================================================================================================================================

def Summary_Data(directory, root_file, hits_tree, hits_branches, summary_tree, summary_branches, 
                 radiation_tree, radiation_branches, spectra_tree, spectra_branches):

    import uproot, dask.array as da

    directory = os.path.join(directory, '')
    if not root_file.endswith('.root'): root_file = root_file + '.root'
    file_path = directory + root_file
    opened_file = uproot.open(file_path)

    try:
        hits_tree = uproot.dask(opened_file[hits_tree], library = 'np', step_size = '50 MB')
        
        for i in range(len(hits_branches)): 
            if hits_branches[i] not in hits_tree.keys(): 
                print(f"Branch: '{hits_branches[i]}', not found in tree: '{hits_tree}'.")
        
        Number_of_Hits = hits_tree[hits_branches[0]]
        Number_of_Hits = len(Number_of_Hits)
        Number_of_Hits = round(Number_of_Hits/1_000_000, 2)

    except Exception as error: print("\033[31m" + f"Error while processing Root file: \n \n {error}" + "\033[0m \n")
    
    try:
        summary_tree = uproot.dask(opened_file[summary_tree], library = 'np', step_size = '50 MB')
        
        for i in range(len(summary_branches)): 
            if summary_branches[i] not in summary_tree.keys(): 
                print(f"Branch: '{summary_branches[i]}', not found in tree: '{summary_tree}'.")
        
        Number_of_Photons = summary_tree[summary_branches[0]]
        Sample_Mass       = summary_tree[summary_branches[1]]
        Energy_Deposition = summary_tree[summary_branches[2]]
        Radiation_Dose    = summary_tree[summary_branches[3]]

        Number_of_Photons = da.sum(Number_of_Photons).compute()
        Number_of_Photons = round(Number_of_Photons / 1_000_000, 2)
        Sample_Mass       = da.mean(Sample_Mass).compute()
        Energy_Deposition = da.sum(Energy_Deposition).compute()
        Radiation_Dose    = da.sum(Radiation_Dose).compute()

    except Exception as error: print("\033[31m" + f"Error while processing Root file: \n \n {error}" + "\033[0m \n")
        
    try:
        radiation_tree = uproot.dask(opened_file[radiation_tree], library = 'np', step_size = '50 MB')
        
        for i in range(len(radiation_branches)): 
            if radiation_branches[i] not in radiation_tree.keys(): 
                print(f"Branch: '{radiation_branches[i]}', not found in tree: '{radiation_tree}'.")
        
        Tissue_Names = (radiation_tree[radiation_branches[0]]).compute()
        Tissue_Dose  = (radiation_tree[radiation_branches[1]]).compute()

        Unique_Tissues = np.unique(Tissue_Names)
        Tissue_Dose = {organ: Tissue_Dose[Tissue_Names == organ].sum() for organ in Unique_Tissues}

    except Exception as error: print("\033[31m" + f"Error while processing Root file: \n \n {error}" + "\033[0m \n")
    
    try:
        spectra_tree = uproot.dask(opened_file[spectra_tree], library = 'np', step_size = '50 MB')
        
        for i in range(len(spectra_branches)): 
            if spectra_branches[i] not in spectra_tree.keys(): 
                raise ValueError(f"Branch: '{spectra_branches[i]}', not found in tree: '{spectra_tree}'.")
        
        Energy    = spectra_tree[spectra_branches[0]]
        Frequency = spectra_tree[spectra_branches[1]]
        Mean_Energy = da.sum(Energy * Frequency) / da.sum(Frequency)
        Mean_Energy = Mean_Energy.compute()

    except Exception as error: 
        print("\033[31m" + f"Error while processing Root file: \n \n {error}" + "\033[0m \n")
        print(f"Falling back to previous calculation of Mean Energy.")

        Mean_Energy = np.array(summary_tree['Initial_Energy_keV'])
        Mean_Energy = Mean_Energy.mean()

    try: print(f"-> Initial Photons Generated:   \033[1m{Number_of_Photons:,.2f} M   \033[0m")
    except: pass
    try: print(f"-> Mean Energy of Photons:      \033[1m{Mean_Energy:,.2f} keV       \033[0m")
    except: pass
    try: print(f"-> Photon's Hits in Detector:   \033[1m{Number_of_Hits:,.2f} M      \033[0m")
    except: pass
    try: print(f"-> Mass of Sample Scanned:      \033[1m{Sample_Mass:,.3f} kg        \033[0m")
    except: pass
    try: print(f"-> Energy Deposited in Tissues: \033[1m{Energy_Deposition:,.3f} TeV \033[0m \n")
    except: pass
    try: print(f"-> Total Dose of Radiation:     \033[1m{Radiation_Dose:,.5f} µSv    \033[0m")
    except: pass
    try: 
        for organ, dose in Tissue_Dose.items(): print(f"  > Radiation Dose in {organ:>7}:  \033[1m{dose:.5f} µSv \033[0m")
    except: pass

def XY_1D_Histogram(directory, root_file, hits_tree, hits_branches, spectra_tree, spectra_branches, range_x, range_y, range_spectra):

    import uproot, dask.array as dask_da; from dask.diagnostics import ProgressBar

    directory = os.path.join(directory, '')
    if not root_file.endswith('.root'): root_file = root_file + '.root'
    file_path = directory + root_file
    opened_file = uproot.open(file_path)

    plt.figure(figsize = (18, 4)); plt.tight_layout()

    if hits_tree is not None:

        hits_tree = uproot.dask(opened_file[hits_tree], library='np', step_size = '50 MB')

        for i in range(len(hits_branches)): 
            if hits_branches[i] not in hits_tree.keys(): 
                raise ValueError(f"Branch: '{hits_branches[i]}', not found in tree: '{hits_tree}'.")

        x_values = hits_tree[hits_branches[0]]
        y_values = hits_tree[hits_branches[1]]
        
        hist_x, range_x = dask_da.histogram(x_values, bins = range_x[2], range = (range_x[0], range_x[1]))
        hist_y, range_y = dask_da.histogram(y_values, bins = range_y[2], range = (range_y[0], range_y[1]))

        with ProgressBar():
            print('\nComputing X-Dimension Histogram (1/3)...')
            hist_x = hist_x.compute()
            print('Computing Y-Dimension Histogram (2/3)...')
            hist_y = hist_y.compute()

        plt.subplot(1, 3, 1)
        width_x = (range_x[1] - range_x[0])
        plt.bar(range_x[:-1], hist_x, width = width_x, align = 'edge', color = 'blue',  alpha = 0.7, edgecolor = 'gray', linewidth = 0.0)
        plt.xlabel('Distance in X (mm)'); plt.ylabel('Frequency'); plt.title('Hits in X')

        plt.subplot(1, 3, 2)
        width_y = (range_y[1] - range_y[0])
        plt.bar(range_y[:-1], hist_y, width = width_y, align = 'edge', color = 'green', alpha = 0.7, edgecolor = 'gray', linewidth = 0.0)
        plt.xlabel('Distance in Y (mm)'); plt.ylabel('Frequency'); plt.title('Hits in Y')

    if spectra_tree is None:
        
        summary_tree = uproot.dask(opened_file['Run Summary'], library='np', step_size = '50 MB')
        Energy = summary_tree['Initial_Energy_keV']

        hist_z, range_z = dask_da.histogram(Energy, bins = range_spectra[2], range = (range_spectra[0], range_spectra[1])) 
        
        print('Computing Z-Dimension Histogram (3/3)...')
        with ProgressBar(): hist_z = hist_z.compute()
        

    if spectra_tree is not None:

        spectra_tree = uproot.dask(opened_file[spectra_tree], library='np', step_size='50 MB')
        
        for i in range(len(spectra_branches)): 
            if spectra_branches[i] not in spectra_tree.keys(): 
                raise ValueError(f"Branch: '{spectra_branches[i]}', not found in tree: '{spectra_tree}'.")
        
        Energy    = spectra_tree[spectra_branches[0]]
        Frequency = spectra_tree[spectra_branches[1]]
        
        hist_z, range_z = dask_da.histogram(Energy, weights = Frequency, bins = range_spectra[2], range = (range_spectra[0], range_spectra[1])) 

        print('Computing Z-Dimension Histogram (3/3)...')
        with ProgressBar(): hist_z = hist_z.compute()

    plt.subplot(1, 3, 3)
    width_z = (range_z[1] - range_z[0])
    plt.bar(range_z[:-1], hist_z, width = width_z, align = 'edge', color = 'red',   alpha = 0.7, edgecolor = 'gray', linewidth = 0.0)
    plt.xlabel('Energy (keV)'); plt.ylabel('Frequency'); plt.title('Energy Spectrum')
    
# 2.0. ========================================================================================================================================================

def Root_to_Heatmap(directory, root_file, root_structure, dimensions, pixel_size, progress_bar):

    import uproot, dask.array as dask_da; from dask.diagnostics import ProgressBar

    directory = os.path.join(directory, '')

    tree_name = root_structure["tree_name"]
    x_branch = root_structure["x_branch"]
    y_branch = root_structure["y_branch"]

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

    x_data_shifted = x_values + dimensions["shift_X"]            
    y_data_shifted = y_values + dimensions["shift_Y"]

    bins_x0 = np.arange(-dimensions["len_X"], dimensions["len_X"] + pixel_size, pixel_size)
    bins_y0 = np.arange(-dimensions["len_Y"], dimensions["len_Y"] + pixel_size, pixel_size)

    heatmap = dask_da.histogram2d(x_data_shifted, y_data_shifted, bins=[bins_x0, bins_y0], density = True)[0] # , density = True
    
    if progress_bar == True: 
        print('Computing heatmap...')
        with ProgressBar(): heatmap = heatmap.compute()
    else: heatmap = heatmap.compute()
    
    heatmap = np.rot90(heatmap.T, 2)

    return heatmap

def Logarithmic_Transform(heatmap):

    max_values = np.max(heatmap, axis = 0, keepdims = True)
    
    heatmap[heatmap <= 0] = np.nan
    heatmap = np.log(max_values / heatmap)
    heatmap[np.isnan(heatmap)] = 0

    # heatmap = np.where(heatmap==0, 1, 0) # for debugging

    return heatmap

def Plot_Heatmap(heatmap, save_as):

    rows = heatmap.shape[0]
    cols = heatmap.shape[1]

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1); plt.imshow(heatmap, cmap="gray"); plt.colorbar()
    if save_as: plt.savefig(save_as + ".png", bbox_inches = "tight", dpi = 900)
    plt.subplot(1, 3, 2); plt.plot(heatmap[rows//2, :])
    plt.subplot(1, 3, 3); plt.plot(heatmap[:, cols//2])

def Plotly_from_memory(projection, size):

    import plotly.graph_objects as go

    fig = go.Figure(go.Heatmap(z = projection, colorscale = [[0, 'black'], [1, 'white']],))
    fig.update_layout(width = size[0], height = size[1], xaxis = dict(autorange = 'reversed'), yaxis = dict(autorange = 'reversed'))
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

    images = [low_energy_img, high_energy_img, SLS_Bone, SLS_Tissue, SSH_Bone, SSH_Tissue, ACNR_Bone, ACNR_Tissue]

    for projection, save_as in zip(images, save_as):
        plt.imshow(projection, cmap='gray'); plt.axis('off')
        if save_as: plt.savefig(f"{save_in}{save_as}", bbox_inches='tight', dpi=600)
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

def Bone_Mineral_Density(SLS_Bone, SLS_Tissue):

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

def Denoising_Auto_Edge_Detection(path, isArray, sigma_color, sigma_spatial):

    from skimage.restoration import denoise_bilateral; from PIL import Image
    
    if isArray == True: original_image = np.array(path)
    else: original_image = Image.open(path)
    denoised_image = denoise_bilateral(original_image, sigma_color = sigma_color, sigma_spatial = sigma_spatial, channel_axis = None)

    save_as = ''

    plt.figure(figsize = (10, 5))

    plt.subplot(1, 2, 1); plt.imshow(denoised_image, cmap = 'gray')
    plt.title('Denoised Image'); plt.axis('off')
    if save_as != '': plt.savefig('RESULTS/' + save_as + '.png', bbox_inches = 'tight', dpi = 900)

    plt.subplot(1, 2, 2); plt.imshow(original_image, cmap = 'gray')
    plt.title('Original Image'); plt.axis('off')

    plt.show()

    return denoised_image

# 6.2 ========================================================================================================================================================

def Denoising_Window(array, isHann, alpha, save_as, isCrossSection):
    
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

def MAC_Template_CT(
    threads              = None,  # Optional parameter
    spectra_mode         = None,  # Optional parameter:   'mono (1)'   or 'poly (2)'
    detector_parameters  = None,  # Optional parameter
    gun_parameters       = None,  # Optional parameter
):

    if spectra_mode        is None: spectra_mode = 'mono'
    if detector_parameters is None: detector_parameters = {'nColumns': 1, 'nRows': 1}
    if gun_parameters      is None: gun_parameters = {'X': 0, 'gaussX': 'true', 'SpanX': 230, 'SpanY': 0.01}

    mac_template = []

    mac_template.extend([
        f"/myDetector/Rotation {'{angle}'}",
        f"/myDetector/nColumns {detector_parameters['nColumns']}",
        f"/myDetector/nRows {detector_parameters['nRows']}",
        f"/run/reinitializeGeometry",
        f""
    ])

    if threads: mac_template.append(f"/run/numberOfThreads {'{Threads}'}")
    mac_template.extend([
        f"/run/initialize",
        f""
    ])

    mac_template.extend([
        f"/Pgun/X {gun_parameters['X']} mm",
        f"/Pgun/gaussX {gun_parameters['gaussX']}",
        f"/Pgun/SpanX {gun_parameters['SpanX']} mm",
        f"/Pgun/Xcos true",
        f"/Pgun/SpanY {gun_parameters['SpanY']} mm",
        f""
    ])

    if spectra_mode == 'mono' or spectra_mode == 0:
        mac_template.append(f"/gun/energy {'{Energy}'} keV \n")

    if spectra_mode == '80kvp' or spectra_mode == 1:
        mac_template.append(f"/Pgun/Mode 1 \n")

    if spectra_mode == '140kvp' or spectra_mode == 2:
        mac_template.extend(f"/Pgun/Mode 2 \n")

    mac_template.append(f"{'{beam_lines}'}")

    return "\n".join(mac_template)  

def CT_Loop(threads, starts_with, angles_range, slices, gun_parameters, beams_per_line, alarm):

    from tqdm import tqdm

    global finished_flag, start_time
    finished_flag = False

    start_time = time.perf_counter()

    executable_file = 'Sim'
    mac_filename = 'CT.mac'

    directory, run_sim, root_folder, mac_filepath, temp_folder = Simulation_Setup(executable_file, mac_filename, None)

    CT_Folder = directory / 'ROOT' / 'Tomography'
    os.makedirs(CT_Folder, exist_ok = True)

    angles = list(angles_range)

    if len(angles) == 1: print(f"-> Calculating Projection at {angles[0]}° Degrees.")
        
    if len(angles) > 1: 
        
        angle_step = angles[1] - angles[0]

        if angle_step == 1: print(f"-> Calculating Projections at Every Degree.")
        if angle_step  > 1: print(f"-> Calculating Projections Every {angle_step} Degrees")

    Button_Main(); Button_Action()

    def CT_Loop():

        global finished_flag
        global angle
    
        for angle in tqdm(angles_range, desc = "Creating CT", unit = " Angles", leave = True):

            if stop_flag.is_set(): 
                global finished_flag
                break
            while not pause_flag.is_set():
                if stop_flag.is_set(): return
                time.sleep(0.1) 

            for file_name in os.listdir(directory):
                if file_name.startswith('CT_') and file_name.endswith('.root'):
                    file_path = os.path.join(directory, file_name)
                    Trash_Folder(file_path)

            mac_template = MAC_Template_CT(threads, spectra_mode='mono', detector_parameters=None, gun_parameters=gun_parameters)
            
            beam_lines = ""
            for y in range(slices["y0"], slices["y1"] + 1, slices["step"]): 
                beam_lines += f"""
                /Pgun/Y {y} mm
                /run/beamOn {beams_per_line}
                """

            energy = 80

            filled_template = mac_template.format(angle = angle, Threads = threads, Energy = energy, beam_lines = beam_lines)
            with open(mac_filepath, 'w') as mac_file: mac_file.write(filled_template)

            if platform.system() == "Windows":

                subprocess.run(run_sim, cwd = directory, shell = True)

            else:
            
                try: subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
                except subprocess.CalledProcessError as error: print(f"Error during simulation: {error}"); continue  # Skip to the next angle

            output_name = f"Aang_{angle}"
            if os.path.exists(CT_Folder / f"{output_name}.root"):
                counter = 0
                while os.path.exists(CT_Folder / f"{output_name}_{counter}.root"): counter = counter + 1
                output_name = f"{output_name}_{counter}"

            with open(os.devnull, "w") as fnull: 
                with redirect_stdout(fnull): Merge_Roots_HADD(root_folder, starts_with, output_name, trim_coords = None)

            merged_file_path = root_folder / f"{output_name}.root"

            try: shutil.move(merged_file_path, CT_Folder)
            except OSError as error: print(f"Error moving file: {error}"); raise

    def Finally():

        global finished_flag

        Simulation_Thread.join()
        
        end_time = time.perf_counter()
        formatted_Time = Formatted_Time(end_time - start_time)
        print(f"Finished Simulating from {angles[0]} to {angle} angles. Time Elapsed: {formatted_Time}")

        if alarm == True: PlayAlarm()
        finished_flag = True

    Simulation_Thread = threading.Thread(target = CT_Loop, daemon = True)
    Simulation_Thread.start()

    finally_thread = threading.Thread(target = Finally, daemon = True)
    finally_thread.start()

def CT_Summary_Data(directory, summary_tree_name, summary_branches, spectra_tree_name, spectra_branches):

    import uproot, dask.array as da

    NumberofPhotons = Mean_Energy_Sum = EnergyDeposition = RadiationDose = num_of_files = 0

    for file in os.listdir(directory):

        file_path = os.path.join(directory, file)
        if not file_path.endswith('.root'): continue
        opened_file = uproot.open(file_path)

        if summary_tree_name is not None:

            summary_tree = opened_file[summary_tree_name]

            for i in range(len(summary_branches)):
                if summary_branches[i] not in summary_tree.keys(): 
                    raise ValueError(f"Branch: '{summary_branches[i]}', not found in tree: '{summary_tree_name}'.")

            NumberofPhotons  += (np.array(summary_tree[summary_branches[0]])).sum()
            EnergyDeposition += (np.array(summary_tree[summary_branches[1]])).sum()
            RadiationDose    += (np.array(summary_tree[summary_branches[2]])).sum()

        if spectra_tree_name is not None:

            spectra_tree = uproot.dask(opened_file[spectra_tree_name], library='np', step_size='50 MB')
            
            for i in range(len(spectra_branches)): 
                if spectra_branches[i] not in spectra_tree.keys(): 
                    raise ValueError(f"Branch: '{spectra_branches[i]}', not found in tree: '{spectra_tree_name}'.")
            
            Energy    = spectra_tree[spectra_branches[0]]
            Frequency = spectra_tree[spectra_branches[1]]

            Mean_Energy = da.sum(Energy * Frequency) / da.sum(Frequency)
            Mean_Energy_Sum = Mean_Energy.compute()
        
        num_of_files += 1

    NumberofPhotons = round(NumberofPhotons/1_0000_000, 2)
    Mean_Energy = np.mean(Mean_Energy_Sum)
    EnergyDeposition = round(EnergyDeposition, 3)
    RadiationDose = round(RadiationDose, 5)

    print(f"Files processed: {num_of_files}")
    print(f"-> Initial Photons in CT:      \033[1m{NumberofPhotons:,.2f} M    \033[0m")
    if spectra_tree_name is not None: print(f"-> Mean Energy of Photons:     \033[1m {Mean_Energy:,.2f} keV      \033[0m")
    print(f"-> Energy Deposited in Tissue: \033[1m{EnergyDeposition:,.3f} TeV \033[0m")
    print(f"-> Dose of Radiation Received: \033[1m{RadiationDose:,.5f} uSv    \033[0m")

def Calculate_Projections(directory, filename, degrees, root_structure, dimensions, pixel_size, gun_span, csv_folder):
    
    import dask; from dask import delayed; from dask.diagnostics import ProgressBar

    projections = np.arange(degrees["start"], degrees["end"]+1, degrees["step"])

    directory = os.path.join(directory, '')
    raw_folder = "0_Raw_Projections/"
    write_folder = csv_folder + raw_folder
    file_name = "CT_Raw_"
    
    os.makedirs(csv_folder, exist_ok = True)
    os.makedirs(write_folder, exist_ok = True)

    @delayed
    def calculate_heatmaps(i, directory, root_structure, dimensions, pixel_size, gun_span, csv_folder):
        
        root_name = f"{filename}_{i}.root"
        heatmap = Root_to_Heatmap(directory, root_name, root_structure, dimensions, pixel_size, progress_bar=False)

        if gun_span is not None:

            x_length = heatmap.shape[1]

            # print("x_length: ", heatmap.shape)
            # print(gun_span/pixel_size)
            
            if i > 90 and i < 270:
            
               theta = (i-180) * (2*np.pi / 360)

               min = int( x_length/2 - (gun_span*np.cos(theta/2) / pixel_size) )
               max = int( x_length/2 + (gun_span*np.cos(theta/2) / pixel_size) )

            #    print("theta: ", i, " min: ", min, " max: ", max)

            else:

                min = int( x_length/2 - (gun_span / pixel_size) )
                max = int( x_length/2 + (gun_span / pixel_size) )

            heatmap[:, : min ] = 0
            heatmap[:, max : ] = 0

        write_path = write_folder + f"{file_name}{i}.csv"
        np.savetxt(write_path, heatmap, delimiter=',', fmt='%.6f')

        return heatmap

    heatmap_tasks = []
    for i in projections: 
        heatmap_tasks += [calculate_heatmaps(i, directory, root_structure, dimensions, pixel_size, gun_span, csv_folder)]
    print('Calculating Heatmaps for Every Angle in CT:')
    # with ProgressBar(): dask.compute(*heatmap_tasks, scheduler='processes')
    dask.compute(*heatmap_tasks, scheduler='processes')

    del heatmap_tasks

    Plotly_from_file(directory = write_folder, filename = f"{file_name}{degrees["start"]}.csv", size = [500, 500])

def RadonReconstruction(csv_read, csv_write, degrees, slices_in, slices_out, sigma, write):

    import dask; from dask.diagnostics import ProgressBar; from dask import delayed
    from skimage.transform import iradon; from scipy import ndimage

    projections = np.arange(degrees["start"], degrees["end"]+1, degrees["step"])

    csv_write = os.path.join(csv_write, '')
    raw_folder = '0_Raw_Projections/'
    raw_file = 'CT_Raw_'
    read_name = f"{csv_read}{raw_folder}{raw_file}{degrees["start"]}.csv"

    sample_heatmap = pd.read_csv(read_name, delimiter=',', header=None)
    rows = sample_heatmap.shape[0]

    slices_num = len(np.arange(slices_in["initial"], slices_in["final"], slices_in["step"]))
    proportion = int(round(rows / slices_num)) # proportion = int(rows / slices_num)

    scaled_slices = {key: value * proportion for key, value in slices_out.items()}
    slices_vector = np.arange(scaled_slices["initial"], scaled_slices["final"], scaled_slices["step"])
    slices_vector = np.round(slices_vector).astype(int)

    heatmap_matrix  = np.zeros(len(projections),   dtype = object)
    sinogram_matrix = np.zeros(len(slices_vector), dtype = object)

    if write == True:

        heatmap_folder  = "/1_Heatmaps/"
        sinogram_folder = "/2_Sinograms/"
        slices_folder   = "/3_Slices/"
        
        os.makedirs(csv_write, exist_ok = True)
        os.makedirs(csv_write + heatmap_folder,  exist_ok = True)
        os.makedirs(csv_write + sinogram_folder, exist_ok = True)
        os.makedirs(csv_write + slices_folder,   exist_ok = True)

        decimals = '%6f'

    @delayed
    def process_heatmap(i, csv_read, sigma):
        
        read_name = f"{csv_read}{raw_folder}{raw_file}{i}.csv"
        raw_heatmap = np.genfromtxt(read_name, delimiter=',')
        
        raw_heatmap = ndimage.gaussian_filter(raw_heatmap, sigma)
        heatmap = Logarithmic_Transform(raw_heatmap)

        if write == True:

            heatmap[np.isnan(heatmap)] = 0
            heatmap[heatmap < 0] = 0

            write_name = f"{csv_write}{heatmap_folder}{'Projection_'}{i}.csv"
            np.savetxt(write_name, heatmap, delimiter = ',', fmt = decimals)

        return heatmap

    @delayed
    def compute_sinogram(i, y, heatmap_matrix):
        
        sinogram = []
        for heatmap in heatmap_matrix: sinogram.append(heatmap[y])
        sinogram = np.array(sinogram).T

        if write == True:

            sinogram[np.isnan(sinogram)] = 0
            sinogram[sinogram < 0] = 0

            write_name = f"{csv_write}{sinogram_folder}{'Sinogram_'}{i}.csv"
            np.savetxt(write_name, sinogram, delimiter = ',', fmt = decimals)

        return sinogram

    @delayed
    def reconstruct_slice(i, sinogram_matrix, projections):
        
        sinogram = sinogram_matrix[i]
        # slice = iradon(sinogram, theta=projections, circle=True, preserve_range=True, filter_name='hamming')
        slice = iradon(sinogram, theta=projections)

        if write == True:

            slice[np.isnan(slice)] = 0
            slice[slice < 0] = 0

            write_name = f"{csv_write}{slices_folder}{'Slice_'}{i}.csv"
            np.savetxt(write_name, slice, delimiter = ',', fmt = decimals)
    
    heatmap_tasks = []
    for i in projections: heatmap_tasks = heatmap_tasks + [process_heatmap(i, csv_read, sigma)]
    print('Reading and Performing Logarithmic Transform (1/3)...')
    with ProgressBar(): heatmaps = dask.compute(*heatmap_tasks, scheduler='processes')
    heatmap_matrix = np.stack(heatmaps, axis=0)
    # print('Heatmap Matrix Shape:', heatmap_matrix.shape)

    sinogram_tasks = []
    for i, y in enumerate(slices_vector): sinogram_tasks = sinogram_tasks + [compute_sinogram(i, y, heatmap_matrix)]
    print('\nComputing Sinograms (2/3)...')
    with ProgressBar(): sinograms = dask.compute(*sinogram_tasks)    
    sinogram_matrix = np.stack(sinograms, axis=0)
    # print('Sinogram Matrix Shape:', sinogram_matrix.shape)

    slices_tasks = []
    for i in range(len(slices_vector)): slices_tasks = slices_tasks + [reconstruct_slice(i, sinogram_matrix, projections)]
    print('\nReconstructing slices (3/3)...')
    with ProgressBar(): dask.compute(*slices_tasks)
    print()

    if write == True: 
        
        print(f"Succesfully Written Data to: {csv_write}")
        Plotly_3x1(csv_write, step=int(len(slices_vector)/3))

    del heatmap_matrix, sinogram_matrix

def CoefficientstoHU(slices, csv_slices, constants, constant_factor, linear_factor, percentile):

    from tqdm.notebook import tqdm

    valid_slices = np.arange(slices["start"], slices["end"]+1, slices["step"])
    base_name = "Slice"

    csv_slices = os.path.join(csv_slices, '') + "3_Slices/"

    print(csv_slices)

    selected_slices = []
    for i in valid_slices:
        if f"{base_name}_{i}.csv" in os.listdir(csv_slices):
            selected_slices.append(f"{csv_slices}{base_name}_{i}.csv")

    HU_images = np.zeros(len(selected_slices), dtype="object")

    for i in tqdm(range(len(selected_slices)), desc = "Converting to HU Units", unit = " Slices", leave = True):
        
        try: slice = np.genfromtxt(selected_slices[i], delimiter=',')
        except Exception as error: print(f"Error processing {selected_slices[i]}: {error}")

        slice = slice + constant_factor
        slice = slice * linear_factor

        # slice = 1000 * (slice - constants["µ_water"]) / (constants["µ_water"] - constants["µ_air"])
        # slice = np.round(slice)
        # slice = slice.astype(int)
        
        slice[slice < constants["air_tolerance"]] = -1000
        
        positive_values = slice[slice > 0]
        if positive_values.size > 0: 
            threshold = np.percentile(positive_values, percentile)
            slice[slice > threshold] = slice.min() #-1000 

        # if slice.max() < 0: print(f"Slice {i} maximum value ({slice.max()}) is lower than 0")
        # if slice.max() > 3000: print(f"Slice {i} maximum value ({slice.max()}) is higher than 3,000")
        # if slice.min() > -1000: print(f"Slice {i} minimum value ({slice.min()}) is higher than -1,000")

        HU_images[i] = slice

    Plotly_from_memory(HU_images[int(i/2)], size=[500, 500])

    return HU_images

def Export_to_Dicom(HU_images, slice_thickness, slice_spacing, directory, compressed):

    import pydicom; from pydicom.uid import RLELossless; from pydicom.encaps import encapsulate; from pydicom.dataset import Dataset
    from tqdm.notebook import tqdm

    def Create_DICOMs(image, i, seriesUID, studyInstance, frameOfReference, slice_thickness, slice_spacing, directory, compressed):

        meta = Dataset()
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        instanceUID_var = pydicom.uid.generate_uid()
        meta.MediaStorageSOPInstanceUID = instanceUID_var
        meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPInstanceUID = instanceUID_var
        ds.SOPClassUID = pydicom.uid.CTImageStorage
        ds.PatientName = "NAME^NONE"
        ds.PatientID = "NOID"
        ds.Modality = "CT"
        ds.SeriesInstanceUID = seriesUID
        ds.StudyInstanceUID = studyInstance
        ds.FrameOfReferenceUID = frameOfReference
        ds.SeriesNumber = 3
        ds.BitsStored = ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.HighBit = 15
        ds.WindowCenter = 30
        ds.WindowWidth = 100
        ds.Rows = image.shape[0]
        ds.Columns = image.shape[1]
        ds.AcquisitionNumber = 1
        ds.ImageOrientationPatient = "1\\0\\0\\0\\1\\0"
        ds.ImageType = "ORIGINAL\\PRIMARY\\AXIAL"
        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"
        ds.PixelSpacing = f"{0.5:.6g}\\{0.5:.6g}"
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 1
        ds.RescaleType = 'HU'

        ds.SliceThickness = str(slice_thickness)
        ds.SpacingBetweenSlices = str(slice_spacing + slice_thickness)
        ds.ImagePositionPatient = f"{0:.6g}\\{0:.6g}\\{slice_spacing * i:.6g}"
        ds.SliceLocation = f"{slice_spacing * i:.6g}"
        
        ds.InstanceNumber = i + 1
        
        image2d = image.astype('int16')
        ds.PixelData = encapsulate([image2d.tobytes()]) if compressed else image2d.tobytes()
        
        if compressed: ds.compress(RLELossless)
    
        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
        ds.save_as(os.path.join(directory, f"I{i}.dcm"))
    
    os.makedirs(directory, exist_ok=True)
    
    seriesUID, studyInstance, frameOfReference = (pydicom.uid.generate_uid() for _ in range(3))
    
    for i, image in enumerate(tqdm(HU_images, desc = "Saving DICOMS", unit = " Slices", leave = True)):
        Create_DICOMs(image, i, seriesUID, studyInstance, frameOfReference, slice_thickness, slice_spacing, directory, compressed)
    
    print(f"Written DICOMS to {directory}")

# CT Plots ====================================================================================================================================================

def Plotly_from_file(directory, filename, size):

    import plotly.graph_objects as go

    projection = np.genfromtxt(f"{directory}/{filename}", delimiter=',')

    lower = np.percentile(projection, 0)
    upper = np.percentile(projection, 95)
    projection = np.clip(projection, lower, upper)

    fig = go.Figure(go.Heatmap(z = projection, colorscale = [[0, 'black'], [1, 'white']],))
    fig.update_layout(
        width = size[0], height = size[1], xaxis = dict(autorange = 'reversed'), yaxis = dict(autorange = 'reversed'),
        annotations = [dict(text = f"Dimensions: {projection.shape} px", x = 0.5, y = -0.15, showarrow=False, xref="paper", yref="paper", font=dict(size=15))]
    )
    fig.show()

def Plotly_3x1(csv_folder, step):

    import plotly.graph_objects as go; from plotly.subplots import make_subplots

    heatmap_folder  = csv_folder + '1_Heatmaps/'
    sinogram_folder = csv_folder + '2_Sinograms/'
    slices_folder   = csv_folder + '3_Slices/'

    heatmap_files = [f for f in os.listdir(heatmap_folder) if os.path.isfile(os.path.join(heatmap_folder, f))]
    slices_files  = [f for f in os.listdir(slices_folder)  if os.path.isfile(os.path.join(slices_folder, f))]

    number_of_projections = len(heatmap_files)
    number_of_slices = len(slices_files)

    i = number_of_projections / number_of_slices
    plotis = range(0, number_of_slices, int(step))

    for index in plotis:

        deg_id = int(index * i)
        
        heatmap  = np.genfromtxt(heatmap_folder  + f'Projection_{deg_id}.csv', delimiter=',')
        sinogram = np.genfromtxt(sinogram_folder + f'Sinogram_{index}.csv',    delimiter=',')
        slice    = np.genfromtxt(slices_folder   + f'Slice_{index}.csv',       delimiter=',')

        lower = np.percentile(heatmap, 0)
        upper = np.percentile(heatmap, 95)
        heatmap = np.clip(heatmap, lower, upper)

        width = sinogram.shape[1]
        if width < 360: sinogram = np.pad(sinogram, ((0, 0), (0, 360-width)), mode='constant', constant_values=0)      

        fig = make_subplots(rows = 1, cols = 3, shared_xaxes = False, shared_yaxes = False, horizontal_spacing = 0.05 )
        
        fig.add_trace(go.Heatmap(z = heatmap,  colorscale = [[0, 'black'], [1, 'white']], showscale = False), row = 1, col = 1)
        fig.add_trace(go.Heatmap(z = sinogram, colorscale = [[0, 'black'], [1, 'white']], showscale = False), row = 1, col = 2)
        fig.add_trace(go.Heatmap(z = slice,    colorscale = [[0, 'black'], [1, 'white']], showscale = False), row = 1, col = 3)
        
        fig.update_layout(
            
            height = 400, width = 1150, margin = dict(l = 40, r = 40, t = 60, b = 60),
            yaxis1 = dict(autorange = 'reversed'), yaxis2 = dict(autorange = 'reversed'), yaxis3 = dict(autorange = 'reversed'),
            annotations = [
            dict(text = "Radiograph Projection",    x = 0.070, y = 1.12, showarrow=False, xref="paper", yref="paper", font=dict(size=15)),
            dict(text = "Sinogram Slice",           x = 0.500, y = 1.12, showarrow=False, xref="paper", yref="paper", font=dict(size=15)),
            dict(text = "Reconstructed Slice",      x = 0.925, y = 1.12, showarrow=False, xref="paper", yref="paper", font=dict(size=15)),
            dict(text = f"Projection: {deg_id+1}",  x = 0.110, y =-0.15, showarrow=False, xref="paper", yref="paper", font=dict(size=14)),
            dict(text = f"Slice: {index+1}",        x = 0.700, y =-0.15, showarrow=False, xref="paper", yref="paper", font=dict(size=14))]
        )
        
        fig.show()

# end ========================================================================================================================================================