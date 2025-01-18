# neccesary libraries:
import os, subprocess, numpy as np, pandas as pd, matplotlib.pyplot as plt, uproot, math, platform
from tqdm import tqdm

# 1.1 ========================================================================================================================================================

def directories():

    mac_filename = 'Bisection.mac'

    if platform.system() == "Darwin":
        directory = 'BUILD/'
        executable_file = "Sim"
        run_sim = f"./{executable_file} {mac_filename} . . ."

    elif platform.system() == "Windows":
        directory = f"build/Release"
        executable_file = "Sim.exe"
        run_sim = fr".\{executable_file} .\{mac_filename} . . ."
        print(run_sim)

    elif platform.system() == "Linux":
        directory = f"build/Release"
        executable_file = "Sim.exe"
        run_sim = fr"./{executable_file} {mac_filename} . . ."
        print(run_sim)

    else: raise EnvironmentError("Unsupported operating system")

    return directory, mac_filename, run_sim


def Create_MAC_Template(threads):

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

def SingleEnergyBisection(threads, root_filename, root_structure, initial_energy, thick_0, thick_1, tolerance):

    directory, mac_filename, run_sim = directories()

    tree_name = root_structure[0]
    branch_1 = root_structure[1]
    branch_2 = root_structure[2]

    ratio = 0
    counter = 1
    counter_2 = 0
    beam_count = 200

    while True:

        thickness = (thick_0 + thick_1) / 2

        mac_template = Create_MAC_Template(threads)
    
        mac_filepath = os.path.join(directory, mac_filename)
        mac_content = mac_template.format(energy=initial_energy, thickness=thickness, beam_count=beam_count)
        with open(mac_filepath, 'w') as f: f.write(mac_content)
        try: subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
        except subprocess.CalledProcessError as e: print(f"Error al ejecutar la simulación: {e}")

        file_path = os.path.join(directory, root_filename)
        if not os.path.isfile(file_path): print("Error: El archivo ROOT no existe."); break
        
        try:
            with uproot.open(file_path) as root_file:
                tree = root_file[tree_name]
                if branch_1 not in tree.keys(): print(f"Branch '{branch_1}' not found in tree '{tree_name}' in {file_path}"); continue
                hits_count = tree[branch_1].array(library = "np")[0]

        except Exception as e: print(f"Error al procesar el archivo ROOT: {e}"); continue
        
        ratio = hits_count / beam_count * 100

        if   ratio > 50 + (tolerance / 2): thick_0 = thickness
        elif ratio < 50 - (tolerance / 2): thick_1 = thickness 
        else:
            if counter_2 > 0: break
            if counter_2 == 0:
                beam_count = 10000
                counter_2 = 1

        counter = counter + 1
        if counter == 35: print("No se encontró una solución en el rango especificado."); break

    try:
        with uproot.open(file_path) as root_file:
            tree = root_file[tree_name]
            
            if branch_2 in tree.keys():
                coeficient = tree[branch_2].array(library="np")[0]  # Assuming you want the first entry
                print('Coeficiente de atenuación:', coeficient)
            else: print(f"Branch '{branch_2}' not found in tree '{tree_name}' in {file_path}")
    
    except Exception as e: print(f"Error al procesar el archivo ROOT: {e}")

    print('Ratio:', ratio)
    print("Número de iteraciones:", counter)
    print(f"Optimización completada: Thickness óptimo = {thickness} mm")

# 2.1 ========================================================================================================================================================

def Loop_for_Bisection(threads, 
                        root_path, 
                        output_file, 
                        tolerance,
                        directory, 
                        mac_filename, 
                        run_sim, 
                        tree_name, 
                        branch_1, 
                        branch_2, 
                        energies_vector):
    
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

            mac_template = Create_MAC_Template(threads)

            mac_filepath = os.path.join(directory, mac_filename)
            mac_content = mac_template.format(energy = energy, thickness = thickness, beam_count = beam_count)
            with open(mac_filepath, 'w') as f: f.write(mac_content)

            try: subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
            except subprocess.CalledProcessError as e: print(f"Error al ejecutar la simulación: {e}")

            if not os.path.isfile(root_path): print("Error: El archivo ROOT no existe."); break          
            try:
                with uproot.open(root_path) as root_file:
                    tree = root_file[tree_name]
                    if branch_1 not in tree.keys():
                        print(f"Branch '{branch_1}' not found in tree '{tree_name}' in {root_path}")
                        continue

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


def BisectionEnergiesNIST(threads, root_filename, outputcsv_name, root_structure, input_csv, tolerance):

    directory, mac_filename, run_sim = directories()

    tree_name = root_structure[0]
    branch_1 = root_structure[1]
    branch_2 = root_structure[2]
    
    root_path = os.path.join(directory + 'ROOT/' + root_filename)
    output_file = os.path.join(directory + 'ROOT/' + outputcsv_name)
    input_file  = os.path.join(directory + 'ROOT/' + input_csv)
    energies_table = pd.read_csv(input_file)

    energies_vector = energies_table['Energy']

    Loop_for_Bisection(
                    threads, 
                    root_path,
                    output_file, 
                    tolerance,
                    directory, 
                    mac_filename, 
                    run_sim, 
                    tree_name, 
                    branch_1, 
                    branch_2, 
                    energies_vector
                    )
    
    print('Finished Bisection')

# 2.2 ========================================================================================================================================================

def BisectionFixedEnergyStep(threads, root_filename, output_file, root_structure, energies, tolerance):
    
    directory, mac_filename, run_sim = directories()

    root_path = os.path.join(directory + 'ROOT/' + root_filename)
    output_file = os.path.join(directory + 'ROOT/' + output_file)

    tree_name = root_structure[0]
    branch_1 = root_structure[1]
    branch_2 = root_structure[2]

    initial_energy = energies[0]
    final_energy = energies[1]
    energy_step = energies[2]

    energies_vector = np.arange(initial_energy, final_energy, energy_step)

    Loop_for_Bisection(
                        threads, 
                        root_path, 
                        output_file, 
                        tolerance,
                        directory, 
                        mac_filename, 
                        run_sim, 
                        tree_name, 
                        branch_1, 
                        branch_2, 
                        energies_vector
                        )

    print('Finished Bisection')

# 4.1 ========================================================================================================================================================

def PlotsFormatting():
    
    SIZE_DEFAULT = 16
    SIZE_LARGE = 20

    plt.rc("font",  family = 'Century Expanded')  
    plt.rc("font",  weight = "normal")  
    plt.rc("axes",  titlesize = SIZE_LARGE  )  
    plt.rc("axes",  labelsize = SIZE_LARGE)  
    plt.rc("font",  size      = SIZE_DEFAULT)  
    plt.rc("xtick", labelsize = SIZE_LARGE)  
    plt.rc("ytick", labelsize = SIZE_LARGE)  

# end ========================================================================================================================================================