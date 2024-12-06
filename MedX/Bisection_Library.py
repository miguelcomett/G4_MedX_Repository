# 1.1 ========================================================================================================================================================

def SingleEnergyBisection(directory, mac_filename, root_filename, tree_name, branch_1, branch_2, initial_energy, thick_0, thick_1, tolerance):
    
    import os; import subprocess; import uproot

    executable_file = "Sim"
    run_sim = f"./{executable_file} {mac_filename} . ."

    ratio = 0
    counter = 1
    counter_2 = 0
    beam_count = 200

    while True:

        thickness = (thick_0 + thick_1) / 2

        mac_template = """\
        /run/numberOfThreads 10
        /run/initialize
        /gun/energy {energy} eV
        /myDetector/ThicknessTarget {thickness:.10f}
        /run/reinitializeGeometry
        /run/beamOn {beam_count}
        """
        
        # create_mac_file
        mac_filepath = os.path.join(directory, mac_filename)
        mac_content = mac_template.format(energy=initial_energy, thickness=thickness, beam_count=beam_count)
        with open(mac_filepath, 'w') as f:
            f.write(mac_content)

        # run_simulation
        try:
            subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
        except subprocess.CalledProcessError as e: 
            print(f"Error al ejecutar la simulación: {e}")

        file_path = os.path.join(directory, root_filename)
        if not os.path.isfile(file_path):
            print("Error: El archivo ROOT no existe.")
            break
        
        try:
            with uproot.open(file_path) as root_file:
                tree = root_file[tree_name]
                
                if branch_1 not in tree.keys():
                    print(f"Branch '{branch_1}' not found in tree '{tree_name}' in {file_path}")
                    continue

                hits_count = tree[branch_1].array(library = "np")[0]

        except Exception as e:
            print(f"Error al procesar el archivo ROOT: {e}")
            continue
        
        ratio = hits_count / beam_count * 100

        if   ratio > 50 + (tolerance / 2):
            thick_0 = thickness
        elif ratio < 50 - (tolerance / 2):
            thick_1 = thickness 
        else:
            if counter_2 > 0:
                break
            if counter_2 == 0:
                beam_count = 10000
                counter_2 = 1

        counter = counter + 1
        if counter == 35:
            print("No se encontró una solución en el rango especificado.")
            break

    try:
        with uproot.open(file_path) as root_file:
            tree = root_file[tree_name]
            
            if branch_2 in tree.keys():
                coeficient = tree[branch_2].array(library="np")[0]  # Assuming you want the first entry
                print('Coeficiente de atenuación:', coeficient)
            else:
                print(f"Branch '{branch_2}' not found in tree '{tree_name}' in {file_path}")
    
    except Exception as e:
        print(f"Error al procesar el archivo ROOT: {e}")

    print('Ratio:', ratio)
    print("Número de iteraciones:", counter)
    print(f"Optimización completada: Thickness óptimo = {thickness} mm")

# 2.1 ========================================================================================================================================================

def BisectionEnergiesNIST(directory, mac_filename, root_filename, outputcsv_name, tree_name, branch_1, branch_2, input_csv, tolerance):
    
    import os
    import subprocess
    import pandas as pd
    import uproot
    import math
    from tqdm import tqdm

    executable_file = "Sim"
    run_sim = f"./{executable_file} {mac_filename} ."
    
    output_file = os.path.join(directory + 'ROOT/' + outputcsv_name)
    input_file  = os.path.join(directory + 'ROOT/' + input_csv)
    energies_table = pd.read_csv(input_file)

    results = []
    counter_3 = 0

    # first_third_energies = energies_table['Energy'][300:301]
    # for energy in tqdm(first_third_energies, desc = "Mapeando", unit = "Energía", leave = True): 
    
    for energy in tqdm(energies_table['Energy'], desc = "Mapeando", unit = "Energía", leave = True): 
    
        # print(f"Processing energy: {energy}")
        energy = energy * 1000

        ratio = 0
        counter_1 = 1
        counter_2 = 0
        counter_4 = 1
        beam_count = 200

        if counter_3 == 1:
            if (energy / previous_energy) < 5:
                thickness_1 = thickness_1 * 5           
            if (energy / previous_energy) < 10:
                thickness_1 = thickness_1 * 10
            else:
                counter_3 = 0
        
        previous_energy = energy

        kev = energy / 1000
        if counter_3 == 0:
            if kev <= 0.1:
                thickness = 0.0001 * kev
            if kev > 0.1 and kev <= 1:
                thickness = 0.0005 * kev
            if kev > 1 and kev <= 10:
                thickness = 0.001 * kev
            if kev > 10 and kev <= 100:
                thickness = 0.05 * kev
            if kev > 100:
                thickness = 0.01 * kev

            thickness_0 = thickness / 100
            thickness_1 = thickness * 100

        # print('Thickness inicial:', thickness)
        # print('energy:', energy)

        while True: 
            
            if counter_4 == 1:
                thickness = math.sqrt(thickness_0 * thickness_1)
                counter_4 = 2 
            
            if counter_4 == 2:
                thickness = (thickness_0 + thickness_1) / 2
                counter_4 = 1 

            mac_template = """\
            /run/numberOfThreads 1
            /run/initialize
            /gun/energy {energy} eV
            /myDetector/ThicknessTarget {thickness:.12f}
            /run/reinitializeGeometry
            /run/beamOn {beam_count}
            """

            mac_filepath = os.path.join(directory, mac_filename)
            mac_content = mac_template.format(energy = energy, thickness = thickness, beam_count = beam_count)
            with open(mac_filepath, 'w') as f:
                f.write(mac_content)

            try:
                subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
            
            except subprocess.CalledProcessError as e: 
                print(f"Error al ejecutar la simulación: {e}")

            file_path = os.path.join(directory, root_filename)
            if not os.path.isfile(file_path):
                print("Error: El archivo ROOT no existe.")
                break          
            try:
                with uproot.open(file_path) as root_file:
                    tree = root_file[tree_name]
                    if branch_1 not in tree.keys():
                        print(f"Branch '{branch_1}' not found in tree '{tree_name}' in {file_path}")
                        continue

                    hits_count = tree[branch_1].array(library="np")[0]  # Assuming you want the first entry

            except Exception as e:
                print(f"Error al procesar el archivo ROOT: {e}")
                continue
            
            ratio = hits_count / beam_count * 100

            if counter_3 == 1:
                if ratio == 0:
                    thickness_0 = thickness_0 / 10
                elif ratio < 10:
                    thickness_0 = thickness_0 / 5
                elif ratio == 100:
                    thickness_1 = thickness_1 * 10
                    print('100:', thickness_1)
                elif ratio > 90:
                    thickness_1 = thickness_1 * 5
                    print('90:', thickness_1)
                
                counter_3 = 0

            if   ratio > 50 + (tolerance / 2):
                thickness_0 = thickness
            elif ratio < 50 - (tolerance / 2):
                thickness_1 = thickness 
            else:
                if counter_2 > 0:
                    try:
                        branch2_array = tree[branch_2].array(library="np")
                        if len(branch2_array) > 0:
                            coeficient = branch2_array[0]
                            results.append({'Energy': energy / 1000, 'Optimal_Thickness': thickness, 'AtCoefficient': coeficient})
                            
                            # print('Energía calculada:', energy/1000, 'keV')
                            # print('Ratio final:', ratio, '%')
                            # print('Iteraciones:', counter_1)
                            # print('Thickness final:', thickness, 'mm')
                            counter_3 = 1
                            break
                        
                        else:
                            print(f"No data in branch '{branch_2}' in tree '{tree_name}' in {file_path}")
                            break
                    except Exception as e:
                        print(f"Error al procesar el branch '{branch_2}': {e}")
                    break

                if counter_2 == 0:
                    beam_count = 100000
                    counter_2 = 1

            counter_1 += 1
            if counter_1 == 100:
                print("No se encontró una solución en el rango especificado.")
                print('Thickness:', thickness, 'mm')
                print('Ratio:', ratio, '%')
                # print(counter_1)
                break

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index = False)

# 2.2 ========================================================================================================================================================

def BisectionFixedEnergyStep(directory, mac_filename, root_filename, outputcsv_name, tree_name, 
             branch_1, branch_2, initial_energy, final_energy, energy_step, tolerance):
    
    import os
    import subprocess
    import pandas as pd
    import uproot
    import math
    from tqdm import tqdm
    
    executable_file = "Sim"
    run_sim = f"./{executable_file} {mac_filename} . "
    output_file = outputcsv_name

    results = []
    counter_3 = 0

    for energy in tqdm(range(initial_energy, final_energy, energy_step)): 

        ratio = 0
        counter_1 = 1
        counter_2 = 0
        counter_4 = 1
        beam_count = 200

        if counter_3 == 1:
            if (energy / (energy - energy_step)) < 5:
                thickness_1 = thickness_1 * 5
            if (energy / (energy - energy_step)) < 10:
                thickness_1 = thickness_1 * 10
            else:
                counter_3 = 0

        kev = energy / 1000
        if counter_3 == 0:
            if kev <= 0.1:
                thickness = 0.0001 * kev
            if kev > 0.1 and kev <= 1:
                thickness = 0.0005 * kev
            if kev > 1 and kev <= 10:
                thickness = 0.001 * kev
            if kev > 10 and kev <= 100:
                thickness = .01 * kev
            if kev > 100:
                thickness = 0.01 * kev

            thickness_0 = thickness / 100
            thickness_1 = thickness * 100

        # print('t0', thickness_0)
        # print('t1', thickness_1)

        while True: 
            
            if counter_4 == 1:
                thickness = math.sqrt(thickness_0 * thickness_1)
                counter_4 = 2 
            
            if counter_4 == 2:
                thickness = (thickness_0 + thickness_1) / 2
                counter_4 = 1 

            # print('Thickness:', thickness)

            mac_template = """\
            /run/numberOfThreads 1
            /run/initialize
            /gun/energy {energy} eV
            /myDetector/ThicknessTarget {thickness:.12f}
            /run/reinitializeGeometry
            /run/beamOn {beam_count}
            """

            mac_filepath = os.path.join(directory, mac_filename)
            mac_content = mac_template.format(energy = energy, thickness = thickness, beam_count = beam_count)
            with open(mac_filepath, 'w') as f:
                f.write(mac_content)

            try:
                subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
            
            except subprocess.CalledProcessError as e: 
                print(f"Error al ejecutar la simulación: {e}")

            file_path = os.path.join(directory, root_filename)
            if not os.path.isfile(file_path):
                print("Error: El archivo ROOT no existe.")
                break          
            try:
                with uproot.open(file_path) as root_file:
                    tree = root_file[tree_name]
                    if branch_1 not in tree.keys():
                        print(f"Branch '{branch_1}' not found in tree '{tree_name}' in {file_path}")
                        continue

                    hits_count = tree[branch_1].array(library="np")[0]  # Assuming you want the first entry

            except Exception as e:
                print(f"Error al procesar el archivo ROOT: {e}")
                continue
            
            ratio = hits_count / beam_count * 100

            # print('ratio:', ratio, '%')

            if counter_3 == 1:
                if ratio == 0:
                    thickness_0 = thickness_0 / 10
                elif ratio < 10:
                    thickness_0 = thickness_0 / 5
                elif ratio == 100:
                    thickness_1 = thickness_1 * 10
                    # print('100:', thickness_1)
                elif ratio > 90:
                    thickness_1 = thickness_1 * 5
                    # print('90:', thickness_1)
                
                counter_3 = 0

            if   ratio > (50 + tolerance / 2):
                thickness_0 = thickness
            elif ratio < (50 - tolerance / 2):
                thickness_1 = thickness 
            else:
                # print('in tolerance')
                if counter_2 > 0:
                    try:
                        branch2_array = tree[branch_2].array(library="np")
                        if len(branch2_array) > 0:
                            coeficient = branch2_array[0]
                            results.append({'Energy': energy / 1000, 'Optimal_Thickness': thickness, 'AtCoefficient': coeficient})

                            # print(results)
                            
                            # print('Energía calculada:', energy/1000, 'keV')
                            # print('Ratio final:', ratio, '%')
                            # print('Iteraciones:', counter_1)
                            # print('Thickness final:', thickness, 'mm')

                            counter_3 = 1
                            break
                        
                        else:
                            print(f"No data in branch '{branch_2}' in tree '{tree_name}' in {file_path}")
                            break
                    except Exception as e:
                        print(f"Error al procesar el branch '{branch_2}': {e}")
                    break

                if counter_2 == 0:
                    beam_count = 100000
                    counter_2 = 1
                    # print('diez mil')

            counter_1 += 1
            if counter_1 == 30:
                print("No se encontró una solución en el rango especificado.")
                print('Thickness:', thickness, 'mm')
                print('Ratio:', ratio, '%')
                break

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

# 4.1 ========================================================================================================================================================

def PlotsFormatting():
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
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