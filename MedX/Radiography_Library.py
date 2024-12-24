# 0.0. ========================================================================================================================================================



# 1.1.1. ========================================================================================================================================================

def MergeRoots(directory, starts_with, output_name):

    import uproot; import os; from tqdm import tqdm

    file_list = []

    # Crear lista de archivos para procesar
    for file in os.listdir(directory):
        if file.endswith('.root') and not file.startswith('merge') and not file.startswith(output_name):
            if starts_with == '' or file.startswith(starts_with):
                file_list.append(os.path.join(directory, file))

    merged_file = os.path.join(directory, output_name)
    counter = 0
    while os.path.exists(f"{merged_file}_{counter}.root"): counter += 1
    merged_file = f"{merged_file}_{counter}.root"

    with uproot.recreate(merged_file) as f_out:
        for file in tqdm(file_list, desc="Merging ROOT files", unit="file"):
            with uproot.open(file) as f_in:
                for key in f_in.keys():
                    base_key = key.split(';')[0]  # Obtener el nombre base sin número de ciclo
                    obj = f_in[key]

                    # Solo procesar si es un TTree
                    if isinstance(obj, uproot.TTree):
                        # Leer los datos por partes para optimizar el uso de memoria
                        for new_data in obj.iterate(library="np", step_size="10 MB"):
                            # Si el árbol ya existe en el archivo de salida, añadir los datos en partes
                            if base_key in f_out:
                                f_out[base_key].extend(new_data)
                            else:
                                # Crear un nuevo TTree en el archivo de salida con los primeros datos
                                f_out[base_key] = new_data

    print("Archivo final creado en:", merged_file)

# 1.1.2. ========================================================================================================================================================

def process_file(file, f_out, lock, trim_coords):

    import uproot

    step_size = "10 MB"
    
    with uproot.open(file) as f_in:
        
        for key in f_in.keys():
            base_key = key.split(';')[0]
            obj = f_in[key]

            if base_key == "Hits" and isinstance(obj, uproot.TTree):
                for new_data in obj.iterate(["x_ax", "y_ax"], library="np", step_size=step_size):
                    if trim_coords:
                        x_min, x_max, y_min, y_max = trim_coords
                        mask = ((new_data['x_ax'] >= x_min) & (new_data['x_ax'] <= x_max) & (new_data['y_ax'] >= y_min) & (new_data['y_ax'] <= y_max))
                        if mask.sum() == 0: print("No data after filtering. Skipping chunk."); continue
                        new_data = {key: value[mask] for key, value in new_data.items()}
                    with lock: # Lock para asegurar que la escritura en f_out sea thread-safe
                        if base_key in f_out: f_out[base_key].extend(new_data)
                        else: f_out[base_key] = new_data

            elif base_key == "Run Summary" and isinstance(obj, uproot.TTree):
                for summary_data in obj.iterate(library="np", step_size=step_size):
                    with lock:
                        if base_key in f_out: f_out[base_key].extend(summary_data)
                        else:f_out[base_key] = summary_data
            
            else: print(f"Skipping unrecognized tree or object: {base_key}")

def MergeRoots_Parallel(directory, starts_with, output_name, trim_coords):
    
    import uproot; import os; from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor; import threading

    max_workers = 9
    
    file_list = []
    for file in os.listdir(directory):
        if file.endswith('.root') and not file.startswith('merge') and not file.startswith(output_name):
            if starts_with == '' or file.startswith(starts_with): file_list.append(os.path.join(directory, file))

    merged_file = directory + output_name 
    if not os.path.exists(merged_file + ".root"): merged_file = merged_file + ".root"
    if os.path.exists(merged_file + ".root"):
        counter = 0
        while os.path.exists(f"{merged_file}_{counter}.root"): counter += 1
        merged_file = f"{merged_file}_{counter}.root"

    lock = threading.Lock() # Crear un lock para el acceso a f_out

    with uproot.recreate(merged_file) as f_out:
        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            futures = [executor.submit(process_file, file, f_out, lock, trim_coords=trim_coords) for file in file_list]
            for future in tqdm(futures, desc="Merging ROOT files", unit="file"): future.result()  # Asegura que se complete cada tarea

    print("Archivo final creado en:", merged_file)

# 1.2. ========================================================================================================================================================

def Summary_Data(directory, root_file, tree_1, branch_1, tree_2, branches_2):

    import uproot

    file_path = directory + root_file

    with uproot.open(file_path) as file:
        
        tree_1 = file[tree_1]
        if branch_1 not in tree_1.keys(): raise ValueError(f"Branch: '{branch_1}', not found in tree: '{tree_1}'.")
        NumberofHits = len(tree_1[branch_1].array(library="np"))

        tree_2 = file[tree_2]
        if branches_2[0] not in tree_2.keys(): raise ValueError(f"Branch: '{branches_2[0]}', not found in tree: '{tree_2}'.")
        if branches_2[1] not in tree_2.keys(): raise ValueError(f"Branch: '{branches_2[1]}', not found in tree: '{tree_2}'.")
        if branches_2[2] not in tree_2.keys(): raise ValueError(f"Branch: '{branches_2[2]}', not found in tree: '{tree_2}'.")

        NumberofPhotons  = tree_2[branches_2[0]].array(library="np").sum()
        EnergyDeposition = tree_2[branches_2[1]].array(library="np").sum()
        RadiationDose    = tree_2[branches_2[2]].array(library="np").sum()
    
    return NumberofHits, NumberofPhotons, EnergyDeposition, RadiationDose

# 1.3. ========================================================================================================================================================

def CT_Summary_Data(directory, tree, branches):

    import uproot; import os

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
        
    return NumberofPhotons, EnergyDeposition, RadiationDose

# 2.0. ========================================================================================================================================================

def Root_to_Heatmap(directory, root_name, tree_name, x_branch, y_branch, size, pixel_size):

    import uproot; import numpy as np; import dask.array as da
    
    chunk_size = 1_000_000

    file_name = directory + root_name

    with uproot.open(file_name) as root_file:
        
        tree = root_file[tree_name]
        if tree is None: print(f"Tree '{tree_name}' not found in {file_name}"); return
        if x_branch not in tree or y_branch not in tree: print(f"Branches '{x_branch}' or '{y_branch}' not found in the tree"); return

        x_values = da.from_array(tree[x_branch].array(library="np"), chunks=chunk_size)
        y_values = da.from_array(tree[y_branch].array(library="np"), chunks=chunk_size)

        xmin = tree[x_branch].array(library="np").min()
        xmax = tree[x_branch].array(library="np").max()
        ymin = tree[y_branch].array(library="np").min()
        ymax = tree[y_branch].array(library="np").max()

        xmin = np.ceil(xmin)
        xmax = np.floor(xmax)
        ymin = np.ceil(ymin)
        ymax = np.floor(ymax)

    x_shift = size[2]
    y_shift = size[3]

    x_down = xmin + x_shift
    x_up   = xmax + x_shift
    y_down = ymin + y_shift
    y_up   = ymax + y_shift

    x_data_shifted = x_values - x_shift            
    y_data_shifted = y_values - y_shift

    bins_x0 = np.arange(x_down, x_up + pixel_size, pixel_size)
    bins_y0 = np.arange(y_down, y_up + pixel_size, pixel_size)

    heatmap = da.full((len(bins_x0) - 1, len(bins_y0) - 1), 0, dtype=float)

    for x_chunk, y_chunk in zip(x_data_shifted.to_delayed(), y_data_shifted.to_delayed()):
        
        x_chunk = da.from_delayed(x_chunk, shape=(chunk_size,), dtype=np.float32)
        y_chunk = da.from_delayed(y_chunk, shape=(chunk_size,), dtype=np.float32)

        chunk_histogram, _, _ = da.histogram2d(x_chunk, y_chunk, bins=[bins_x0, bins_y0])
        heatmap = heatmap + chunk_histogram

    heatmap = heatmap.compute()
    heatmap = np.rot90(heatmap.T, 2)

    return heatmap, bins_x0, bins_y0

def Logaritmic_Transform(heatmap, size, pixel_size):

    import numpy as np; import matplotlib.pyplot as plt
    
    maxi_vector = heatmap[0, :]
    heatmap[heatmap == 0] = np.nan
    heatmap = np.log(maxi_vector / heatmap)

    size_x = size[0]; 
    size_y = size[1]; 

    bins_x1 = np.arange(-size_x, size_x + pixel_size, pixel_size)
    bins_y1 = np.arange(-size_y, size_y + pixel_size, pixel_size)

    new_size = np.zeros((len(bins_x1), len(bins_y1)))
    
    size_0 = new_size.shape
    size_1 = heatmap.shape

    if size_0 > size_1:
    
        start_row = (size_0[0] - size_1[0]) // 2
        start_col = (size_0[1] - size_1[1]) // 2

        padded_matrix = np.zeros(size_0)
        # padded_matrix = np.full(size_0, maxi)
        padded_matrix[start_row:start_row + size_1[0], start_col:start_col + size_1[1]] = heatmap
        heatmap = padded_matrix
    
    else: 
    
        start_row = (size_1[0] - size_0[0]) // 2
        start_col = (size_1[1] - size_0[1]) // 2

        cropped_matrix = heatmap[start_row:start_row + size_0[0], start_col:start_col + size_0[1]]
        heatmap = cropped_matrix
    
    return heatmap, bins_x1, bins_y1

def Plot_Heatmap(heatmap, set_bins_x, set_bins_y, save_as):

    import matplotlib.pyplot as plt

    rows = heatmap.shape[0]

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1); plt.imshow(heatmap, cmap="gray", extent=[set_bins_x[0], set_bins_y[-1], set_bins_x[0], set_bins_y[-1]]); plt.colorbar()
    if save_as: plt.savefig(save_as + ".png", bbox_inches="tight", dpi=900)
    plt.subplot(1, 3, 2); plt.plot(heatmap[2*rows//3, :])
    plt.subplot(1, 3, 3); plt.plot(heatmap[:, rows//2])

# 3.0. ========================================================================================================================================================

def IsolateTissues(low_energy_img, high_energy_img, sigma1, sigma2, wn, save_in, save_as):

    from scipy.ndimage import gaussian_filter; import matplotlib.pyplot as plt

    save_as_1 = save_as[0]
    save_as_2 = save_as[1]
    save_as_3 = save_as[2]
    save_as_4 = save_as[3]
    save_as_5 = save_as[4]
    save_as_6 = save_as[5]
    save_as_7 = save_as[6]
    save_as_8 = save_as[7]

    U_b_l = 0.7519 # mu1
    U_b_h = 0.3012 # mu2
    U_t_l = 0.26 # mu3
    U_t_h = 0.18 # mu4

    SLS_Bone = ( (U_t_h/U_t_l) * low_energy_img ) - high_energy_img
    SLS_Tissue = high_energy_img - ( low_energy_img * (U_b_h/U_b_l) )

    SSH_Bone = ( (U_t_h/U_t_l) * low_energy_img) - gaussian_filter(high_energy_img, sigma = sigma1)
    SSH_Tissue = gaussian_filter(high_energy_img, sigma = sigma1) - ( low_energy_img * (U_b_h/U_b_l) )

    ACNR_Bone = SLS_Bone + (gaussian_filter(SLS_Tissue, sigma = sigma1)*wn) - 1
    ACNR_SSH_Bone = SSH_Bone + (gaussian_filter(SSH_Tissue, sigma = sigma2) * wn) - 1
    ACNR_Tissue = SLS_Tissue + (gaussian_filter(SLS_Bone, sigma = sigma1)*wn) - 1

    # w  = U_t_h / U_t_l
    # wc = U_b_h / U_b_l
    # low  = - (wn * wc * gaussian_filter(low_energy_img, sigma = sigma1) ) + (w * low_energy_img)
    # high = - high_energy_img + ( wn * gaussian_filter(high_energy_img, sigma = sigma1))
    # ACNR_LONG_bone = low + high

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

    plt.figure(figsize = (18, 10))
    plt.tight_layout()
    plt.subplot(2, 4, 1); plt.imshow(low_energy_img,    cmap='gray'); plt.axis('off');  plt.title("Low Energy")
    plt.subplot(2, 4, 2); plt.imshow(high_energy_img,   cmap='gray'); plt.axis('off');  plt.title("High Energy")
    plt.subplot(2, 4, 3); plt.imshow(SLS_Bone,          cmap='gray'); plt.axis('off');  plt.title("Bone [SLS]")
    plt.subplot(2, 4, 4); plt.imshow(SLS_Tissue,        cmap='gray'); plt.axis('off');  plt.title("Tissue [SLS]")
    plt.subplot(2, 4, 5); plt.imshow(SSH_Bone,          cmap='gray'); plt.axis('off');  plt.title("Bone [SSH]")
    plt.subplot(2, 4, 6); plt.imshow(SSH_Tissue,        cmap='gray'); plt.axis('off');  plt.title("Tissue [SSH]")
    plt.subplot(2, 4, 7); plt.imshow(ACNR_Bone,         cmap='gray'); plt.axis('off');  plt.title("Bone [ACNR]")
    # plt.subplot(2, 4, 8); plt.imshow(ACNR_SSH_Bone,     cmap='gray'); plt.axis('off');  plt.title("Bone [ACNR + SSH]")
    plt.subplot(2, 4, 8); plt.imshow(ACNR_Tissue,       cmap='gray'); plt.axis('off');  plt.title("Tissue [ACNR]")
   
    return SLS_Bone, SLS_Tissue, SSH_Bone, SSH_Tissue, ACNR_Bone, ACNR_Tissue

# 4.0. ========================================================================================================================================================

def BMO(SLS_Bone, SLS_Tissue, save_as):

    import matplotlib.pyplot as plt

    U_b_l = 0.7519 # mu1
    U_b_h = 0.3012 # mu2
    # U_t_l = 0.26 # mu3
    # U_t_h = 0.18 # mu4
    U_t_l = 0.281
    U_t_h = 0.192

    Thick_cons_bone = (U_t_l) / ( (U_t_h * U_b_l) - (U_t_l * U_b_h) )
    thickness_bone = Thick_cons_bone * SLS_Bone
    Thick_cons_tissue = (U_t_l) / ( (U_t_l * U_b_h) - (U_t_h * U_b_l) )
    thickness_tissue = Thick_cons_tissue * SLS_Tissue

    plt.figure(figsize = (12, 3))
    plt.subplot(1, 3, 1); plt.imshow(thickness_bone); plt.colorbar()
    plt.subplot(1, 3, 2); plt.plot(thickness_bone[120,:])
    plt.subplot(1, 3, 3); plt.plot(thickness_bone[:,120])
    if save_as != '': plt.savefig(save_as, bbox_inches = 'tight', dpi = 600); 
    plt.show()

    return thickness_bone

# 5.1 ========================================================================================================================================================

def Interactive_CNR(cropped_image):

    import numpy as np; import matplotlib.pyplot as plt

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
    
    from PIL import Image; import numpy as np; import matplotlib.pyplot as plt

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

    import numpy as np; import matplotlib.pyplot as plt
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
    
    import numpy as np; import matplotlib.pyplot as plt
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

    import plotly.io as pio; import plotly.graph_objects as go

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

    import plotly.graph_objects as go
    import plotly.io as pio

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

    import os

    if os.path.exists(directory):
        
        for file_name in os.listdir(directory):

            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                try: os.remove(file_path)
                except Exception as e: print(f"Error deleting file {file_path}: {e}")

def CT_Loop(directory, starts_with, angles):

    import Radiography_Library as RadLib; import platform; from tqdm.notebook import tqdm;
    import os; import subprocess; import shutil; from contextlib import redirect_stdout, redirect_stderr

    if platform.system() == "Darwin":
        executable_file = "Sim"
        mac_filename = 'CT.mac'
        run_sim = f"./{executable_file} {mac_filename} . . ."
    elif platform.system() == "Windows":
        executable_file = "Sim.exe"
        mac_filename = 'CT.mac'
        run_sim = f".\{executable_file} .\{mac_filename} . . ."
    else: raise EnvironmentError("Unsupported operating system")

    root_folder = directory + "ROOT/"
    mac_filepath = directory + mac_filename
    ct_folder = directory + "ROOT/" + "CT/"
    os.makedirs(ct_folder, exist_ok = True)
    
    for angle in tqdm(range(angles[0], angles[1]), desc = "Creating CT", unit = "Angles", leave = True):
        
        ClearFolder(root_folder)

        mac_template = \
        """ \
        /myDetector/Rotation {angle}
        /myDetector/nColumns 1
        /myDetector/nRows 1
        /run/reinitializeGeometry

        /run/numberOfThreads 10
        /run/initialize

        /Pgun/X 0 mm
        /Pgun/gaussX true
        /Pgun/Xcos true
        /Pgun/SpanY 0.01 mm

        /gun/energy 80 keV

        {beam_lines}
        """

        start = -280
        end = 245
        step = 4
        beam_lines = "\n".join(f"        /Pgun/Y {y} mm\n        /run/beamOn 150000" for y in range(start, end + 1, step))

        filled_template = mac_template.format(angle = angle, beam_lines = beam_lines)
        with open(mac_filepath, 'w') as f: f.write(filled_template)

        try: subprocess.run(run_sim, cwd = directory, check = True, shell = True, stdout = subprocess.DEVNULL)
        except subprocess.CalledProcessError as e: print(f"Error al ejecutar la simulación: {e}")
    
        output_name = f'Aang_{angle}'
        if os.path.exists(ct_folder + output_name + '.root'):
            counter = 0
            while os.path.exists(ct_folder + output_name + '_' + str(counter) + '.root'): counter = counter + 1
            output_name = root_folder + output_name + '_' + str(counter)

        with open(os.devnull, "w") as fnull: 
            with redirect_stdout(fnull), redirect_stderr(fnull):
                RadLib.MergeRoots_Parallel(root_folder, starts_with, output_name, trim_coords = None)

        merged_file_path = root_folder + output_name + '.root'

        if os.path.exists(merged_file_path): shutil.move(merged_file_path, ct_folder)

        ClearFolder(root_folder)


def Calculate_Projections(directory, filename, roots, tree_name, x_branch, y_branch, dimensions, pixel_size, csv_folder):
    
    import numpy as np; from tqdm import tqdm; import matplotlib.pyplot as plt

    start = roots[0]
    end = roots[1]
    deg = roots[2]
    projections = np.arange(start, end+1, deg)

    for i in tqdm(projections, desc = 'Calculating heatmaps', unit = ' Heatmap', leave = True):

        root_name = filename + '_' + str(i) + '.root'
        htmp_array, xlim, ylim = Root_to_Heatmap(directory, root_name, tree_name, x_branch, y_branch, dimensions, pixel_size)

        name = csv_folder + "/CT_" + str(i) + ".csv"
        np.savetxt(name, htmp_array, delimiter=',', fmt='%d')

    return htmp_array, xlim, ylim


def LoadHeatmapsFromCSV(csv_folder, roots):

    import numpy as np; from tqdm import tqdm

    start = roots[0]
    end = roots[1]
    deg = roots[2]
    sims = np.arange(start, end+1, deg)
    
    htmps = np.zeros(len(sims), dtype=object)
    for i, sim in tqdm(enumerate(sims), desc = 'Creating heatmaps from CSV files', unit = ' heatmaps', leave = True):
        name = csv_folder + f"CT_{round(sim)}.csv"
        htmps[i] = np.genfromtxt(name, delimiter = ',')

    return htmps


def RadonReconstruction(roots, htmps, layers):

    import numpy as np; import matplotlib.pyplot as plt; import plotly.graph_objects as go; import plotly.io as pio; from tqdm import tqdm; from skimage.transform import iradon
    
    initial = layers[0]
    final = layers[1]
    spacing = layers[2]
    slices = np.round(np.arange(initial, final, spacing))
    
    start = roots[0]
    end = roots[1]
    deg = roots[2]

    thetas = np.arange(start, end+1, deg)
    reconstructed_imgs = np.zeros(len(slices), dtype="object")

    for i, layer in tqdm(enumerate(slices), desc = 'Reconstructing slices', unit = ' Slices', leave = True):

        p = np.array([heatmap[layer] for heatmap in htmps]).T
        reconstructed_imgs[i] = iradon(p, theta = thetas)

    # plt.figure(figsize = (6,6)); plt.imshow(reconstructed_imgs[slices//2], cmap = 'gray'); plt.colorbar(); plt.show()
    
    fig = go.Figure(go.Heatmap(z = reconstructed_imgs[0]))
    fig.update_layout(width = 600, height = 600, xaxis = dict(autorange = 'reversed'), yaxis = dict(autorange = 'reversed'))
    fig.show()

    return reconstructed_imgs


def CoefficientstoHU(reconstructed_imgs, slices, mu_water, air_parameter):

    import numpy as np; import plotly.graph_objects as go; import plotly.io as pio 

    initial = slices[0]
    final = slices[1]
    spacing = slices[2]
    slices = np.round(np.arange(initial, final, spacing))

    HU_images = np.zeros(len(slices), dtype="object")

    for i in range(len(HU_images)):

        HU_images[i] = np.round(1000 * ((reconstructed_imgs[i] - mu_water) / mu_water)).astype(int)
        HU_images[i][HU_images[i] < air_parameter] = -1000

    # fig = go.Figure(go.Heatmap(z = HU_images[0], colorscale = [[0, 'black'], [1, 'white']],))
    # fig.update_layout(width = 600, height = 600, xaxis = dict(autorange = 'reversed'), yaxis = dict(autorange = 'reversed'))
    # fig.show()

    return HU_images


def export_to_dicom(HU_images, size_y, directory, compressed):

    import numpy as np; import pydicom; from pydicom.pixels import compress
    from pydicom.dataset import Dataset, FileDataset; from pydicom.uid import RLELossless 
    from pydicom.uid import ExplicitVRLittleEndian; from pydicom.encaps import encapsulate
    
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

def Root_to_Dask(directory, root_name, tree_name, x_branch, y_branch):
    
    import uproot; import numpy as np
    import dask.array as da; import dask.dataframe as dd

    file_name = directory + root_name 

    with uproot.open(file_name) as root_file:
        tree = root_file[tree_name]
        if tree is None:
            print(f"Tree '{tree_name}' not found in {file_name}")
            return

        x_values = tree[x_branch].array(library="np") if x_branch in tree else print('error_x')
        y_values = tree[y_branch].array(library="np") if y_branch in tree else print('error_y')

        decimal_places = 1

        if x_values is not None:
            x_values = np.round(x_values, decimal_places)
        if y_values is not None:
            y_values = np.round(y_values, decimal_places)

        if x_values is None or y_values is None:
            print(f"Could not retrieve data for branches {x_branch} or {y_branch}")
            return

        x_dask_array = da.from_array(x_values, chunks="auto")
        y_dask_array = da.from_array(y_values, chunks="auto")

        dask_df = dd.from_dask_array(da.stack([x_dask_array, y_dask_array], axis=1), columns=[x_branch, y_branch])

        x_data = dask_df[x_branch].to_dask_array(lengths=True)
        y_data = dask_df[y_branch].to_dask_array(lengths=True)
        
        return x_data, y_data

def Heatmap_from_Dask(x_data, y_data, size, log_factor, x_shift, y_shift, save_as):

    import matplotlib.pyplot as plt; import numpy as np
    import dask.array as da; import dask.dataframe as dd
    
    x_data_shifted = x_data - x_shift
    y_data_shifted = y_data - y_shift

    pixel_size = 0.5 # mm
    set_bins = np.arange(-size, size + pixel_size, pixel_size)

    heatmap, x_edges, y_edges = da.histogram2d(x_data_shifted, y_data_shifted, bins = [set_bins, set_bins])
    heatmap = heatmap.T
    heatmap = np.rot90(heatmap, 2)
    # print('Heatmap size:', heatmap.shape, '[pixels]')
    rows = heatmap.shape[0]

    heatmap = heatmap.compute()  
    x_edges = x_edges.compute()  
    y_edges = y_edges.compute()

    heatmap[heatmap == 0] = log_factor
    maxi = np.max(heatmap)
    normal_map = np.log(maxi / heatmap)

    plt.figure(figsize = (14, 4))
    plt.subplot(1, 3, 1); plt.imshow(normal_map, cmap = 'gray', extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]); plt.axis('off')
    if save_as != '': plt.savefig(save_as + '.png', bbox_inches = 'tight', dpi = 900)
    plt.subplot(1, 3, 2); plt.plot(normal_map[2*rows//3,:])
    plt.subplot(1, 3, 3); plt.plot(normal_map[:,rows//2])

    return normal_map, x_edges, y_edges

def LoadRoots(directory, rootnames, tree_name, x_branch, y_branch):

    x_1, y_1 = Root_to_Dask(directory, rootnames[0], tree_name, x_branch, y_branch)
    x_2, y_2 = Root_to_Dask(directory, rootnames[1], tree_name, x_branch, y_branch)
    print("Dataframes created")

    return x_1, y_1, x_2, y_2

def CT_Heatmap_from_Dask(x_data, y_data, size_x, size_y, x_shift, y_shift, pixel_size):

    import matplotlib.pyplot as plt; import numpy as np
    import dask.array as da; import dask.dataframe as dd

    x_data_shifted = x_data - x_shift
    y_data_shifted = y_data - y_shift

    set_bins_x = np.arange(-size_x, size_x + pixel_size, pixel_size)
    set_bins_y = np.arange(-size_y, size_y + pixel_size, pixel_size)
    heatmap, x_edges, y_edges = da.histogram2d(x_data_shifted, y_data_shifted, bins = [set_bins_x, set_bins_y])
    heatmap = heatmap.T
    heatmap = np.rot90(heatmap, 2)

    heatmap = heatmap.compute() 
    x_edges = x_edges.compute()  
    y_edges = y_edges.compute()

    return heatmap, x_edges, y_edges

def LogaritmicTransformation(radiographs, pixel_size, sigma):
    
    import matplotlib.pyplot as plt; import numpy as np; from scipy import ndimage; from tqdm import tqdm

    htmps = np.zeros(len(radiographs), dtype = 'object')

    for i, radiograph in tqdm(enumerate(radiographs), desc = 'Computing logarithmic transformation', unit = ' Heatmaps', leave = True):
        radiograph = ndimage.gaussian_filter(radiograph, sigma)
        maxi = np.max(radiograph)
        htmps[i][htmps[i] == 0] = np.nan
        htmps[i] = np.log(maxi/radiograph) / (pixel_size * 0.1)

    plt.imshow(htmps[-1]); plt.colorbar(); plt.show()

    return htmps