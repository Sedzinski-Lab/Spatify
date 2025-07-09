import os
import glob
import pandas as pd
from tqdm import tqdm
from cellpose.models import CellposeModel #for cellpose 4.0
from tifffile import imread, imwrite
import numpy as np
from scipy.spatial import distance

from scipy.spatial import KDTree

import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm
from matplotlib.path import Path  # <-- for point-in-polygon test

def process_text_files_with_distance(root_directory):
    """
    Processes all `*Results.txt` files in subdirectories of the specified root directory,
    and adds signed distance (positive = inside, negative = outside) to the tissue outline based on `sliceVertices*.dat`.

    Returns:
    - pd.DataFrame: with columns ['x', 'y', 'z', 'gene', 'score', 'Source_File', 'Stage', 'Time', 'Distance_To_Outline']
    """
    all_data = pd.DataFrame()

    # Recursively find all *Results.txt files
    txt_files = glob.glob(os.path.join(root_directory, "*", "*Results.txt"))

    for file_path in tqdm(txt_files):
        try:
            file_name = os.path.basename(file_path)
            folder_path = os.path.dirname(file_path)
            folder_name = os.path.basename(folder_path)

            # Extract stage info (e.g., from W1A3_R8_DAPI_Results.txt)
            try:
                time_stage = file_name.split('_')[2]
            except IndexError:
                time_stage = None

            # Read molecule data
            df = pd.read_csv(file_path, sep="\t", header=None)
            df.columns = ['x', 'y', 'z', 'gene', 'score']

            # Load ROI outline
            roi_candidates = glob.glob(os.path.join(folder_path, 'sliceVertices*.dat'))
            if roi_candidates:
                outline = np.loadtxt(roi_candidates[0])
                coords = df[['x', 'y']].values

                # 1. Compute distances to outline using KDTree
                tree = KDTree(outline)
                distances, _ = tree.query(coords, k=1)

                # 2. Determine inside or outside using Path
                path = Path(outline)  # assume outline is closed
                inside_mask = path.contains_points(coords)

                # 3. Assign signed distance
                signed_distances = np.where(inside_mask, distances, -distances)
                df['Distance_To_Outline'] = signed_distances
            else:
                print(f"Warning: No ROI found in {folder_path}")
                df['Distance_To_Outline'] = np.nan

            # Add metadata
            df['Source_File'] = file_name
            df['Stage'] = time_stage
            df['Time'] = folder_name

            all_data = pd.concat([all_data, df], ignore_index=True)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return all_data



def run_cellpose_on_directory(root_dir):
    """
    Runs Cellpose segmentation on all DAPI.tiff images under the root directory,
    and saves the resulting masks in the same folder with '_mask.tiff' suffix.

    Parameters:
    - root_dir (str): Root directory containing subfolders with DAPI.tiff images.
    """
    # Initialize Cellpose v4+ model (no model_type)
    model = CellposeModel(gpu=True)

    # Recursively find all DAPI.tiff or DAPI.tif files
    dapi_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith('dapi.tiff') or f.lower().endswith('dapi.tif'):
                dapi_files.append(os.path.join(root, f))

    for file_path in tqdm(dapi_files):
        print(f"Processing: {file_path}")

        try:
            # Read image
            img = imread(file_path)

            # Run Cellpose prediction (v4+ style)
            masks, flows, styles = model.eval(img, 
                                              diameter=None, 
                                              flow_threshold=0.5)

            # Save the mask in the same folder with '_mask.tiff' suffix
            base_name = os.path.splitext(file_path)[0]
            mask_path = f'{base_name}_mask.tiff'
            imwrite(mask_path, masks.astype(np.uint16))

            print(f"Saved mask to: {mask_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")