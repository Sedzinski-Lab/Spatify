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
from scipy.ndimage import binary_fill_holes

def process_text_files_with_distance(root_directory, reference_image_path=None):
    """
    Processes all `*Results.txt` files in subdirectories of the specified root directory,
    and adds signed distance (positive = inside, negative = outside) to the tissue outline based on `sliceVertices*.dat`.

    Parameters:
    - root_directory (str): Root folder containing folders like '10_5', '13', etc.
    - reference_image_path (str, optional): Path to a reference image (e.g., DAPI.tiff) to get the full image shape.

    Returns:
    - pd.DataFrame: with columns ['x', 'y', 'z', 'gene', 'score', 'Source_File', 'Stage', 'Time', 'Distance_To_Outline']
    """
    all_data = pd.DataFrame()

    txt_files = glob.glob(os.path.join(root_directory, "*", "*Results.txt"))

    for file_path in tqdm(txt_files):
        try:
            file_name = os.path.basename(file_path)
            folder_path = os.path.dirname(file_path)
            folder_name = os.path.basename(folder_path)

            try:
                time_stage = file_name.split('_')[2]
            except IndexError:
                time_stage = None

            # Read transcript data
            df = pd.read_csv(file_path, sep="\t", header=None)
            df.columns = ['x', 'y', 'z', 'gene', 'score']
            coords = df[['x', 'y']].values

            # Load outline
            roi_candidates = glob.glob(os.path.join(folder_path, 'sliceVertices*.dat'))
            if not roi_candidates:
                print(f"Warning: No ROI found in {folder_path}")
                df['Distance_To_Outline'] = np.nan
            else:
                outline = np.loadtxt(roi_candidates[0])

                # Create binary mask from outline
                outline = np.round(outline).astype(int)
                if reference_image_path:
                    ref_img = imread(reference_image_path)
                    img_shape = ref_img.shape
                else:
                    # Estimate bounding box from outline if no image provided
                    img_shape = (outline[:, 1].max() + 1, outline[:, 0].max() + 1)

                mask = np.zeros(img_shape, dtype=bool)
                yx = outline[:, ::-1]
                yx[:, 0] = np.clip(yx[:, 0], 0, img_shape[0]-1)
                yx[:, 1] = np.clip(yx[:, 1], 0, img_shape[1]-1)
                mask[yx[:, 0], yx[:, 1]] = 1

                filled_mask = binary_fill_holes(mask)

                # Determine inside/outside from mask
                int_coords = np.round(coords).astype(int)
                int_coords[:, 0] = np.clip(int_coords[:, 0], 0, img_shape[1]-1)
                int_coords[:, 1] = np.clip(int_coords[:, 1], 0, img_shape[0]-1)
                inside_mask = filled_mask[int_coords[:, 1], int_coords[:, 0]]

                # Distance using KDTree
                tree = KDTree(outline)
                distances, _ = tree.query(coords, k=1)

                signed_distances = np.where(inside_mask, distances, -distances)
                df['Distance_To_Outline'] = signed_distances

            # Add metadata
            df['Source_File'] = file_name
            df['Stage'] = time_stage
            df['Time'] = folder_name.replace('_', '.')

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