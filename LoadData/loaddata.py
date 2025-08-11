import os
import glob
import pandas as pd
from tqdm import tqdm
from cellpose.models import CellposeModel #for cellpose 4.0
from tifffile import imread, imwrite
import numpy as np
from scipy.spatial import distance, KDTree
from collections import defaultdict
from scipy import sparse
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
import scanpy as sc

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
            

def build_spatial_transcriptomics_dataset(root_dir):
    """
    Iterates through each time folder in root_dir, uses Apical_layers.tiff to filter
    cells overlapping with the apical domain, assigns transcript data to them,
    and builds a merged AnnData object across timepoints.

    Parameters:
    - root_dir (str): Root directory containing timepoint folders.

    Returns:
    - AnnData: Apical domain single-cell spatial transcriptomics dataset.
    """
    all_adata = []

    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for folder in tqdm(subdirs, desc="Processing time folders"):
        try:
            mask_files = glob.glob(os.path.join(folder, "*_mask.tiff"))
            txt_files = glob.glob(os.path.join(folder, "*Results.txt"))
            apical_mask_path = os.path.join(folder, "Apical_layers.tiff")

            if not mask_files or not txt_files:
                print(f"Skipping {folder}: missing mask or transcript file.")
                continue
            if not os.path.exists(apical_mask_path):
                print(f"Skipping {folder}: missing Apical_layers.tiff.")
                continue

            mask_path = mask_files[0]
            txt_path = txt_files[0]
            time_tag = os.path.basename(folder).replace('_', '.')

            # Load files
            mask_img = imread(mask_path)
            df = pd.read_csv(txt_path, sep="\t", header=None)
            df.columns = ['x', 'y', 'z', 'gene', 'score']
            apical_mask = imread(apical_mask_path).astype(bool)

            gene_coords = df[['y', 'x']].values
            gene_labels = df['gene'].values

            cell_gene_data = []

            # pre-filtering: only process cell IDs overlapping apical domain
            overlap_ids = np.unique(mask_img[apical_mask])
            overlap_ids = overlap_ids[overlap_ids > 0]

            for cell_id in overlap_ids:
                cell_mask = mask_img == cell_id
                prop = regionprops(cell_mask.astype(np.uint8))[0]  # only 1 region

                centroid_y, centroid_x = prop.centroid

                # Assign genes
                inside_indices = []
                for i, (y, x) in enumerate(gene_coords):
                    yi, xi = int(round(y)), int(round(x))
                    if 0 <= yi < cell_mask.shape[0] and 0 <= xi < cell_mask.shape[1]:
                        if cell_mask[yi, xi]:
                            inside_indices.append(i)

                if not inside_indices:
                    continue

                gene_counts = defaultdict(int)
                for idx in inside_indices:
                    gene = gene_labels[idx]
                    gene_counts[gene] += 1

                unique_cell_id = f"{time_tag}_Cell{cell_id}"
                cell_entry = {
                    'Time': time_tag,
                    'Cell_ID': unique_cell_id,
                    'Centroid_Y': centroid_y,
                    'Centroid_X': centroid_x,
                    **gene_counts
                }
                cell_gene_data.append(cell_entry)

            if not cell_gene_data:
                print(f"No apical cells found in {folder}")
                continue

            df_cell = pd.DataFrame(cell_gene_data).set_index("Cell_ID")

            obs = pd.DataFrame({
                'time': df_cell['Time'],
                'sample_name': df_cell.index.str.split("_", expand=True).get_level_values(0)
            }, index=df_cell.index)

            obsm = {'spatial': df_cell[['Centroid_X', 'Centroid_Y']].values}
            var = pd.DataFrame(index=df_cell.drop(['Time', 'Centroid_X', 'Centroid_Y'], axis=1).columns)
            X = sparse.csr_matrix(df_cell.drop(['Time', 'Centroid_X', 'Centroid_Y'], axis=1).fillna(0).values)

            adata = sc.AnnData(X=X, obs=obs, var=var, obsm=obsm)
            all_adata.append(adata)

        except Exception as e:
            print(f"Error in {folder}: {e}")
            continue

    if not all_adata:
        print("No valid apical cells found in any folder.")
        return None

    ST_data = all_adata[0].concatenate(*all_adata[1:], join='outer', index_unique=None)
    return ST_data