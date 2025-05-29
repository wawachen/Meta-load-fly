import scipy.spatial
import scipy.io
import numpy as np
import os
import scipy.ndimage
import sys
from typing import Dict, Tuple, List

# Configuration for different tasks
TASK_CONFIGS = {
    1: {"path": "wind_x0.0_y0.0_2agents_L0.6", "start_idx": 210},
    2: {"path": "wind_x0.3_y0.0_2agents_L1.0", "start_idx": 210},
    3: {"path": "wind_x0.5_y0.0_2agents_L0.8", "start_idx": 210},
    4: {"path": "wind_x0.8_y0.0_2agents_L1.2", "start_idx": 210},
    5: {"path": "wind_x1.0_y0.0_2agents_L0.8", "start_idx": 210},
    6: {"path": "wind_x0.6_y0.0_2agents_L1.4", "start_idx": 210}
}

def setup_paths(task_num: int) -> Tuple[str, str, int]:
    """Setup paths and get file numbers for a given task."""
    if task_num not in TASK_CONFIGS:
        raise ValueError(f"Invalid task number: {task_num}")
    
    config = TASK_CONFIGS[task_num]
    base_path = "/home/wawa/catkin_meta/src/MBRL_transport/train_point_clouds"
    load_path = os.path.join(base_path, config["path"])
    save_path = os.path.join(load_path, "preprocess")
    
    files = os.listdir(load_path)
    file_num = len(files) - 1 + config["start_idx"]
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise FileExistsError(f"Save path already exists: {save_path}")
    
    return load_path, save_path, file_num

def setup_grid(N: int = 401) -> Tuple[np.ndarray, scipy.spatial.cKDTree]:
    """Setup the grid and KD-tree for point cloud processing."""
    voxel_origin = [0, 2]
    voxel_size = 4.0 / (N - 1)
    
    overall_index = np.arange(0, N ** 2)
    samples = np.zeros([N ** 2, 3])
    
    samples[:, 0] = overall_index % N
    samples[:, 1] = (overall_index // N) % N
    
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = -(samples[:, 1] * voxel_size) + voxel_origin[1]
    
    return samples, scipy.spatial.cKDTree(samples[:, :2])

def process_point_cloud(points: np.ndarray, samples: np.ndarray, mytree: scipy.spatial.cKDTree, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process point cloud data and compute SDF."""
    samples[:, 2] = 0
    dist_ckd, indexes_ckd = mytree.query(points[:, :2])
    indexes_ckd_uni = list(np.unique(indexes_ckd))
    
    if len(indexes_ckd_uni) < 1500:
        return None, None, None
    
    samples[indexes_ckd_uni, 2] = 1
    indexes_ckd_on = np.where(samples[:, 2] == 1)[0]
    indexes_ckd_off = np.where(samples[:, 2] == 0)[0]
    
    # Transform grid map into signed distance function
    img_tensor = samples[:, 2].reshape(N, N)
    neg_distances = scipy.ndimage.morphology.distance_transform_edt(img_tensor)
    sd_img = (img_tensor - 1).astype(np.uint8)
    signed_distances = scipy.ndimage.morphology.distance_transform_edt(sd_img) - neg_distances
    signed_distances /= float(img_tensor.shape[1])
    
    return signed_distances.reshape((-1, 1)), indexes_ckd_on, indexes_ckd_off

def save_processed_data(save_path: str, file_idx: int, configs: np.ndarray, samples: np.ndarray, 
                       signed_distances: np.ndarray, on_idx: np.ndarray, off_idx: np.ndarray,
                       subdir: str = None) -> None:
    """Save processed data to MAT file."""
    save_dir = os.path.join(save_path, subdir) if subdir else save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    mdic = {
        "configuration": configs,
        "top_all": samples[:, :2],
        "sdf": signed_distances,
        "on_surface_index": on_idx,
        "off_surface_index": off_idx
    }
    scipy.io.savemat(os.path.join(save_dir, f"{file_idx}.mat"), mdic)

def main():
    task_num = int(sys.argv[1])
    load_path, save_path, file_num = setup_paths(task_num)
    samples, mytree = setup_grid()
    
    # Special processing for tasks 5 and 6
    if task_num in [5, 6]:
        file_count = 0
        for i in range(210, file_num + 1):
            mat = scipy.io.loadmat(os.path.join(load_path, f"{i}.mat"))
            signed_distances, on_idx, off_idx = process_point_cloud(mat["top"], samples, mytree, 401)
            if signed_distances is not None:
                save_processed_data(save_path, file_count, mat["configuration"], samples, 
                                 signed_distances, on_idx, off_idx)
                file_count += 1
        return
    
    # Processing for other tasks with train/validation split
    start_idx = TASK_CONFIGS[task_num]["start_idx"]
    file_list = list(range(start_idx, file_num + 1))
    np.random.shuffle(file_list)
    
    train_num = int(len(file_list) * 0.9)
    train_files = file_list[:train_num]
    test_files = file_list[train_num:]
    
    for file_set, subdir, start_count in [(train_files, "train", 0), (test_files, "validation", 0)]:
        file_count = start_count
        for i in file_set:
            mat = scipy.io.loadmat(os.path.join(load_path, f"{i}.mat"))
            signed_distances, on_idx, off_idx = process_point_cloud(mat["top"], samples, mytree, 401)
            if signed_distances is not None:
                save_processed_data(save_path, file_count, mat["configuration"], samples,
                                 signed_distances, on_idx, off_idx, subdir)
                file_count += 1

if __name__ == "__main__":
    main()




