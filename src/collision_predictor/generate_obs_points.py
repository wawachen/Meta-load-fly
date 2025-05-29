import scipy
import numpy as np
import scipy.spatial
from typing import List, Tuple

def generate_circular_cloud(pos: List[float], radius: float, sample_num: int) -> np.ndarray:
    """Generate random points within a circular area."""
    cloud = np.zeros((sample_num**2, 2))
    for i in range(sample_num**2):
        r = radius * np.sqrt(np.random.random())
        theta = np.random.random() * 2 * np.pi
        cloud[i] = [pos[0] + r * np.cos(theta), pos[1] + r * np.sin(theta)]
    return cloud

def create_sample_grid(N: int = 401) -> Tuple[np.ndarray, float]:
    """Create a grid of sample points."""
    voxel_origin = [0, 2]
    voxel_size = 4.0 / (N - 1)
    
    overall_index = np.arange(0, N ** 2)
    samples = np.zeros([N ** 2, 3])
    samples[:, 0] = overall_index % N
    samples[:, 1] = (overall_index / N) % N
    
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = -(samples[:, 1] * voxel_size) + voxel_origin[1]
    
    return samples[:, :2], voxel_size

def process_obstacles(clouds: List[np.ndarray], samples: np.ndarray) -> List[np.ndarray]:
    """Process obstacle clouds to get final obstacle points."""
    mytree = scipy.spatial.cKDTree(samples)
    final_points = []
    
    for cloud in clouds:
        _, indexes = mytree.query(cloud)
        unique_indexes = list(np.unique(indexes))
        final_obs = samples[unique_indexes]
        # Normalize
        final_obs[:, 0] /= 4  # max_x
        final_obs[:, 1] /= 2  # max_y
        final_points.append(final_obs)
    
    return final_points

def main():
    resolution = 400
    shape = "square"  # Options: "square", "cross", "crowd_1"
    
    if shape == "square":
        obstacles = [
            {"pos": [3.0, 0.5], "radius": 0.2},
            {"pos": [1, -1], "radius": 0.3}
        ]
    elif shape == "cross":
        obstacles = [
            {"pos": [2.0, 0.0], "radius": 0.2}
        ]
    elif shape == "crowd_1":
        obstacles = [
            {"pos": [2.0, 1.0], "radius": 0.2},
            {"pos": [1.8, 0.0], "radius": 0.1},
            {"pos": [2.0, -1.0], "radius": 0.2},
            {"pos": [3.0, 0.0], "radius": 0.1}
        ]
    
    # Generate cloud points for each obstacle
    clouds = []
    for obs in obstacles:
        sample_num = round((obs["radius"] * 2) / (4.0 / resolution))
        cloud = generate_circular_cloud(obs["pos"], obs["radius"], sample_num)
        clouds.append(cloud)
    
    # Create sample grid and process obstacles， map points to the grid
    samples, _ = create_sample_grid()
    final_points = process_obstacles(clouds, samples)
    
    # Save results
    save_data = {}
    for i, (obs, final_point) in enumerate(zip(obstacles, final_points)):
        suffix = "" if i == 0 else str(i)
        save_data[f"obs{suffix}"] = final_point
        save_data[f"obs_pos{suffix}"] = obs["pos"]
        save_data[f"obs_radius{suffix}"] = obs["radius"]
    
    output_path = "/home/wawa/catkin_meta/src/MBRL_transport/"
    if shape == "square":
        output_file = "obs_points2.mat"
    elif shape == "cross":
        output_file = "obs_points1.mat"
    else:  # crowd_1
        output_file = "obs_points_crowd_1.mat"
    
    scipy.io.savemat(output_path + output_file, save_data)

if __name__ == "__main__":
    main()
    
