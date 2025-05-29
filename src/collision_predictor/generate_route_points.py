import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class RouteConfig:
    """Configuration for route generation"""
    route_name: str
    task_num: int
    center: np.ndarray = np.array([2.0, 0.0, 0.8])
    d_t: float = 0.02
    visualize: bool = False # Added visualization flag
    
    def get_offsets(self) -> Tuple[float, float, float, float]:
        """Get UAV offsets based on task number"""
        offsets = {
            0: (0.3024, 0.0786, -0.294, 0.0662),
            1: (0.2164, -0.0198, -0.4932, -0.018),
            2: (0.4876, -0.01, -0.5756, -0.0092)
        }
        return offsets.get(self.task_num, (0, 0, 0, 0))

def generate_waypoint(base_point: np.ndarray, dx1: float, dy1: float, dx2: float, dy2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate waypoints for UAV1 and UAV2 based on base point and offsets"""
    uav1_point = np.array([base_point[0] + dx1, base_point[1] + dy1, base_point[2]])
    uav2_point = np.array([base_point[0] + dx2, base_point[1] + dy2, base_point[2]])
    return uav1_point, uav2_point

def generate_square_xy_path(config: RouteConfig) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Generate square path in XY plane"""
    segments = [
        {"len": 2.0, "direction": [1, 0, 0]},    # Right
        {"len": 0.1, "direction": [0, -1, 0]},   # Down
        {"len": 0.4, "direction": [-1, 0, 0]},   # Left
        {"len": 0.8, "direction": [0, -1, 0]},   # Down
        {"len": 0.4, "direction": [1, 0, 0]},    # Right
        {"len": 1.1, "direction": [0, -1, 0]},   # Down
        {"len": 1.5, "direction": [-1, 0, 0]},   # Left
        {"len": 0.5, "direction": [0, 1, 0]},    # Up
        {"len": 0.5, "direction": [-1, 0, 0]},   # Left
        {"len": 1.5, "direction": [0, 1, 0]}     # Up
    ]
    
    dx1, dy1, dx2, dy2 = config.get_offsets()
    start_point = config.center - np.array([1.0, -1.0, 0.0])
    waypoints = [start_point]
    waypoints_uav1 = []
    waypoints_uav2 = []
    
    # Generate initial UAV positions
    uav1_point, uav2_point = generate_waypoint(start_point, dx1, dy1, dx2, dy2)
    waypoints_uav1.append(uav1_point)
    waypoints_uav2.append(uav2_point)
    
    # Generate path segments
    for segment in segments:
        num_points = round(segment["len"] / config.d_t)
        increment = np.array(segment["direction"]) * (segment["len"] / num_points)
        
        for _ in range(num_points):
            new_point = waypoints[-1] + increment
            waypoints.append(new_point)
            uav1_point, uav2_point = generate_waypoint(new_point, dx1, dy1, dx2, dy2)
            waypoints_uav1.append(uav1_point)
            waypoints_uav2.append(uav2_point)
    
    return waypoints, waypoints_uav1, waypoints_uav2

def generate_cross_path(config: RouteConfig) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Generate cross path"""
    dx1, dy1, dx2, dy2 = config.get_offsets()
    start_point = config.center - np.array([1.0, -1.0, 0.0])
    waypoints = [start_point]
    waypoints_uav1 = []
    waypoints_uav2 = []
    
    # Generate initial UAV positions
    uav1_point, uav2_point = generate_waypoint(start_point, dx1, dy1, dx2, dy2)
    waypoints_uav1.append(uav1_point)
    waypoints_uav2.append(uav2_point)
    
    # Generate diagonal path
    seq_num = 70
    d = (np.sqrt(8)/seq_num)/np.sqrt(2)
    
    for _ in range(seq_num):
        new_point = waypoints[-1] + np.array([d, -d, 0])
        waypoints.append(new_point)
        uav1_point, uav2_point = generate_waypoint(new_point, dx1, dy1, dx2, dy2)
        waypoints_uav1.append(uav1_point)
        waypoints_uav2.append(uav2_point)
    
    return waypoints, waypoints_uav1, waypoints_uav2

def save_waypoints(waypoints: List[np.ndarray], waypoints_uav1: List[np.ndarray], 
                  waypoints_uav2: List[np.ndarray], config: RouteConfig):
    """Save waypoints to file"""
    output_path = '/home/wawa/catkin_meta/src/MBRL_transport/'
    filename = f'save_waypoints_collision_{config.route_name}_{config.task_num}.mat'
    savemat(output_path + filename, 
            mdict={'load': waypoints, 'uav1': waypoints_uav1, 'uav2': waypoints_uav2})

def visualize_paths(waypoints: List[np.ndarray], waypoints_uav1: List[np.ndarray], 
                    waypoints_uav2: List[np.ndarray]):
    """Visualize the generated paths"""
    load_path = np.array(waypoints)
    uav1_path = np.array(waypoints_uav1)
    uav2_path = np.array(waypoints_uav2)

    plt.figure(figsize=(10, 8))
    
    # Plot XY plane
    plt.subplot(2, 1, 1)
    plt.plot(load_path[:, 0], load_path[:, 1], label='Load Path (XY)', marker='.')
    plt.plot(uav1_path[:, 0], uav1_path[:, 1], label='UAV1 Path (XY)', marker='x')
    plt.plot(uav2_path[:, 0], uav2_path[:, 1], label='UAV2 Path (XY)', marker='o')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Path Visualization (XY Plane)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Plot XZ plane (assuming Z is mostly constant or relevant for some paths)
    # If Z is always constant, this plot might not be very informative for XY paths
    plt.subplot(2, 1, 2)
    plt.plot(load_path[:, 0], load_path[:, 2], label='Load Path (XZ)', marker='.')
    plt.plot(uav1_path[:, 0], uav1_path[:, 2], label='UAV1 Path (XZ)', marker='x')
    plt.plot(uav2_path[:, 0], uav2_path[:, 2], label='UAV2 Path (XZ)', marker='o')
    plt.xlabel('X coordinate')
    plt.ylabel('Z coordinate')
    plt.title('Path Visualization (XZ Plane)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    # Example: Visualize the 'cross' path for task 0
    config = RouteConfig(route_name="square_xy", task_num=0, visualize=True) 
    
    if config.route_name == "square_xy":
        waypoints, waypoints_uav1, waypoints_uav2 = generate_square_xy_path(config)
    elif config.route_name == "cross":
        waypoints, waypoints_uav1, waypoints_uav2 = generate_cross_path(config)
    else:
        raise ValueError(f"Unknown route name: {config.route_name}")
    
    save_waypoints(waypoints, waypoints_uav1, waypoints_uav2, config)

    if config.visualize:
        visualize_paths(waypoints, waypoints_uav1, waypoints_uav2)

if __name__ == "__main__":
    main()


    


    
