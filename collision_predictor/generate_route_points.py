#This file is going to generate the route points for different paths
#collision free: figure8 xy, square xz
#collision: square, cross
import scipy
import numpy as np
from scipy.io import savemat
from openai_ros.task_envs.task_commons import Trajectory, Metrics, figure8_trajectory,figure8_trajectory_3d,figure8_trajectory_3d_xy
import matplotlib.pyplot as plt

# square_xy cross
route_name = "cross" 

#cc task0:wind 0.0 L0.6
# task1:wind 0.3 L1.0
# task2:wind 0.5 L0.8
# task3:wind 0.8 L1.2

task_num = 2

if task_num == 0:
    dx1 = 0.3024
    dy1 = 0.0786

    dx2 = -0.294
    dy2 = 0.0662

if task_num == 1:
    dx1 = 0.2164
    dy1 = -0.0198

    dx2 = -0.4932
    dy2 = -0.018

if task_num == 2:
    dx1 = 0.4876
    dy1 = -0.01

    dx2 = -0.5756
    dy2 = -0.0092

#transform into 256x256 grid
if route_name == "square_xy":
    d_t = 0.02
    len_x = 2
    len_y = 2

    len_y1 = 0.1
    len_x1 = 0.4
    len_y2 = 0.8
    len_x2 = 0.4
    len_y3 = 1.1

    len_x3 = 1.5
    len_y4 = 0.5
    len_x4 = 0.5
    len_y5 = 1.5

    num_pts_per_side_x = round(len_x/d_t)  # 30 seconds
    num_pts_per_side_y = round(len_y/d_t)  # 15 seconds

    num_pts_per_side_y1 = round(len_y1/d_t)
    num_pts_per_side_x1 = round(len_x1/d_t)
    num_pts_per_side_y2 = round(len_y2/d_t)
    num_pts_per_side_x2 = round(len_x2/d_t)
    num_pts_per_side_y3 = round(len_y3/d_t)

    num_pts_per_side_x3 = round(len_x3/d_t)
    num_pts_per_side_y4 = round(len_y4/d_t)
    num_pts_per_side_x4 = round(len_x4/d_t)
    num_pts_per_side_y5 = round(len_y5/d_t)

    center = np.array([2.0, 0.0, 0.8])
    inc_x = len_x / num_pts_per_side_x
    inc_y = len_y / num_pts_per_side_y

    inc_x1 = len_x1 / num_pts_per_side_x1
    inc_y1 = len_y1 / num_pts_per_side_y1
    inc_x2 = len_x2 / num_pts_per_side_x2
    inc_y2 = len_y2 / num_pts_per_side_y2
    inc_x3 = len_x3 / num_pts_per_side_x3
    inc_y3 = len_y3 / num_pts_per_side_y3
    inc_x4 = len_x4 / num_pts_per_side_x4
    inc_y4 = len_y4 / num_pts_per_side_y4
    inc_y5 = len_y5 / num_pts_per_side_y5

    #change here if you have three actions
    waypoints = [center - np.array([len_x / 2.0, -len_y / 2.0, 0.0])]  # start
    waypoints_uav1_p = np.zeros(3)
    waypoints_uav1_p[0] = waypoints[-1][0]+dx1
    waypoints_uav1_p[1] = waypoints[-1][1]+dy1
   
    waypoints_uav1 = [waypoints_uav1_p]
    waypoints_uav2_p = np.zeros(3) 
    waypoints_uav2_p[0] = waypoints[-1][0]+dx2
    waypoints_uav2_p[1] = waypoints[-1][1]+dy2
    
    waypoints_uav2 = [waypoints_uav2_p]

    for i in range(num_pts_per_side_x):
        path_component = waypoints[-1] + np.array([inc_x, 0, 0])
        waypoints += [path_component]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
        
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
        
        waypoints_uav2 += [waypoints_uav2_p]

    for i in range(num_pts_per_side_y1):
        path_component1 = waypoints[-1] + np.array([0, -inc_y1, 0])
        waypoints += [path_component1]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
        
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
       
        waypoints_uav2 += [waypoints_uav2_p]

    for i in range(num_pts_per_side_x1):
        path_component2 = waypoints[-1] + np.array([-inc_x1, 0, 0])
        waypoints += [path_component2]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
        
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
        
        waypoints_uav2 += [waypoints_uav2_p]

    for i in range(num_pts_per_side_y2):
        path_component3 = waypoints[-1] + np.array([0, -inc_y2, 0])
        waypoints += [path_component3]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
       
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
        
        waypoints_uav2 += [waypoints_uav2_p]

    for i in range(num_pts_per_side_x2):
        path_component4 = waypoints[-1] + np.array([inc_x2, 0, 0])
        waypoints += [path_component4]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
        
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
       
        waypoints_uav2 += [waypoints_uav2_p]

    for i in range(num_pts_per_side_y3):
        path_component5 = waypoints[-1] + np.array([0, -inc_y3, 0])
        waypoints += [path_component5]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
        
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
       
        waypoints_uav2 += [waypoints_uav2_p]

    for i in range(num_pts_per_side_x3):
        path_component6 = waypoints[-1] + np.array([-inc_x3, 0, 0])
        waypoints += [path_component6]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
        
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
        
        waypoints_uav2 += [waypoints_uav2_p]

    for i in range(num_pts_per_side_y4):
        path_component7 = waypoints[-1] + np.array([0, inc_y4, 0])
        waypoints += [path_component7]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
        
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
        
        waypoints_uav2 += [waypoints_uav2_p]

    for i in range(num_pts_per_side_x4):
        path_component8 = waypoints[-1] + np.array([-inc_x4, 0, 0])
        waypoints += [path_component8]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
       
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
        
        waypoints_uav2 += [waypoints_uav2_p]

    for i in range(num_pts_per_side_y5):
        path_component9 = waypoints[-1] + np.array([0, inc_y5, 0])
        waypoints += [path_component9]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
        
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
        
        waypoints_uav2 += [waypoints_uav2_p]

    savemat('/home/wawa/catkin_meta/src/MBRL_transport/save_waypoints_collision_square_{0}.mat'.format(task_num), mdict={'load': waypoints,'uav1':waypoints_uav1,'uav2':waypoints_uav2})

if route_name == "cross":
    len_x = 2
    len_y = 2
    center = np.array([2.0, 0.0, 0.8])
    waypoints = [center - np.array([len_x / 2.0, -len_y / 2.0, 0.0])]
    seq_num = 70.0
    d = (np.sqrt(8)/seq_num)/np.sqrt(2)
    waypoints_uav1_p = np.zeros(3)
    waypoints_uav1_p[0] = waypoints[-1][0]+dx1
    waypoints_uav1_p[1] = waypoints[-1][1]+dy1
    
    waypoints_uav1 = [waypoints_uav1_p]
    waypoints_uav2_p = np.zeros(3) 
    waypoints_uav2_p[0] = waypoints[-1][0]+dx2
    waypoints_uav2_p[1] = waypoints[-1][1]+dy2
    
    waypoints_uav2 = [waypoints_uav2_p]

    for i in range(int(seq_num)):
        waypoints+=[waypoints[-1]+np.array([d,-d,0])]
        waypoints_uav1_p = np.zeros(3)
        waypoints_uav1_p[0] = waypoints[-1][0]+dx1
        waypoints_uav1_p[1] = waypoints[-1][1]+dy1
       
        waypoints_uav1 += [waypoints_uav1_p]
        waypoints_uav2_p = np.zeros(3) 
        waypoints_uav2_p[0] = waypoints[-1][0]+dx2
        waypoints_uav2_p[1] = waypoints[-1][1]+dy2
        
        waypoints_uav2 += [waypoints_uav2_p]

    savemat('/home/wawa/catkin_meta/src/MBRL_transport/save_waypoints_collision_cross_{0}.mat'.format(task_num), mdict={'load': waypoints,'uav1':waypoints_uav1,'uav2':waypoints_uav2})


# plt.scatter(np.array(waypoints)[:,0],np.array(waypoints)[:,1],c='r')  
# plt.scatter(np.array(waypoints_uav1)[:,0],np.array(waypoints_uav1)[:,1],c='g') 
# plt.scatter(np.array(waypoints_uav2)[:,0],np.array(waypoints_uav2)[:,1],c='b') 
# plt.xlim(0, 4)
# # plt.ylim(-2, 2)
# plt.axis('equal')
# plt.show()
# if route_name == "figure8_1":
#     waypoints = figure8_trajectory_3d_xy(2.0, 0.0, 0.35,num_points_per_rot=100)
#     savemat('/home/wawa/catkin_meta/src/MBRL_transport/save_waypoints_figure8xy.mat', mdict={'arr': waypoints})

# if route_name == "square_xz":  
#     # iterates along sides
#     d_t = 0.02
#     len_x = 2
#     len_z = 1.0
#     len_x_1 = 1.7
#     len_x_2 = 0.3
#     len_z_1 = 0.3
#     len_z_2 = 0.7

#     num_pts_per_side_x = round(len_x/d_t)  # 30 seconds
#     num_pts_per_side_x1 = round(len_x_1/d_t)  # 30 seconds
#     num_pts_per_side_x2 = round(len_x_2/d_t) # 30 seconds
#     num_pts_per_side_z = round(len_z/d_t)  # 15 seconds
#     num_pts_per_side_z1 = round(len_z_1/d_t)  # 15 seconds
#     num_pts_per_side_z2 = round(len_z_2/d_t)  # 15 seconds

#     center = np.array([2.0, 0.0, 0.8])
#     inc_x = len_x / num_pts_per_side_x
#     inc_x1 = len_x_1 / num_pts_per_side_x1
#     inc_x2 = len_x_2 / num_pts_per_side_x2
#     inc_z = len_z / num_pts_per_side_z
#     inc_z1 = len_z_1 / num_pts_per_side_z1
#     inc_z2 = len_z_2 / num_pts_per_side_z2
#     #change here if you have three actions
#     waypoints = [center - np.array([len_x / 2.0, 0.0, -len_z / 2.0])]  # start

#     # clockwise
#     for i in range(num_pts_per_side_x):
#         path_component = waypoints[-1] + np.array([inc_x, 0, 0])
#         waypoints += [path_component]

#     for i in range(num_pts_per_side_z1):
#         path_component1 = waypoints[-1] + np.array([0,0, -inc_z1])
#         waypoints += [path_component1]
            
#     for i in range(num_pts_per_side_x1):
#         path_component2 = waypoints[-1] + np.array([-inc_x1, 0, 0])
#         waypoints += [path_component2]

#     for i in range(num_pts_per_side_z2):
#         path_component3 = waypoints[-1] + np.array([0,0, -inc_z2])
#         waypoints += [path_component3]

#     for i in range(num_pts_per_side_x2):
#         path_component4 = waypoints[-1] + np.array([-inc_x2,0, 0])
#         waypoints += [path_component4]

#     for i in range(num_pts_per_side_z):
#         path_component5 = waypoints[-1] + np.array([0, 0, inc_z])
#         waypoints += [path_component5]
    
#     savemat('/home/wawa/catkin_meta/src/MBRL_transport/save_waypoints_square_xz.mat', mdict={'arr': waypoints})


    


    
