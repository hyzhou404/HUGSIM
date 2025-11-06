import pickle
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from shapely.geometry import LineString, Point
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
import os
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as SCR

ego_verts_canonic = np.array([[0.5, 0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 1.0], [0.5, -0.5, 1.0],
                    [-0.5, -0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0]])

# Define boundaries
boundaries = {
    'max_abs_lat_accel': 4.89,  # [m/s^2]
    'max_lon_accel': 2.40,  # [m/s^2]
    'min_lon_accel': -4.05,  # [m/s^2]
    'max_abs_yaw_accel': 1.93,  # [rad/s^2]
    'max_abs_lon_jerk': 8.37,  # [m/s^3],
    'max_abs_yaw_rate': 0.95,  # [rad/s]
}

score_weight = {
    'ttc': 5,
    'c': 2,
    'ep': 5,
}

def create_rectangle(center_x, center_y, width, length, yaw):
    """Create a rectangle polygon."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    x_offs = [length/2, length/2, -length/2, -length/2]
    y_offs = [width/2, -width/2, -width/2, width/2]

    x_pts = [center_x + x_off*cos_yaw - y_off *
                sin_yaw for x_off, y_off in zip(x_offs, y_offs)]
    y_pts = [center_y + x_off*sin_yaw + y_off *
                cos_yaw for x_off, y_off in zip(x_offs, y_offs)]

    return ShapelyPolygon(zip(x_pts, y_pts))

def bg_collision_det(points, box):
    O, A, B, C = box[0], box[1], box[2], box[5]
    OA = A - O
    OB = B - O
    OC = C - O
    POA, POB, POC = (points @ OA[..., None])[:, 0], (points @ OB[..., None])[:, 0], (points @ OC[..., None])[:, 0]
    mask = (torch.dot(O, OA) < POA) & (POA < torch.dot(A, OA)) & \
        (torch.dot(O, OB) < POB) & (POB < torch.dot(B, OB)) & \
        (torch.dot(O, OC) < POC) & (POC < torch.dot(C, OC))
    return True if torch.sum(mask) > 100 else False

def calculate_trajectory_kinematics(traj, timestep):
    """
    Calculate kinematic parameters for a given trajectory.

    :param traj: List of tuples (x, y, yaw) for each point in the trajectory
    :param timestep: Time interval between each point in the trajectory
    :return: Dictionary containing lists of calculated parameters
    """
    # Convert trajectory to numpy array for easier calculations
    x, y, yaw = zip(*traj)
    x, y, yaw = np.array(x), np.array(y), np.array(yaw)

    # Calculate velocities
    dx = np.diff(x) / timestep
    dy = np.diff(y) / timestep

    # Calculate yaw rate
    dyaw = np.diff(yaw)
    dyaw = np.where(dyaw > np.pi, dyaw - 2*np.pi, dyaw)
    dyaw = np.where(dyaw < -np.pi, dyaw + 2*np.pi, dyaw)
    dyaw = dyaw / timestep
    ddyaw = np.diff(dyaw) / timestep

    # Calculate speed
    speed = np.sqrt(dx**2 + dy**2)

    # Calculate accelerations
    accel = np.diff(speed) / timestep
    jerk = np.diff(accel) / timestep

    # Calculate yaw rate (already calculated as dyaw)
    yaw_rate = dyaw
    # Calculate yaw acceleration
    yaw_accel = ddyaw

    lon_accel = accel
    lat_accel = np.zeros_like(lon_accel)
    lon_jerk = jerk

    # Pad arrays to match the original trajectory length
    yaw_rate = np.pad(yaw_rate, (0, 1), 'edge')
    yaw_accel = np.pad(yaw_accel, (0, 2), 'edge')
    lon_accel = np.pad(lon_accel, (0, 2), 'edge')
    lat_accel = np.pad(lat_accel, (0, 2), 'edge')
    lon_jerk = np.pad(lon_jerk, (0, 3), 'edge')

    return {
        'speed': speed,
        'yaw_rate': yaw_rate,
        'yaw_accel': yaw_accel,
        'lon_accel': lon_accel,
        'lat_accel': lat_accel,
        'lon_jerk': lon_jerk,
    }


class ScoreCalculator:
    def __init__(self, data):
        self.data = data

        self.pdms = 0.0
        self.driving_score = None
        pass

    def transform_to_ego_frame(self, traj, ego_box):
        """
        Transform trajectory from global frame to ego-centric frame.

        :param traj: List of tuples (x, y, yaw) in global frame
        :param ego_box: Tuple (x, y, z, w, l, h, yaw) of ego vehicle in global frame
        :return: Numpy array of transformed trajectory
        """
        ego_x, ego_y, _, _, _, _, ego_yaw = ego_box

        # Create rotation matrix
        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        R = np.array([[c, -s], [s, c]])

        # Transform each point
        transformed_traj = []
        for x, y, yaw in traj:
            # Translate
            x_translated, y_translated = x - ego_x, y - ego_y

            # Rotate
            x_rotated, y_rotated = R @ np.array([x_translated, y_translated])

            # Adjust yaw
            yaw_adjusted = yaw - ego_yaw

            transformed_traj.append((x_rotated, y_rotated, yaw_adjusted))

        return np.array(transformed_traj)

    def get_vehicle_corners(self, x, y, yaw, length, width):
        """
        Calculate the corner points of the vehicle given its position, orientation, and dimensions.

        :param x: x-coordinate of the vehicle's center
        :param y: y-coordinate of the vehicle's center
        :param yaw: orientation of the vehicle in radians
        :param length: length of the vehicle
        :param width: width of the vehicle
        :return: numpy array of corner coordinates (4x2)
        """
        c, s = np.cos(yaw), np.sin(yaw)
        front_left = np.array([x + c * length / 2 - s * width / 2,
                               y + s * length / 2 + c * width / 2])
        front_right = np.array([x + c * length / 2 + s * width / 2,
                                y + s * length / 2 - c * width / 2])
        rear_left = np.array([x - c * length / 2 - s * width / 2,
                              y - s * length / 2 + c * width / 2])
        rear_right = np.array([x - c * length / 2 + s * width / 2,
                               y - s * length / 2 - c * width / 2])
        return np.array([front_left, front_right, rear_right, rear_left])

    def plot_trajectory_on_drivable_mask(self, drivable_mask, transformed_traj, vehicle_width, vehicle_length):
        """
        Plot the transformed trajectory and vehicle bounding boxes on the drivable mask.

        :param drivable_mask: 2D numpy array representing the drivable area (200x200)
        :param transformed_traj: Numpy array of transformed trajectory points
        :param vehicle_width: Width of the vehicle in meters
        :param vehicle_length: Length of the vehicle in meters
        """

        plt.figure(figsize=(10, 10))
        plt.imshow(drivable_mask, cmap='gray', extent=[-50, 50, -50, 50])

        # Scale factor (200 pixels represent 100 meters)
        scale_factor = 200 / 100  # pixels per meter

        # Plot trajectory
        x_coords, y_coords, yaws = transformed_traj.T
        plt.plot(x_coords, y_coords, 'r-', linewidth=2)

        # Plot vehicle bounding boxes
        for x, y, yaw in transformed_traj:
            corners = self.get_vehicle_corners(
                x, y, yaw, vehicle_length, vehicle_width)
            plt.gca().add_patch(Polygon(corners, fill=False, edgecolor='blue'))

        # Plot start and end points
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'bo', markersize=10, label='End')

        plt.title('Trajectory and Vehicle Bounding Boxes on Drivable Mask')
        plt.legend()
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def _calculate_drivable_area_compliance(self, ground, traj, vehicle_width, vehicle_length):
        m, n = 2, 2
        dac = 1.0
        for traj_i, (x, y, yaw) in enumerate(traj):
            cnt = 0
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            ground_in_ego = (np.linalg.inv(R) @ (ground + np.array([-x, -y])).T).T
            x_bins = np.linspace(-vehicle_length/2, vehicle_length/2, m+1)
            y_bins = np.linspace(-vehicle_width/2, vehicle_width/2, n+1)
            for xi in range(m):
                for yi in range(n):
                    min_x, max_x = x_bins[xi], x_bins[xi+1]
                    min_y, max_y = y_bins[yi], y_bins[yi+1]
                    ground_mask = (min_x < ground_in_ego[:, 0]) & (ground_in_ego[:, 0] < max_x) & \
                                    (min_y < ground_in_ego[:, 1]) & (ground_in_ego[:, 1] < max_y)
                    if ground_mask.sum() > 0:
                        cnt += 1
            drivable_ratio = cnt / (m*n)
            if drivable_ratio < 0.3:
                return 0.0
            elif drivable_ratio < 0.5:
                dac = 0.5
        return dac
    
    def _single_frame_drivable_area_compliance(self, ground, ego, vehicle_width, vehicle_length):
        m, n = 2, 2
        dac = 1.0
        x, y, _, _, _, _, yaw = ego
        cnt = 0
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        ground_in_ego = (np.linalg.inv(R) @ (ground + np.array([-x, -y])).T).T
        x_bins = np.linspace(-vehicle_length/2, vehicle_length/2, m+1)
        y_bins = np.linspace(-vehicle_width/2, vehicle_width/2, n+1)
        for xi in range(m):
            for yi in range(n):
                min_x, max_x = x_bins[xi], x_bins[xi+1]
                min_y, max_y = y_bins[yi], y_bins[yi+1]
                ground_mask = (min_x < ground_in_ego[:, 0]) & (ground_in_ego[:, 0] < max_x) & \
                                (min_y < ground_in_ego[:, 1]) & (ground_in_ego[:, 1] < max_y)
                if ground_mask.sum() > 0:
                    cnt += 1
        drivable_ratio = cnt / (m*n)
        if drivable_ratio < 0.3:
            return 0.0
        elif drivable_ratio < 0.5:
            dac = 0.5
        return dac

    def _calculate_progress(self, planned_traj, ref_taj):
        def calculate_curve_length(points):
            """Calculate the total length of a curve given by a set of points."""
            curve = LineString(points)
            return curve.length

        def project_curve_onto_curve(curve_a, curve_b):
            """Project curve_b onto curve_a and calculate the projected length."""
            projected_points = []
            for point in curve_b.coords:
                projected_point = curve_a.interpolate(
                    curve_a.project(Point(point)))
                projected_points.append(projected_point)
            projected_curve = LineString(projected_points)
            return projected_curve.length

        # Create Shapely LineString objects
        plan_curve = LineString([(x, y) for x, y, _ in planned_traj])
        ref_curve = LineString([(x, y) for x, y, _ in ref_taj])

        # Calculate lengths
        plan_curve_length = calculate_curve_length(plan_curve)
        ref_curve_length = calculate_curve_length(ref_curve)
        projected_length = project_curve_onto_curve(ref_curve, plan_curve)
        # print(f"plan_curve_length: {plan_curve_length}, ref_curve_length: {ref_curve_length}, project plan to ref_length: {projected_length}")

        ep = 0.0
        if max(plan_curve_length, ref_curve_length) < 5.0 or ref_curve_length < 1e-6:
            ep = 1.0
        else:
            ep = projected_length / ref_curve_length
        return ep

    def _calculate_is_comfortable(self, traj, timestep):
        """
        Check if all kinematic parameters of a trajectory are within specified boundaries.

        :param traj: List of tuples (x, y, yaw) representing the trajectory, in ego's local frame
        :param timestep: Time interval between trajectory points in seconds
        :return: 1.0 if all parameters are within boundaries, 0.0 otherwise
        """

        # Calculate kinematic parameters
        if len(traj) < 4:
            return 1.0

        kinematics = calculate_trajectory_kinematics(traj, timestep)

        # Check each parameter against its boundary
        checks = [
            np.all(np.abs(kinematics['lat_accel']) <=
                   boundaries['max_abs_lat_accel']),
            np.all(kinematics['lon_accel'] <= boundaries['max_lon_accel']),
            np.all(kinematics['lon_accel'] >= boundaries['min_lon_accel']),
            np.all(np.abs(kinematics['lon_jerk']) <=
                   boundaries['max_abs_lon_jerk']),
            np.all(np.abs(kinematics['yaw_accel']) <=
                   boundaries['max_abs_yaw_accel']),
            np.all(np.abs(kinematics['yaw_rate']) <=
                   boundaries['max_abs_yaw_rate'])
        ]

        # if not all(checks):
        #     print(traj)
        #     print(kinematics)
        print(f"comfortable: {checks}")

        # Return 1.0 if all checks pass, 0.0 otherwise
        return 1.0 if all(checks) else 0.0
    
    def _calculate_actual_comfort(self, traj, timestep):
        """
        Check if all kinematic parameters of a trajectory are within specified boundaries.

        :param traj: List of tuples (x, y, yaw) representing the trajectory, in ego's local frame
        :param timestep: Time interval between trajectory points in seconds
        :return: 1.0 if all parameters are within boundaries, 0.0 otherwise
        """

        kinematics = calculate_trajectory_kinematics(traj, timestep)

        # Check each parameter against its boundary
        checks = [
            np.all(np.abs(kinematics['lat_accel']) <=
                   boundaries['max_abs_lat_accel']),
            np.all(kinematics['lon_accel'] <= boundaries['max_abs_lat_accel']),
            np.all(kinematics['lon_accel'] >= boundaries['min_lon_accel']),
            np.all(np.abs(kinematics['lon_jerk']) <=
                   boundaries['max_abs_lon_jerk']),
            np.all(np.abs(kinematics['yaw_accel']) <=
                   boundaries['max_abs_yaw_accel']),
            np.all(np.abs(kinematics['yaw_rate']) <=
                   boundaries['max_abs_yaw_rate'])
        ]
        
        lat_accel_check = np.abs(kinematics['lat_accel']) <= boundaries['max_abs_lat_accel']
        lon_accel_check = (kinematics['lon_accel'] >= boundaries['min_lon_accel']) & (kinematics['lon_accel'] <= boundaries['max_lon_accel'])
        lon_jerk_check = np.abs(kinematics['lon_jerk']) <= boundaries['max_abs_lon_jerk']
        yaw_accel_check = np.abs(kinematics['yaw_accel']) <= boundaries['max_abs_yaw_accel']
        yaw_rate_check = np.abs(kinematics['yaw_rate']) <= boundaries['max_abs_yaw_rate']
        checks = lat_accel_check & lon_accel_check & lon_jerk_check & yaw_accel_check & yaw_rate_check

        print(f"comfortable: {checks}")

        # Return 1.0 if all checks pass, 0.0 otherwise
        return checks

    def _calculate_no_collision(self, ego_box, planned_traj, obs_lists, scene_xyz):
        ego_x, ego_y, z, ego_w, ego_l, ego_h, ego_yaw = ego_box
        ego_verts_local = ego_verts_canonic * np.array([ego_l, ego_w, ego_h])
        for idx in range(planned_traj.shape[0]):
            ego_x, ego_y, ego_yaw = planned_traj[idx]  # ego_state= (x,y,yaw)
            ego_trans_mat = np.eye(4)
            ego_trans_mat[:3, :3] = SCR.from_euler('z', ego_yaw).as_matrix()
            ego_trans_mat[:3, 3] = np.array([ego_x, ego_y, z])
            ego_verts_global = (ego_trans_mat[:3, :3] @ ego_verts_local.T).T + ego_trans_mat[:3, 3]
            ego_verts_global = torch.from_numpy(ego_verts_global).float().cuda()
            bk_collision = bg_collision_det(scene_xyz, ego_verts_global)
            # scene_local = scene_xyz - np.array([ego_x, ego_y, z])
            # bk_collision = np.sum(
            #     (-ego_l/2 < scene_local[:, 0]) & (scene_local[:, 0] < ego_l/2) & \
            #     (-ego_w/2 < scene_local[:, 1]) & (scene_local[:, 1] < ego_w/2) & \
            #     (-ego_h/2 < scene_local[:, 2]) & (scene_local[:, 2] < ego_h/2)
            # ) > 100
            if bk_collision:
                print(f"collision with background detected! @ timestep{idx}")
                return 0.0
            ego_poly = create_rectangle(ego_x, ego_y, ego_w, ego_l, ego_yaw)
            obs_list = obs_lists[idx if idx < len(obs_lists) else -1]
            for obs in obs_list:
                # obs = (x,y,z,w,l,h,yaw)
                obs_x, obs_y, _, obs_w, obs_l, _, obs_yaw = obs
                obs_poly = create_rectangle(
                    obs_x, obs_y, obs_w, obs_l, obs_yaw)
                if ego_poly.intersects(obs_poly):
                    print(f"collision with obstacle detected! @ timestep{idx}")
                    print(
                        f"ego_poly: {(ego_x, ego_y, ego_yaw,obs_w, obs_l)}, obs_poly: {(obs_x, obs_y, obs_yaw,obs_w, obs_l )}")
                    return 0.0  # Collision detected
        return 1.0

    def _single_frame_no_collision(self, ego_box, obs_list, scene_xyz, timestamp):
        ego_x, ego_y, z, ego_w, ego_l, ego_h, ego_yaw = ego_box
        ego_verts_local = ego_verts_canonic * np.array([ego_l, ego_w, ego_h])
        ego_trans_mat = np.eye(4)
        ego_trans_mat[:3, :3] = SCR.from_euler('z', ego_yaw).as_matrix()
        ego_trans_mat[:3, 3] = np.array([ego_x, ego_y, z])
        ego_verts_global = (ego_trans_mat[:3, :3] @ ego_verts_local.T).T + ego_trans_mat[:3, 3]
        ego_verts_global = torch.from_numpy(ego_verts_global).float().cuda()
        bk_collision = bg_collision_det(scene_xyz, ego_verts_global)
        if bk_collision:
            print(f"collision with background detected! @ {timestamp}")
            return 0.0
        ego_poly = create_rectangle(ego_x, ego_y, ego_w, ego_l, ego_yaw)
        for obs in obs_list:
            # obs = (x,y,z,w,l,h,yaw)
            obs_x, obs_y, _, obs_w, obs_l, _, obs_yaw = obs
            obs_poly = create_rectangle(
                obs_x, obs_y, obs_w, obs_l, obs_yaw)
            if ego_poly.intersects(obs_poly):
                print(f"collision with obstacle detected! @ {timestamp}")
                print(
                    f"ego_poly: {(ego_x, ego_y, ego_yaw,obs_w, obs_l)}, obs_poly: {(obs_x, obs_y, obs_yaw,obs_w, obs_l )}")
                return 0.0  # Collision detected
        return 1.0

    def _calculate_time_to_collision(self, ego_box, planned_traj, obs_lists, scene_xyz, timestep):
        # breakpoint()
        t_list = [0.5, 1]  # ttc time

        for t in t_list:
            # Calculate velocities
            velocities = np.diff(planned_traj[:, :2], axis=0) / timestep

            # Use the velocity of the second point for the first point
            velocities = np.vstack([velocities[0], velocities])

            # Calculate the displacement
            displacement = velocities * t

            # Create the new trajectory
            new_traj = planned_traj.copy()
            new_traj[:, :2] += displacement

            is_collide_score = self._calculate_no_collision(
                ego_box, new_traj, obs_lists, scene_xyz)
            if is_collide_score == 0.0:
                print(f" failed to pass ttc collision check, t={t}")
                # breakpoint()
                return 0.0

        return 1.0

    def calculate(self, ):

        print(f"current exp has {len(self.data['frames'])} frames")
        if len(self.data['frames']) == 0:
            return None

        score_list = {}
        
        # ego_traj = []
        # for frame in self.data['frames']:
        #     x, y, _, _, _, _, yaw = frame['ego_box']
        #     ego_traj.append((x, y, yaw))
        # comfort = self._calculate_actual_comfort(
        #     ego_traj, 0.25
        # )
        
        for i in range(0, len(self.data['frames']), 1):
            frame = self.data['frames'][i]
            if frame['is_key_frame'] == False:
                continue

            print(f"frame {i} / {len(self.data['frames'])}")
            timestamp = frame['time_stamp']
            planned_last_timestamp = timestamp + \
                len(frame['planned_traj']['traj']) * \
                frame['planned_traj']['timestep']
            ego_x, ego_y, _, ego_w, ego_l, _, ego_yaw = frame['ego_box']
            # frame['planned_traj']['traj']
            if len(frame['planned_traj']['traj'])<2:
                continue
            traj = frame['planned_traj']['traj']

            planned_traj = np.concatenate(([np.array([ego_x, ego_y, ego_yaw])], traj), axis=0)
            # print(planned_traj)

            # if the car is stopped, there may be error in the yaw of the planned trajectory
            traj_distance =  np.linalg.norm(planned_traj[-1, :2]  - planned_traj[0, :2] )
            if traj_distance<1:
                 planned_traj[:, 2] = planned_traj[0, 2]  # set all yaw to the first yaw

            current_timestamp = timestamp
            current_frame_idx = i
            obs_lists = []
            while current_timestamp <= planned_last_timestamp+1e-5:
                if abs(current_timestamp - self.data['frames'][current_frame_idx]['time_stamp']) < 1e-5:
                    obs_list = []
                    for idx, obj in enumerate(self.data['frames'][current_frame_idx]['obj_boxes']):
                        # obs_list.append(obj)
                        if self.data['frames'][current_frame_idx]['obj_names'][idx] == 'car':
                            obs_list.append(obj)
                    obs_lists.append(obs_list)
                    current_timestamp += frame['planned_traj']['timestep']

                current_frame_idx += 1
                if current_frame_idx >= len(self.data['frames']):
                    break

            # breakpoint()
            # plt.imshow(frame['drivable_mask'].astype(np.uint8))
            # plt.show()
            # transformed_traj = self.transform_to_ego_frame(frame['planned_traj']['traj'], frame['ego_box'])
            transformed_traj = self.transform_to_ego_frame(
                planned_traj, frame['ego_box'])
            # breakpoint()

            score_nc = self._calculate_no_collision(
                frame['ego_box'], planned_traj, obs_lists, self.data['scene_xyz'])
            score_dac = self._calculate_drivable_area_compliance(
                self.data['ground_xy'], planned_traj, ego_w, ego_l)
            
            # score_nc = self._single_frame_no_collision(
            #     frame['ego_box'], obs_lists[0], self.data['scene_xyz'], timestamp)
            # # score_nc = 0.0 if frame['collision'] else 1.0
            # score_dac = self._single_frame_drivable_area_compliance(
            #     self.data['ground_xy'], frame['ego_box'], ego_w, ego_l)
            
            score_ttc = self._calculate_time_to_collision(
                frame['ego_box'], planned_traj, obs_lists, self.data['scene_xyz'], frame['planned_traj']['timestep'])
            
            # score_c = 1.0 if comfort[i] else 0.0
            score_c = self._calculate_is_comfortable(
                transformed_traj, frame['planned_traj']['timestep'])
            
            # score_ep = self._calculate_progress(
            #     planned_traj, ref_traj)

            score_pdms = score_nc*score_dac*(score_weight['ttc']*score_ttc+score_weight['c']*score_c)/(
                score_weight['ttc']+score_weight['c'])
            score_list[timestamp] = {'nc': score_nc, 'dac': score_dac,
                                     'ttc': score_ttc, 'c': score_c, 'pdms': score_pdms}
        
        totals = {metric: 0 for metric in next(iter(score_list.values()))}
        for scores in score_list.values():
            for metric, value in scores.items():
                totals[metric] += value

        # avg scores
        num_entries = len(score_list)
        averages = {metric: total / num_entries for metric,
                    total in totals.items()}

        #     writer.writerow(averages.values())

        mean_score = averages['pdms']
        route_completion = max([f['rc'] for f in self.data['frames']])
        route_completion = route_completion if route_completion < 1 else 1.0
        driving_score = mean_score*route_completion
        return mean_score, route_completion, driving_score, averages, score_list


def calculate(data):
    print(f"this pkl file contains {len(data)} experiment records.")
    # print(f"the first item metadata is {data[0]['metas']}.")
    # breakpoint()

    def process_exp_data(exp_data):
        score_calc = ScoreCalculator(exp_data)
        score = score_calc.calculate()
        print(f"The score of experiment is {score}.")
        final_score_dict = score[3]
        final_score_dict['rc'] = score[1]
        final_score_dict['hdscore'] = score[2]
        final_score_dict['details'] = score[-1]
        return final_score_dict
    
    assert len(data) == 1, "only support single experiment now"
    score_dict = process_exp_data(data[0])

    # Output Results
    print("=============================Results=============================")
    for key in ['nc', 'dac', 'ttc', 'c', 'pdms', 'rc', 'hdscore']:
        print(f"'{key}': {score_dict[key]}")
    return score_dict

def parse_data(test_path):
    data_file_name = os.path.join(test_path, "data.pkl")
    ground_pcd_file_name = os.path.join(test_path, "ground.ply")
    scene_pcd_file_name = os.path.join(test_path, "scene.ply")

    # Open the file and load the data
    with open(data_file_name, 'rb') as f:
        data = pickle.load(f)
    
    ground_pcd = o3d.io.read_point_cloud(ground_pcd_file_name)
    ground_xyz = np.asarray(ground_pcd.points) # in camera coordinates
    ground_xy = np.stack([ground_xyz[:, 2], -ground_xyz[:, 0]], axis=1) # in imu coordinates
    
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_file_name)
    scene_xyz = np.asarray(scene_pcd.points) # in camera coordinates
    # in imu coordinates
    scene_xyz = np.stack([scene_xyz[:, 2], -scene_xyz[:, 0], -scene_xyz[:, 1]], axis=1) 
    
    data[0]['ground_xy'] = ground_xy
    data[0]['scene_xyz'] = torch.from_numpy(scene_xyz).cuda()
    # data[0]['scene_xyz'] = scene_xyz
    return data


def hugsim_evaluate(test_data, ground_xyz, scene_xyz, new_score_weight=None):
    if new_score_weight is not None:
        global score_weight
        score_weight = new_score_weight
    ground_xy = np.stack([ground_xyz[:, 2], -ground_xyz[:, 0]], axis=1) # in imu coordinates
    scene_xyz = np.stack([scene_xyz[:, 2], -scene_xyz[:, 0], -scene_xyz[:, 1]], axis=1) 
    test_data[0]['ground_xy'] = ground_xy
    test_data[0]['scene_xyz'] = torch.from_numpy(scene_xyz).float().cuda()
    results = calculate(test_data)
    return results


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    data = parse_data(args.test_path)
    
    # Call the main function with the loaded data
    calculate(data)