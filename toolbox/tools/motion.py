import numpy as np
from typing import List, Tuple, Dict


class MotionTools:
    @staticmethod
    def flow_magnitude_map(flow: np.ndarray) -> np.ndarray:
        return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    @staticmethod
    def motion_peak_detector(magnitudes: List[float], threshold: float) -> List[int]:
        return [i for i, mag in enumerate(magnitudes) if mag > threshold]
    
    @staticmethod
    def trajectory_builder(track_data: List[Dict], track_id: int) -> List[Tuple]:
        trajectory = []
        for det in track_data:
            if det.get('track_id') == track_id:
                bbox = det['bbox']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                trajectory.append((cx, cy, det['frame_id']))
        return trajectory
    
    @staticmethod
    def velocity_calculator(trajectory: List[Tuple]) -> List[float]:
        velocities = []
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            velocities.append(np.sqrt(dx**2 + dy**2))
        return velocities
    
    @staticmethod
    def takeoff_landing_detector(trajectory: List[Tuple]) -> int:
        y_coords = [point[1] for point in trajectory]
        for i in range(1, len(y_coords)):
            if y_coords[i] < y_coords[i-1] - 10:
                return trajectory[i][2]
        return -1
