from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
import random


with open("../data_SSL/poses_t/agua.pose", "rb") as f:
    pose = Pose.read(f.read())
    pose = pose.augment2d(rotation_std=0.2, shear_std=0.5, scale_std=0.5)

v = PoseVisualizer(pose)

v.save_video("example.mp4", v.draw())