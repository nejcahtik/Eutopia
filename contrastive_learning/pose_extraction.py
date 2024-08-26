from pose_format import Pose
from pose_format.torch import TorchPoseBody



f = open('../dgs_corpus/poses/1413925_1a1.pose', "rb")
pose = Pose.read(f.read(), TorchPoseBody)

numpy_data = pose.body.data

print(4)
