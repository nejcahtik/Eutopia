import cv2
import random
import numpy as np
from pytorch_metric_learning.losses import NTXentLoss
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from moviepy.editor import *
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
import pose_format.utils.siren as siren
from numpy import ma
from pose_format.pose_visualizer import PoseVisualizer
import torch.nn.functional as F


class PoseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if smaller_set:
            self.pose_files_og = self.get_poses("../data_SSL/poses_t/")
        else:
            self.pose_files_og = self.get_poses("../data_SSL/poses/")

        if init_load:
            self.augmented_poses = {}

            print("loading poses")
            print(len(self.pose_files_aug))

            for path in self.pose_files_aug:
                self.augmented_poses[path] = self.load_pose(path)
            print("loading poses finished")

    def get_poses(self, data_dir):
        pose_files = []
        pose_files_dict = {}

        for f in os.listdir(data_dir):
            name = data_dir + f

            if name not in pose_files_dict and not name.endswith("_mov.pose"):
                pose_files_dict[name] = True
                pose_files.append(name)

        for i in range(len(pose_files)):
            if i % 2 == 0:
                pose_files[i] = pose_files[i][:-5] + "_mov.pose"

        return pose_files

    def __len__(self):
        return len(self.pose_files_og)

    def __getitem__(self, idx):
        return self.pose_files_og[idx]

    def load_pose(self, pose_path):
        pass


class TestPoseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if smaller_set:
            self.pose_files_og = self.get_poses("../data_SSL/poses_t/")
        else:
            self.pose_files_og = self.get_poses("../data_SSL/poses/")

        if init_load:
            self.augmented_poses = {}

            print("loading poses")
            print(len(self.pose_files_aug))

            for path in self.pose_files_aug:
                self.augmented_poses[path] = self.load_pose(path)
            print("loading poses finished")

    def get_poses(self, data_dir):
        pose_files = []
        pose_files_dict = {}

        for f in os.listdir(data_dir):
            name = data_dir + f

            if name not in pose_files_dict and not name.endswith("_mov.pose"):
                pose_files_dict[name] = True
                pose_files.append(name)

        for i in range(len(pose_files)):

            if i % 2 == 1:
                pose_files[i] = pose_files[i][:-5] + "_mov.pose"

        return pose_files

    def __len__(self):
        return len(self.pose_files_og)

    def __getitem__(self, idx):
        return self.pose_files_og[idx]

    def load_pose(self, pose_path):
        pass


class Rnn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Rnn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 256)

        self.mlp1 = nn.Linear(128, 256)
        self.relu1 = nn.ReLU()
        self.mlp2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.mlp3 = nn.Linear(128, 32)
        self.relu3 = nn.ReLU()

    def equalize_poses(self, poses):

        ps = []
        i = 0
        for pose in poses:
            no_of_frames = 250
            if pose.size(0) < no_of_frames:
                last_frame = pose[-1, :, :, :].unsqueeze(0)
                first_frame = pose[-1, :, :, :].unsqueeze(0)
                repeated_last_frame = last_frame.repeat(no_of_frames - pose.size(0), 1, 1, 1)
                ps.append(torch.cat((pose, repeated_last_frame), dim=0))
            elif pose.size(0) > no_of_frames:
                ps.append(pose[:no_of_frames, :, :, :])
            else:
                ps.append(pose)

        return torch.stack(ps)

    def pose_to_siren_to_pose(self, p: Pose, fps=None) -> Pose:

        p.body.zero_filled()
        mu, std = p.normalize_distribution()
        net = siren.get_pose_siren(p, total_steps=3000, steps_til_summary=100, learning_rate=1e-4)
        new_fps = fps if fps is not None else p.body.fps
        coords = siren.PoseDataset.get_coords(time=len(p.body.data) / p.body.fps, fps=new_fps)
        pred = net(coords).cpu().numpy()
        pose_body = NumPyPoseBody(fps=new_fps, data=ma.array(pred), confidence=np.ones(shape=tuple(pred.shape[:3])))
        p = Pose(header=p.header, body=pose_body)
        p.unnormalize_distribution(mu, std)
        return p

    def change_fps(self, p):
        # self.save_pose(p, "before" + str(random.randint(1, 100)) + ".mp4")

        if p.body.fps == 50:
            p.body = p.body.frame_dropout_given_percent(0.5)[0]
            p.body.fps = 25


        p_body = p.body.frame_dropout_given_percent(np.random.uniform(0, 0.2))
        p.body = p_body[0]
        # self.save_pose(p, "after" + str(random.randint(1, 100)) + ".mp4")
        # frame_rate = 0  # duplicate every 'frame_rate' frame
        # if pose.body.data.shape[0] < 60:
        #     frame_rate = random.randint(1, 5)
        # elif 60 <= pose.body.data.shape[0] <= 140:
        #     r = random.randint(0, 1)
        #     if r == 0:
        #         frame_rate = random.randint(-5, -2)
        #     else:
        #         frame_rate = random.randint(1, 5)
        # else:
        #     frame_rate = random.randint(-5, -2)
        #
        # new_data = []
        # new_confidence = []
        # new_mask = []
        #
        # for i in range(pose.body.data.shape[0]):
        #     if not i % (frame_rate*(-1)) == 0 and frame_rate < 0 or \
        #             i % frame_rate == 0 and frame_rate > 0:
        #         new_data.append(pose.body.data[i])
        #         new_confidence.append(pose.body.confidence[i])
        #         new_mask.append(pose.body.mask[i])
        #
        # pose.body.data = np.array(new_data)
        # pose.body.confidence = np.ma.MaskedArray(new_confidence)
        # # pose.body.mask = np.array(new_mask)
        #
        # if frame_rate < 0:
        #     pose.body.fps = pose.body.fps * (1 - 1/frame_rate)
        # else:
        #     pose.body.fps = pose.body.fps * (1 + 1/frame_rate)

    def save_pose(self, pose, file_name):
        v = PoseVisualizer(pose)
        v.save_video(file_name, v.draw())

    def get_augmentations_from_batch(self, batch):
        poses = []
        i = 0

        for item_path in batch:
            f = open(item_path, "rb")
            p = Pose.read(f.read())
            # pose = self.pose_to_siren_to_pose(pose)
            # header = pose.header
            # pose.normalize(pose.header.normalization_info(
            #     p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
            #     p2=("LEFT_HAND_LANDMARKS", "WRIST")
            # ))
            # pose.normalize_distribution()
            pose = p.augment2d(rotation_std=random.random() / 8, shear_std=random.random() / 8,
                                  scale_std=random.random() / 8)
            self.change_fps(pose)
            i += 1

            # pose = pose.interpolate_fps(24, kind='linear')
            pose = pose.body.data.filled(0)

            if pose.shape[0] < 250:
                frames_missing = 250 - pose.shape[0]

                frames_missing = frames_missing // 2
                first_frame_stack = []
                for j in range(frames_missing):
                    first_frame_stack.append(pose[0])

                pose = np.concatenate((first_frame_stack, pose))

                while pose.shape[0] < 250:
                    pose = np.concatenate((pose, [pose[-1]]))

            elif pose.shape[0] > 250:
                diff = pose.shape[0] - 250
                diff = diff/2
                pose = pose[diff:diff + 250]

            assert(pose.shape[0] == 250)

            pp = torch.tensor(pose).to(torch.float32)
            poses.append(pp)
            p.body.data = np.ma.MaskedArray(pose)
            self.save_pose(p, "augmented_"+str(random.randint(1,100))+".mp4")

        return torch.stack(poses)

    def load_poses(self, batch):
        poses = []
        for item_path in batch:
            f = open(item_path, "rb")
            pose = Pose.read(f.read(), NumPyPoseBody)
            pose = torch.tensor(pose.body.data.filled(0)).to(torch.float32)
            poses.append(pose)

        return self.equalize_poses(poses)

    def forward(self, batch, train=True):
        h0 = torch.zeros(self.num_layers, len(batch), self.hidden_size)
        c0 = torch.zeros(self.num_layers, len(batch), self.hidden_size)

        if train:

            aug1 = self.get_augmentations_from_batch(batch)
            aug2 = self.get_augmentations_from_batch(batch)
            aug3 = self.get_augmentations_from_batch(batch)
            aug4 = self.get_augmentations_from_batch(batch)

            aug_reshaped = aug1.view(aug1.size(0), aug1.size(1), -1)
            out = self.lstm(aug_reshaped, (h0, c0))
            out = out[0][:, -1, :]
            out1 = self.fc(out)

            aug_reshaped = aug2.view(aug2.size(0), aug2.size(1), -1)
            out = self.lstm(aug_reshaped, (h0, c0))
            out = out[0][:, -1, :]
            out2 = self.fc(out)

            aug_reshaped = aug3.view(aug3.size(0), aug3.size(1), -1)
            out = self.lstm(aug_reshaped, (h0, c0))
            out = out[0][:, -1, :]
            out3 = self.fc(out)

            aug_reshaped = aug4.view(aug4.size(0), aug4.size(1), -1)
            out = self.lstm(aug_reshaped, (h0, c0))
            out = out[0][:, -1, :]
            out4 = self.fc(out)


            out1 = F.adaptive_max_pool1d(out1, 128)
            out2 = F.adaptive_max_pool1d(out2, 128)
            out3 = F.adaptive_max_pool1d(out3, 128)
            out4 = F.adaptive_max_pool1d(out4, 128)

        else:
            b_reshaped = self.load_poses(batch)
            b_reshaped = b_reshaped.view(b_reshaped.size(0), b_reshaped.size(1), -1)
            out = self.lstm(b_reshaped, (h0, c0))
            out = out[0][:, -1, :]
            out = self.fc(out)

            return F.adaptive_max_pool1d(out, output_size=128)

        out1 = self.mlp1(out1)
        out1 = self.relu1(out1)
        out1 = self.mlp2(out1)
        out1 = self.relu2(out1)
        out1 = self.mlp3(out1)
        out1 = self.relu3(out1)

        out2 = self.mlp1(out2)
        out2 = self.relu1(out2)
        out2 = self.mlp2(out2)
        out2 = self.relu2(out2)
        out2 = self.mlp3(out2)
        out2 = self.relu3(out2)

        out3 = self.mlp1(out3)
        out3 = self.relu1(out3)
        out3 = self.mlp2(out3)
        out3 = self.relu2(out3)
        out3 = self.mlp3(out3)
        out3 = self.relu3(out3)

        out4 = self.mlp1(out4)
        out4 = self.relu1(out4)
        out4 = self.mlp2(out4)
        out4 = self.relu2(out4)
        out4 = self.mlp3(out4)
        out4 = self.relu3(out4)

        return out1, out2, out3, out4


# save representation of a pose (aka vec) to a dictionary
# so that when testing test files this dictionary can be used
# as an anchor point for representation of test files
def store_coords(vec, pose_data_paths, train_coords):
    i = 0
    for pose_data_path in pose_data_paths:
        pose_name = pose_data_path.split("/")[-1]

        if pose_name.endswith("_mov.pose"):
            pose_name = pose_name[:-9]
        else:
            pose_name = pose_name[:-5]

        train_coords[pose_name] = vec[i]
        i += 1
    return train_coords


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)


def is_correctly_classified(train_coords, output, pose_name):
    min_dist = -1
    max_dist = -1
    pred = ""
    max_pred = ""
    dists = []
    for key in train_coords:
        dist = cosine_similarity(train_coords[key], output)
        dists.append(dist)
        if min_dist == -1 or dist < min_dist:
            pred = key
            min_dist = dist
        if dist > max_dist:
            max_pred = key
            max_dist = dist

    legit_dist = cosine_similarity(train_coords[pose_name], output)
    dists.sort(reverse=True)
    legit_ix = -1

    for i in range(len(dists)):
        if dists[i] == legit_dist:
            legit_ix = i

    if max_pred == pose_name:
        return min_dist, True, max_dist, legit_ix
    return min_dist, False, max_dist, legit_ix


def get_errors(data, out):
    batch_error = 0
    correctly_classified = []
    legit_ixs = []
    for i in range(len(data)):
        pose_path = data[i]
        pose_name = pose_path.split("/")[-1]
        if pose_name.endswith("_mov.pose"):
            pose_name = pose_name[:-9]
        else:
            pose_name = pose_name[:-5]
        batch_error += cosine_similarity(train_coords[pose_name], out[i])
        dist, icc, max_dist, legit_ix = is_correctly_classified(train_coords, out[i], pose_name)
        legit_ixs.append(legit_ix)
        print("min_dist: " + str(dist) +
              ", correct_distance: " + str(cosine_similarity(train_coords[pose_name], out[i])) + ", max_dist: " + str(
            max_dist)
              + ", legit_ix: " + str(legit_ix))
        correctly_classified.append(icc)

    ai = sum(legit_ixs) / len(legit_ixs)

    # print("----------------------------------------")
    # print(correctly_classified)
    # print("----------------------------------------")
    return batch_error, ai


temperature = 0.9
loss_func = NTXentLoss(temperature=temperature)

init_load = False
smaller_set = False
batch_size = 32

device = torch.device("cpu")
model = Rnn(576 * 3, 256, 1)

# model = Cnn()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
dataset = PoseDataset("")
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_dataset = TestPoseDataset("")
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

torch.set_num_threads(8)
print("num of threads: " + str(torch.get_num_threads()))


def get_ix_matrix(data_len):
    m = []
    for i in range(data_len, data_len * 2):
        m.append(i)
    for i in range(0, data_len):
        m.append(i)
    return m


def print_gtl(gtl):
    for i in range(gtl.size(0)):
        for j in range(gtl.size(1)):
            print(gtl[i][j], end=" ")
        print("\n")


def train():
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(data_loader)):
        optimizer.zero_grad()

        # Get data representations
        out1, out2, out3, out4 = model(data)
        # Prepare for loss
        embeddings = torch.cat((out1, out2, out3, out4))
        # The same index corresponds to a positive pair
        indices = torch.arange(0, out1.size(0))
        labels = torch.cat((indices, indices, indices, indices))
        # embeddings = torch.tensor([[1.0,2.0,3.0,4.0], [110.0,111.0,112.0,113.0], [220.0,221.0,222.0,223.0], [330.0, 331.0, 332.0, 333.0], [1.0,2.0,3.0,4.0], [110.0,111.0,112.0,113.0], [220.0,221.0,222.0,223.0], [330.0, 331.0, 332.0, 333.0]])
        # labels = torch.tensor([0,1,2,3,0,1,2,3])

        # xcs = F.cosine_similarity(embeddings[None,:,:], embeddings[:,None,:], dim=-1)
        # xcs[torch.eye(len(data)*2).bool()] = float("-inf")
        # target = torch.tensor(get_ix_matrix(len(data)))
        # # print_gtl(ground_truth_labels)
        #
        # loss2 = F.cross_entropy(xcs/temperature, target, reduction="mean")

        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()


for epoch in range(1, 20):
    train()
    scheduler.step()

    if epoch % 1 == 0:
        model.eval()
        train_coords = {}
        with torch.no_grad():
            correct = 0
            total = 0
            for _, data in enumerate(tqdm.tqdm(data_loader)):
                out = model(data, train=False)
                train_coords = store_coords(out, data, train_coords)

            error = 0
            avg_ixs = []
            for _, data in enumerate(tqdm.tqdm(test_data_loader)):
                out_test = model(data, train=False)
                e, avg_ix = get_errors(data, out_test)
                error += e
                avg_ixs.append(avg_ix)

            avg_ix_final = sum(avg_ixs) / len(avg_ixs)
            print(f'Cosine similarity : {error}')
            print(f'Avg ix : {avg_ix_final}')
