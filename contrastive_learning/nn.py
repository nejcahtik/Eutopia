import cv2
import random
import numpy as np
from pytorch_metric_learning.losses import NTXentLoss
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from moviepy.editor import *


seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)


class VideoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.video_files_og = [f for f in os.listdir("../data_SSL/videos/") if f.endswith('.mp4')]
        if smaller_set:
            self.video_files_og = self.get_og_videos_not_all("../data_SSL/augmented_videos_t/")
        else:
            self.video_files_og = self.get_og_videos_not_all("../data_SSL/augmented_videos/")

        if init_load:
            self.augmented_videos = {}

            print("loading videos")
            print(len(self.video_files_aug))



            for path in self.video_files_aug:
                self.augmented_videos[path] = self.load_video(path)
            print("loading videos finished")

    def get_og_videos_not_all(self, data_dir):
        video_files = []
        video_files_dict = {}

        for f in os.listdir(data_dir):
            name = data_dir + f[:-9] + ".mp4"

            if name not in video_files_dict:
                video_files_dict[name] = True
                video_files.append(name)

        for i in range(len(video_files)):

            if i % 2 == 0:
                video_files[i] = video_files[i][:-4] + ".mov"

        return video_files

    def __len__(self):
        return len(self.video_files_og)

    def __getitem__(self, idx):
        return self.video_files_og[idx]

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        f = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[1::2, 1::2, :]
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)

            first_row = frame[:, :1, :]
            last_row = frame[:, -1:, :]

            if frame.size(1) > 360:
                diff = frame.size(1) - 360
                diff = int(diff/2)
                a = frame.size(1)
                frame = frame[:, diff:(frame.size(1)-diff), :]

            if frame.size(2) > 360:
                diff = frame.size(2) - 360
                diff = int(diff/2)
                frame = frame[:, :, diff:(frame.size(2)-diff)]

            while frame.size(1) < 360:
                frame = torch.cat((first_row, frame, last_row), dim=1)

            f = frame
            frames.append(frame)

            if len(frames) == 250:
                break

        while len(frames) < 250 and f is not None:
            frames.append(f)

        frames = frames[::2]

        cap.release()
        return torch.stack(frames).permute(1, 0, 2, 3)


class TestVideoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.video_files_og = [f for f in os.listdir("../data_SSL/videos/") if f.endswith('.mp4')]
        if smaller_set:
            self.video_files_og = self.get_og_videos_not_all("../data_SSL/augmented_videos_t/")
        else:
            self.video_files_og = self.get_og_videos_not_all("../data_SSL/augmented_videos/")

        if init_load:
            self.augmented_videos = {}

            print("loading videos")
            print(len(self.video_files_aug))



            for path in self.video_files_aug:
                self.augmented_videos[path] = self.load_video(path)
            print("loading videos finished")

    def get_og_videos_not_all(self, data_dir):
        video_files = []
        video_files_dict = {}

        for f in os.listdir(data_dir):
            name = data_dir + f[:-9] + ".mp4"

            if name not in video_files_dict:
                video_files_dict[name] = True
                video_files.append(name)

        for i in range(len(video_files)):

            if i % 2 == 1:
                video_files[i] = video_files[i][:-4] + ".mov"

        return video_files

    def __len__(self):
        return len(self.video_files_og)

    def __getitem__(self, idx):
        return self.video_files_og[idx]

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        f = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[1::2, 1::2, :]
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)

            first_row = frame[:, :1, :]
            last_row = frame[:, -1:, :]

            if frame.size(1) > 360:
                diff = frame.size(1) - 360
                diff = int(diff/2)
                a = frame.size(1)
                frame = frame[:, diff:(frame.size(1)-diff), :]

            if frame.size(2) > 360:
                diff = frame.size(2) - 360
                diff = int(diff/2)
                frame = frame[:, :, diff:(frame.size(2)-diff)]


            while frame.size(1) < 360:
                frame = torch.cat((first_row, frame, last_row), dim=1)

            f = frame
            frames.append(frame)


            if len(frames) == 250:
                break

        while len(frames) < 250 and f is not None:
            frames.append(f)

        frames = frames[::2]

        cap.release()
        return torch.stack(frames).permute(1, 0, 2, 3)






class Cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
        )

        self.max_pool1 = nn.MaxPool2d(kernel_size=50, stride=10)

        self.conv2 = nn.Conv3d(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=1,
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=50, stride=10)

        self.fc1 = nn.Linear(in_features=192, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=32)

        # self.mlp = MLP([128, 256, 32], norm=None)

    def get_aug_video_file_name(self, video_file_og, aug_ix):
        return video_file_og[:-4] + "_aug" + str(aug_ix) + ".mp4"

    def get_augmentations_from_batch(self, batch):
        aug = []
        for i in range(len(batch)):
            random_aug = random.randint(0, 7)

            if init_load:
                aug.append(dataset.augmented_videos[
                           self.get_aug_video_file_name(batch[i], random_aug)
                       ]
                )
            else:
                aug.append(dataset.load_video(
                               self.get_aug_video_file_name(batch[i], random_aug)
                           ))
        return torch.stack(aug)



    def forward(self, batch, train=True):
        if train:

            aug1 = self.get_augmentations_from_batch(batch)
            aug2 = self.get_augmentations_from_batch(batch)

            x1 = self.conv1(aug1)
            x1 = self.max_pool1(x1.view(x1.size(0), -1, x1.size(3), x1.size(4)))
            x1 = x1.view(x1.size(0), 4, -1, x1.size(2), x1.size(3))
            x1 = self.conv2(x1)
            # h_points_1 = self.lin1(x1, dim=1)
            h_points_1 = self.max_pool2(x1.view(x1.size(0), -1, x1.size(3), x1.size(4)))

            x1 = self.conv1(aug2)
            x1 = self.max_pool1(x1.view(x1.size(0), -1, x1.size(3), x1.size(4)))
            x1 = x1.view(x1.size(0), 4, -1, x1.size(2), x1.size(3))
            x1 = self.conv2(x1)
            # h_points_1 = self.lin1(x1, dim=1)
            h_points_2 = self.max_pool2(x1.view(x1.size(0), -1, x1.size(3), x1.size(4)))

            h1 = torch.flatten(h_points_1, start_dim=1)
            h2 = torch.flatten(h_points_2, start_dim=1)

            h1 = self.fc1(h1)
            h1 = self.fc2(h1)

            h2 = self.fc1(h2)
            h2 = self.fc2(h2)


        else:
            x1 = self.conv1(batch)
            x2 = torch.flatten(x1)
            # h_points = self.mlp(x2)
            return x2

        # compact_h_1 = self.mlp(h_1)
        # compact_h_2 = self.mlp(h_2)
        return h1, h2, h1, h2


class Rnn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Rnn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 32)


    def get_aug_video_file_name(self, video_file_og, aug_ix):
        ext = video_file_og[-4:]
        return video_file_og[:-4] + "_aug" + str(aug_ix) + ext

    def get_augmentations_from_batch(self, batch):
        aug = []

        for i in range(len(batch)):
            random_aug = random.randint(0, 7)
            if init_load:
                aug.append(dataset.augmented_videos[
                           self.get_aug_video_file_name(batch[i], random_aug)
                       ]
                )
            else:
                aug.append(dataset.load_video(
                               self.get_aug_video_file_name(batch[i], random_aug)
                           ))

        return torch.stack(aug)

    # paths for original videos are not correct
    # change the path to the original, not augmented video
    def oopsie_doopsie_change_path(self, fake_path):
        splited_fake_path = fake_path.split("/")

        if smaller_set:
            splited_fake_path[2] = "videos_t"
        else:
            splited_fake_path[2] = "videos"

        return "/".join(splited_fake_path)


    def load_videos(self, batch):
        aug = []

        for i in range(len(batch)):
            changed_path = self.oopsie_doopsie_change_path(batch[i])
            if init_load:
                aug.append(test_dataset.augmented_videos[batch[i]])
            else:
                aug.append(test_dataset.load_video(changed_path))

        return torch.stack(aug)


    def forward(self, batch, train=True):
        h0 = torch.zeros(self.num_layers, len(batch), self.hidden_size)
        c0 = torch.zeros(self.num_layers, len(batch), self.hidden_size)

        if train:

            aug1 = self.get_augmentations_from_batch(batch)
            aug2 = self.get_augmentations_from_batch(batch)
            aug3 = self.get_augmentations_from_batch(batch)

            aug_reshaped = aug1.view(aug1.size(0), aug1.size(2), -1)
            out = self.lstm(aug_reshaped, (h0, c0))
            out = out[0][:, -1, :]
            out1 = self.fc(out)

            aug_reshaped = aug2.view(aug2.size(0), aug2.size(2), -1)
            out = self.lstm(aug_reshaped, (h0, c0))
            out = out[0][:, -1, :]
            out2 = self.fc(out)

            aug_reshaped = aug2.view(aug3.size(0), aug3.size(2), -1)
            out = self.lstm(aug_reshaped, (h0, c0))
            out = out[0][:, -1, :]
            out3 = self.fc(out)

        else:
            b_reshaped = self.load_videos(batch)
            b_reshaped = b_reshaped.view(b_reshaped.size(0), b_reshaped.size(2), -1)
            out = self.lstm(b_reshaped, (h0, c0))
            out = out[0][:, -1, :]
            out = self.fc(out)

            return out

        return out1, out2, out3


# save representation of a video (aka vec) to a dictionary
# so that when testing test files this dictionary can be used
# as an anchor point for representation of test files
def store_coords(vec, video_data_paths, train_coords):

    i = 0
    for video_data_path in video_data_paths:
        video_name = video_data_path.split("/")[-1]

        # remove '.mp4'
        video_name = video_name[:-4]

        train_coords[video_name] = vec[i]
        i += 1
    return train_coords


def cosine_similarity(vec1, vec2):

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)

def is_correctly_classified(train_coords, output, video_name):

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

    legit_dist = cosine_similarity(train_coords[video_name], output)
    dists.sort()
    legit_ix = -1

    for i in range(len(dists)):
        if dists[i] == legit_dist:
            legit_ix = i

    if max_pred == video_name:
        return min_dist, True, max_dist, legit_ix
    return min_dist, False, max_dist, legit_ix


def get_errors(data, out):
    batch_error = 0
    correctly_classified = []
    legit_ixs = []
    for i in range(len(data)):
        video_path = data[i]
        video_name = video_path.split("/")[-1]
        video_name = video_name[:-4]
        batch_error += cosine_similarity(train_coords[video_name], out[i])
        dist, icc, max_dist, legit_ix = is_correctly_classified(train_coords, out[i], video_name)
        legit_ixs.append(legit_ix)
        print("dist: " + str(dist) + ", correctly_classified: " + str(icc) +
              ", correct_distance: " + str(cosine_similarity(train_coords[video_name], out[i])) + ", max_dist: " + str(max_dist)
              + ", legit_ix: " + str(legit_ix))
        correctly_classified.append(icc)

    ai = sum(legit_ixs)/len(legit_ixs)

    # print("----------------------------------------")
    # print(correctly_classified)
    # print("----------------------------------------")
    return batch_error, ai


loss_func = NTXentLoss(temperature=0.20)

init_load = False
smaller_set = False

device = torch.device("cpu")
model = Rnn(360*360*3, 80, 1)
batch_size=16
# model = Cnn()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
dataset = VideoDataset("")
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_dataset = TestVideoDataset("")
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

torch.set_num_threads(8)
print("num of threads: " + str(torch.get_num_threads()))


def train():
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(data_loader)):
        optimizer.zero_grad()

        # Get data representations
        out1, out2, out3 = model(data)
        # Prepare for loss
        embeddings = torch.cat((out1, out2, out3))
        # The same index corresponds to a positive pair
        indices = torch.arange(0, out1.size(0))
        labels = torch.cat((indices, indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * len(dataset.video_files_og)
        optimizer.step()
    return total_loss / len(dataset)

for epoch in range(1, 20):
    train()
    scheduler.step()

    if epoch % 4 == 0:
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

            avg_ix_final = sum(avg_ixs)/len(avg_ixs)
            print(f'Error : {error}')
            print(f'Avg ix : {avg_ix_final}')
