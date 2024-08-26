import augly.video as av
import os
import torchaudio
import ffmpeg
import librosa
import vidgear

video_data_path = "../data_SSL/videos/"
aug_video_data_path = "../data_SSL/augmented_videos/"



def create_video_augmentations(filename):
    aug0 = av.AddNoise()
    aug0(video_data_path + filename, aug_video_data_path + filename[:-4] + "_aug0.mov")

    aug1 = av.RandomBrightness()
    aug1(video_data_path + filename, aug_video_data_path + filename[:-4] + "_aug1.mov")

    aug2 = av.Rotate(10)
    aug2(video_data_path + filename, aug_video_data_path + filename[:-4] + "_aug2.mov")

    aug3 = av.Compose(
        [
            av.OverlayDots(),
            av.ColorJitter(-0.4, 40, 0.6)
        ]
    )
    aug3(video_data_path + filename, aug_video_data_path + filename[:-4] + "_aug3.mov")

    aug4 = av.Blur(0.4)
    aug4(video_data_path + filename, aug_video_data_path + filename[:-4] + "_aug4.mov")

    aug5 = av.Compose(
        [
            av.Grayscale(),
            av.RandomEmojiOverlay(opacity=0.4, emoji_size=0.05)
        ]
    )
    aug5(video_data_path + filename, aug_video_data_path + filename[:-4] + "_aug5.mov")

    aug6 = av.Compose(
        [
            av.RandomBrightness(max_level=0.7),
            av.RandomBlur(max_sigma=5, p=0.5)

        ]
    )
    aug6(video_data_path + filename, aug_video_data_path + filename[:-4] + "_aug6.mov")


    aug7 = av.Compose(
        [
            av.OverlayDots(),
            av.Rotate(355),
            av.RandomVideoSpeed(min_factor=0.8, max_factor=1.2)
        ]
    )
    aug7(video_data_path + filename, aug_video_data_path + filename[:-4] + "_aug7.mov")


def augment_data():
    for filename in os.listdir(video_data_path):

        full_file_name = video_data_path + filename

        print("augmenting data")

        # only augment train data (.mp4 files)
        if os.path.isfile(full_file_name) and full_file_name.endswith(".mov"):
            create_video_augmentations(filename)


augment_data()


