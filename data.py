import os
import zipfile
import numpy as np

from scipy import misc
from skimage import color
from urllib.request import urlretrieve


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class TrainSet:
    def __init__(self, benchmark, batch_size=64, patch_size=41, scaling_factors=(2, 3, 4)):
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.scaling_factors = scaling_factors
        self.images_completed = 0
        self.epochs_completed = 0
        self.root_path = os.path.join(DATA_PATH, 'train', benchmark)
        self.images = []
        self.targets = []

        if not os.path.exists(self.root_path):
            download()

        for file_name in os.listdir(self.root_path):
            image = misc.imread(os.path.join(self.root_path, file_name))

            if len(image.shape) == 3:
                image = color.rgb2ycbcr(image)[:, :, 0].astype(np.uint8)

            width, height = image.shape
            width = width - width % 12
            height = height - height % 12
            n_horizontal_patches = width // patch_size
            n_vertical_patches = height // patch_size
            image = image[:width, :height]

            for scaling_factor in scaling_factors:
                downscaled = misc.imresize(image, 1 / scaling_factor, 'bicubic', mode='L')
                rescaled = misc.imresize(downscaled, float(scaling_factor), 'bicubic', mode='L')
                high_res_image = image.astype(np.float32) / 255
                low_res_image = np.clip(rescaled.astype(np.float32) / 255, 0.0, 1.0)

                for horizontal_patch in range(n_horizontal_patches):
                    for vertical_patch in range(n_vertical_patches):
                        h_start = horizontal_patch * patch_size
                        v_start = vertical_patch * patch_size
                        high_res_patch = high_res_image[h_start:h_start + patch_size, v_start:v_start + patch_size]
                        low_res_patch = low_res_image[h_start:h_start + patch_size, v_start:v_start + patch_size]

                        for _ in range(4):
                            high_res_patch = np.rot90(high_res_patch)
                            low_res_patch = np.rot90(low_res_patch)

                            self.targets.append(np.expand_dims(high_res_patch, axis=2))
                            self.images.append(np.expand_dims(low_res_patch, axis=2))

                        high_res_patch = np.fliplr(high_res_patch)
                        low_res_patch = np.fliplr(low_res_patch)

                        for _ in range(4):
                            high_res_patch = np.rot90(high_res_patch)
                            low_res_patch = np.rot90(low_res_patch)

                            self.targets.append(np.expand_dims(high_res_patch, axis=2))
                            self.images.append(np.expand_dims(low_res_patch, axis=2))

        self.images = np.array(self.images)
        self.targets = np.array(self.targets)

        self.shuffle()
        self.length = len(self.images)
        self.length = self.length - self.length % batch_size
        self.images = self.images[:self.length]
        self.targets = self.targets[:self.length]

    def batch(self):
        images = self.images[self.images_completed:(self.images_completed + self.batch_size)]
        targets = self.targets[self.images_completed:(self.images_completed + self.batch_size)]

        self.images_completed += self.batch_size

        if self.images_completed >= self.length:
            self.images_completed = 0
            self.epochs_completed += 1
            self.shuffle()

        return images, targets

    def shuffle(self):
        indices = list(range(len(self.images)))
        np.random.shuffle(indices)

        self.images = self.images[indices]
        self.targets = self.targets[indices]


class TestSet:
    def __init__(self, benchmark, scaling_factors=(2, 3, 4)):
        self.benchmark = benchmark
        self.scaling_factors = scaling_factors
        self.images_completed = 0
        self.root_path = os.path.join(DATA_PATH, 'test', self.benchmark)
        self.file_names = os.listdir(self.root_path)
        self.images = []
        self.targets = []

        if not os.path.exists(self.root_path):
            download()

        for file_name in os.listdir(self.root_path):
            image = misc.imread(os.path.join(self.root_path, file_name))

            width, height = image.shape[0], image.shape[1]
            width = width - width % 12
            height = height - height % 12
            image = image[:width, :height]

            if len(image.shape) == 3:
                ycbcr = color.rgb2ycbcr(image)
                y = ycbcr[:, :, 0].astype(np.uint8)
            else:
                y = image

            for scaling_factor in self.scaling_factors:
                downscaled = misc.imresize(y, 1 / scaling_factor, 'bicubic', mode='L')
                rescaled = misc.imresize(downscaled, float(scaling_factor), 'bicubic', mode='L')

                if len(image.shape) == 3:
                    low_res_image = ycbcr
                    low_res_image[:, :, 0] = rescaled
                    low_res_image = color.ycbcr2rgb(low_res_image)
                    low_res_image = (np.clip(low_res_image, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    low_res_image = rescaled

                self.images.append(low_res_image)
                self.targets.append(image)

        self.length = len(self.images)

    def fetch(self):
        if self.images_completed >= self.length:
            return None
        else:
            self.images_completed += 1

            return self.images[self.images_completed - 1], self.targets[self.images_completed - 1]


def download():
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    for partition in ['train', 'test']:
        partition_path = os.path.join(DATA_PATH, partition)
        zip_path = os.path.join(partition_path, '%s_data.zip' % partition)
        url = 'http://cv.snu.ac.kr/research/VDSR/%s_data.zip' % partition

        if not os.path.exists(partition_path):
            os.mkdir(partition_path)

        if not os.path.exists(zip_path):
            urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path) as f:
            f.extractall(partition_path)
