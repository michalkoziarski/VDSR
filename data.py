import os
import zipfile
import numpy as np

from scipy import misc
from urllib.request import urlretrieve


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class TrainSet:
    def __init__(self, benchmark, batch_size, scaling_factors=(2, 3, 4)):
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.scaling_factors = scaling_factors
        self.images_completed = 0
        self.epochs_completed = 0
        self.root_path = os.path.join(DATA_PATH, 'train', '%s_augmented' % self.benchmark)
        self.paths = os.listdir(self.root_path)
        self.images = []
        self.targets = []

        for file_name in self.paths:
            full_name, extension = file_name.split('.')
            original_name, count, scaling_factor = full_name.split('-')

            if int(scaling_factor) in self.scaling_factors:
                target_name = '%s-%s-1.%s' % (original_name, count, extension)

                self.images.append(self.__read_image(file_name))
                self.targets.append(self.__read_image(target_name))

        self.images = np.array(self.images)
        self.targets = np.array(self.targets)

        self.shuffle()
        self.length = len(self.images)
        self.length = self.length - self.length % self.batch_size
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

    def __read_image(self, file_name):
        return np.expand_dims((misc.imread(os.path.join(self.root_path, file_name)).astype(np.float) / 255), axis=2)


class TestSet:
    def __init__(self, benchmark, scaling_factors=(2, 3, 4)):
        self.benchmark = benchmark
        self.scaling_factors = scaling_factors
        self.images_completed = 0
        self.root_path = os.path.join(DATA_PATH, 'test', '%s_augmented' % self.benchmark)
        self.paths = os.listdir(self.root_path)
        self.images = []
        self.targets = []

        for file_name in self.paths:
            full_name, extension = file_name.split('.')
            original_name, count, scaling_factor = full_name.split('-')

            if int(scaling_factor) in self.scaling_factors:
                target_name = '%s-%s-1.%s' % (original_name, count, extension)

                self.images.append(self.__read_image(file_name))
                self.targets.append(self.__read_image(target_name))

        self.length = len(self.images)

    def fetch(self):
        if self.images_completed >= self.length:
            return None
        else:
            self.images_completed += 1

            return np.array([self.images[self.images_completed - 1]]), np.array([self.targets[self.images_completed - 1]])

    def __read_image(self, file_name):
        return np.expand_dims((misc.imread(os.path.join(self.root_path, file_name)).astype(np.float) / 255), axis=2)


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
            print('Downloading %s data...' % partition)

            urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path) as f:
            print('Extracting %s data...' % partition)

            f.extractall(partition_path)


def augment(partition, benchmark, patch_size=41):
    benchmark_path = os.path.join(DATA_PATH, partition, benchmark)
    augmented_path = os.path.join(DATA_PATH, partition, '%s_augmented' % benchmark)

    if not os.path.exists(augmented_path):
        os.mkdir(augmented_path)

    if partition == 'train':
        for file_name in os.listdir(benchmark_path):
            image = misc.imread(os.path.join(benchmark_path, file_name), mode='YCbCr')[:, :, 0]
            width, height = image.shape
            width = width - width % 12
            height = height - height % 12
            n_horizontal_patches = width // patch_size
            n_vertical_patches = height // patch_size
            image = image[:width, :height]
            scaled_images = {1.0: image}

            for scale in [2.0, 3.0, 4.0]:
                downscaled = misc.imresize(image, 1.0 / scale, 'bicubic')
                scaled_images[scale] = misc.imresize(downscaled, scale, 'bicubic')

            for scale in [1.0, 2.0, 3.0, 4.0]:
                count = 0

                for horizontal_patch in range(n_horizontal_patches):
                    for vertical_patch in range(n_vertical_patches):
                        h_start = horizontal_patch * patch_size
                        v_start = vertical_patch * patch_size
                        patch = scaled_images[scale][h_start:h_start + patch_size, v_start:v_start + patch_size]

                        for flip in [True, False]:
                            if flip:
                                flipped = np.fliplr(patch)
                            else:
                                flipped = patch

                            for angle in [0, 90, 180, 270]:
                                rotated = misc.imrotate(flipped, angle, 'bicubic')

                                name, extension = file_name.split('.')
                                count += 1
                                out_name = '%s-%d-%d.%s' % (name, count, scale, extension)
                                misc.imsave(os.path.join(augmented_path, out_name), rotated)

    if partition == 'test':
        for file_name in os.listdir(benchmark_path):
            image = misc.imread(os.path.join(benchmark_path, file_name), mode='YCbCr')[:, :, 0]
            width, height = image.shape
            width = width - width % 12
            height = height - height % 12
            image = image[:width, :height]
            scaled_images = {1.0: image}

            for scale in [2.0, 3.0, 4.0]:
                downscaled = misc.imresize(image, 1.0 / scale, 'bicubic')
                scaled_images[scale] = misc.imresize(downscaled, scale, 'bicubic')

            for scale in [1.0, 2.0, 3.0, 4.0]:
                name, extension = file_name.split('.')
                out_name = '%s-1-%d.%s' % (name, scale, extension)
                misc.imsave(os.path.join(augmented_path, out_name), scaled_images[scale])


def load(partition, benchmark, batch_size=64, patch_size=41, scaling_factors=(2, 3, 4), verbose=True):
    assert partition in ['train', 'test']

    if partition == 'train':
        assert benchmark in ['91', '291']
    else:
        assert benchmark in ['B100', 'Set5', 'Set14', 'Urban100']

    benchmark_path = os.path.join(DATA_PATH, partition, benchmark)
    augmented_path = os.path.join(DATA_PATH, partition, '%s_augmented' % benchmark)

    if not os.path.exists(benchmark_path):
        if verbose:
            print('Downloading data...')

        download()

        if verbose:
            print('Downloading complete.')

    if not os.path.exists(augmented_path):
        if verbose:
            print('Augmenting data...')

        augment(partition, benchmark, patch_size)

        if verbose:
            print('Augmenting complete.')

    if verbose:
        print('Loading data to the memory...')

    if partition == 'train':
        data_set = TrainSet(benchmark, batch_size, scaling_factors=scaling_factors)
    else:
        data_set = TestSet(benchmark, scaling_factors=scaling_factors)

    if verbose:
        print('Loading complete.')

    return data_set
