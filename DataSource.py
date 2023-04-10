import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np

class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()
        self.images = []




        # TODO: Define preprocessing
        self.resize_tranforms = T.Resize((256,455))
        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
            print(self.mean_image.shape)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()
            # np.save(self.mean_image_path, a)

        if train:
            self.transform = T.Compose([
                T.ToTensor(),
                T.RandomCrop(crop_size),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.CenterCrop(crop_size),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        for index in range(len(self.images_path)):
            data = Image.open(self.images_path[index])
                # TODO: Perform preprocessing
            data = self.resize_tranforms(data)
            data = np.array(data, dtype=np.float)
            data -= self.mean_image
            data = data.astype(np.uint8)
            data = Image.fromarray(data)
            # print(data.shape)

            data = self.transform(data)
            self.images.append(data)

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")

        # TODO: Compute mean image
        total = 0.0
        for img_path in self.images_path:
            # Initialize mean_image
            img = Image.open(img_path)
            # Iterate over all training images
            img = self.resize_tranforms(img)
            img = np.asarray(img, dtype=np.float32)
            # Store mean image
            total += img

        mean_image = total/len(self.images_path)
        np.save(self.mean_image_path, mean_image)

        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        data = self.images[index]
        img_pose = self.image_poses[index]

        
        return data, img_pose

    def __len__(self):
        return len(self.images_path)