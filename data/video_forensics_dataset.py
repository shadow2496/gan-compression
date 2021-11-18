import os
import pickle

import numpy as np
from PIL import Image
from torchvision import transforms

from data.base_dataset import BaseDataset


class VideoForensicsDataset(BaseDataset):
    def __init__(self, opt):
        super(VideoForensicsDataset, self).__init__(opt)

        with open(os.path.join(opt.dataroot, f'VideoForensicsHQ_Images/{opt.phase}_dict.pkl'), 'rb') as f:
            self.id_dict = pickle.load(f)
        self.id_list = list(self.id_dict.keys())

        self.transform = transforms.Compose([
            transforms.Resize((opt.load_size, opt.load_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        id_name = self.id_list[index]
        img_list = self.id_dict[id_name]

        interval = np.random.randint(1, 10)
        idx1 = np.random.randint(0, len(img_list) - interval)
        img1 = Image.open(os.path.join(self.root, id_name, img_list[idx1]))
        img2 = Image.open(os.path.join(self.root, id_name, img_list[idx1 + interval]))

        # FIXME: crop 위치가 random이 아닌 것 같습니다
        # 두 이미지에 같은 random crop 적용
        if self.opt.is_train and np.random.rand(1)[0] < self.opt.crop_prob:
            h = img1.size[0]  # 모든 이미지는 이미 정사각형
            crop_scale_x = np.random.uniform(self.opt.crop_scale, 1)
            crop_scale_y = np.random.uniform(self.opt.crop_scale, 1)
            x = int(h * (1 - crop_scale_x))  # random crop 시작 좌표 구하기
            y = int(h * (1 - crop_scale_y))  # random crop 시작 좌표 구하기
            crop_size = int(h * min(crop_scale_x, crop_scale_y))
            img1 = img1.crop((x, y, x + crop_size, y + crop_size))
            img2 = img2.crop((x, y, x + crop_size, y + crop_size))

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return {'img1': img1, 'img2': img2, 'img_paths': id_name}

    def __len__(self):
        return len(self.id_dict)  # train: 757
