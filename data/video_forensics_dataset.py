import os
import pickle

from PIL import Image
import torch
from torchvision import transforms

from data.base_dataset import BaseDataset


class VideoForensicsDataset(BaseDataset):
    def __init__(self, opt):
        super(VideoForensicsDataset, self).__init__(opt)

        with open(os.path.join(opt.dataroot, f'VideoForensicsHQ_Images/{opt.phase}_dict.pkl'), 'rb') as f:
            self.id_dict = pickle.load(f)
        self.id_list = list(self.id_dict.keys())

        if opt.crop_more:
            self.transform = transforms.Compose([
                transforms.Resize((700, 700), Image.BICUBIC),
                transforms.CenterCrop(opt.load_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((opt.load_size, opt.load_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        id_name = self.id_list[index]
        img_list = self.id_dict[id_name]

        interval = torch.randint(1, self.opt.max_interval, [1]).item()
        idx1 = torch.randint(0, len(img_list) - interval, [1]).item()
        img1 = Image.open(os.path.join(self.root, id_name, img_list[idx1]))
        img2 = Image.open(os.path.join(self.root, id_name, img_list[idx1 + interval]))

        # 두 이미지에 같은 random crop 적용
        if self.opt.isTrain and torch.rand(1).item() < self.opt.crop_prob:
            h = img1.size[0]  # 모든 이미지는 이미 정사각형
            if self.opt.crop_separate:
                crop_scale1 = (self.opt.crop_scale + torch.rand(1) * (1 - self.opt.crop_scale)).item()
                crop_scale2 = (self.opt.crop_scale + torch.rand(1) * (1 - self.opt.crop_scale)).item()
                crop_size1 = int(h * crop_scale1)
                crop_size2 = int(h * crop_scale2)

                x1 = torch.randint(0, h - crop_size1 + 1, [1]).item()
                y1 = torch.randint(0, h - crop_size1 + 1, [1]).item()
                x2 = torch.randint(0, h - crop_size2 + 1, [1]).item()
                y2 = torch.randint(0, h - crop_size2 + 1, [1]).item()
                img1 = img1.crop((x1, y1, x1 + crop_size1, y1 + crop_size1))
                img2 = img2.crop((x2, y2, x2 + crop_size2, y2 + crop_size2))
            else:
                crop_scale = (self.opt.crop_scale + torch.rand(1) * (1 - self.opt.crop_scale)).item()
                crop_size = int(h * crop_scale)
                shake_scale_x = (torch.rand(1) * self.opt.shake_scale).item()
                shake_scale_y = (torch.rand(1) * self.opt.shake_scale).item()
                shake_size_x = min(int(crop_size * shake_scale_x), h - crop_size)
                shake_size_y = min(int(crop_size * shake_scale_y), h - crop_size)

                x = torch.randint(0, h - crop_size - shake_size_x + 1, [1]).item()  # random crop 시작 좌표 구하기
                y = torch.randint(0, h - crop_size - shake_size_y + 1, [1]).item()  # random crop 시작 좌표 구하기
                coords = [[x, x + shake_size_x, y, y + shake_size_y],
                          [x, x + shake_size_x, y + shake_size_y, y],
                          [x + shake_size_x, x, y, y + shake_size_y],
                          [x + shake_size_x, x, y + shake_size_y, y]]
                case = int(torch.rand(1).item() * 4)
                img1 = img1.crop((coords[case][0], coords[case][2], coords[case][0] + crop_size, coords[case][2] + crop_size))
                img2 = img2.crop((coords[case][1], coords[case][3], coords[case][1] + crop_size, coords[case][3] + crop_size))

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return {'img1': img1, 'img2': img2, 'img_paths': f'{id_name.split("/")[-1]}_{idx1:05d}_{interval}'}

    def __len__(self):
        return len(self.id_dict)  # train: 757
