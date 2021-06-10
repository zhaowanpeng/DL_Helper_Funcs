import os,torch,math
from torch.utils.data import Dataset
from torchvision import transforms
from funcs.img_read import read_img,filter_cannot_read
from funcs.img_transform import *

class MyDataset(Dataset):

    def __init__(self, true_paths=None,false_paths=None, color=True,img_mode="3s_64",filter=False,random=False):

        self.color=color
        self.img_mode=img_mode
        self.true_paths=true_paths
        self.false_paths=false_paths
        self.random=random
        self.imgs=[]
        self.init_imgs_list()
        if filter:
            self.imgs=filter_cannot_read(self.imgs)


    def __getitem__(self, index):
        imgpath=self.imgs[index]
        img = read_img(imgpath,color=self.color)

        if self.random:
            random_dict = {
                "RT": (1, random_rotate_func),
                "MR": (1, random_mirror),
                "RS": (5, random_resize),
                "CP": (5, random_crop),
                "OG": (0, get_original)
            }
            random_transformer = RandomFunc(random_dict)
            random_trans_func = random_transformer.get_random_func()
            img=random_trans_func(img)

        if self.img_mode=="3s_64":
            img = self.read_3s_img(img,len=64)

        else:
            img=None
            print("sorry!author not do perfect")
            exit()
        img_splts = imgpath.split("/")
        filepath="/".join(img_splts[:-1])+"/"
        if filepath in self.true_paths:
            label = 1
        else:
            label = 0
        label = torch.tensor([label], dtype=torch.float)
        return img,label#,imgpath

    def __len__(self):
        return len(self.imgs)

    def init_imgs_list(self):
        paths = self.true_paths + self.false_paths
        path_num = len(paths)
        for i in range(path_num):
            path=paths[i]
            print("[{}/{}]initing imgs path ...     [path:{}]".format(i + 1, path_num, path))
            imgs = os.listdir(path)
            prcs_imgs=[path+img for img in imgs]
            self.imgs+=prcs_imgs
        print("* * * init completed * * *")
        print("* * * imgs total num : {} * * * ".format(len(self.imgs)))


    def read_3s_img(self,img,len=64):

        (w,h)=img.size
        (min_len,min_name) = (h,"h") if h < w else (w,"w")
        if min_name=="h":
            sclr=len/h
            rs_h=len
            rs_w=math.ceil(w*sclr)
        else:
            sclr=len/w
            rs_w=len
            rs_h=math.ceil(h*sclr)
        tran_5 = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((rs_h, rs_w)),
            transforms.ToTensor(),
            transforms.FiveCrop((len, len)),
        ])
        tranimgs=tran_5(img)
        if min_name=="h":
            img_mix = torch.torch.cat([tranimgs[0], tranimgs[1], tranimgs[4]], dim=0)
        else:
            img_mix = torch.torch.cat([tranimgs[0], tranimgs[2], tranimgs[4]], dim=0)

        return img_mix