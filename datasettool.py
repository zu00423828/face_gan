import os
import random
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self,root,transform=None):
        self.path=root#os.path.join(root,subfolder)
        self.image_list=os.listdir(self.path)
        self.transform=transform
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, item):
        image_path=os.path.join(self.path,self.image_list[item])
        image=cv2.imread(image_path,flags=cv2.IMREAD_COLOR)
        # image=image[:,:,::-1].copy()
        half_w=image.shape[1]//2
        x=image[:,:half_w,:]
        y=image[:,half_w:,:]
        if self.transform is not None:
            x=self.transform(x)
            y=self.transform(y)
            # x=image[:,:,:half_w]
            # y=image[:,:,half_w:]
        return x,y
    
def loadData(root,subfolder,batch_size,shuffle=True):
    transform = transforms.Compose([
    transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1)
    ])
    dataset=MyDataset(root,subfolder,transform=transform)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)




class BaseDataSet(Dataset):
    def __init__(self,dir_root,resize=None,transform=None):
        self.dir_root=dir_root
        self.landmark_dir=os.path.join(dir_root,"landmarks")
        self.reference_dir=os.path.join(dir_root,"original")
        self.target_dir=os.path.join(dir_root,"target")
        self.transform=transform
        self.landmark_list=[os.path.join(self.landmark_dir,filename) for filename in os.listdir(self.landmark_dir)]
        self.reference_list=[os.path.join(self.reference_dir,filename) for filename in os.listdir(self.reference_dir)]
        self.target_list=[os.path.join(self.target_dir,filename) for filename in os.listdir(self.target_dir)]
        self.combinelist=self.rand_combine()
        self.resize=resize
    def __len__(self):
        return len(self.combinelist)
    def __getitem__(self, index):
        landmark_img=cv2.imread(self.combinelist[index][0],cv2.IMREAD_GRAYSCALE)
        reference_img=cv2.imread(self.combinelist[index][1])
        target_img=cv2.imread(self.combinelist[index][2])
        if self.resize is not None:
            landmark_img=cv2.resize(landmark_img,(self.resize,self.resize))
            reference_img=cv2.resize(reference_img,(self.resize,self.resize))
            target_img=cv2.resize(target_img,(self.resize,self.resize))
        if self.transform is not None:
            landmark_img=self.transform(landmark_img)
            reference_img=self.transform(reference_img)
            target_img=self.transform(target_img)
        return landmark_img,reference_img,target_img
    def rand_combine(self):
        combine=[]
        for idx,(landmark_img,target_img) in enumerate(zip(self.landmark_list,self.target_list)):
            for i in range(50):#產生隨機數量 x1 x2 y組合
                randidx=random.randint(0,len(self.landmark_list)-1)
            # for idx2,reference_img in enumerate(self.reference_list):
                while idx==randidx:
                    randidx=random.randint(0,len(self.landmark_list)-1)
                combine.append([landmark_img,self.reference_list[randidx],target_img])
        return combine
if __name__=='__main__':
    transform=transforms.Compose([transforms.ToTensor()])
    dataset=BaseDataSet("raw",256,transform)
    # dataset=BaseDataSet("photos")
    print(len(dataset))
    dataloader=DataLoader(dataset,batch_size=2,drop_last=True)
    print(len(dataloader))
    for i,(x1,x2,y) in enumerate(dataloader):
        print(x1.shape)
        print(x2.shape)
        print(y.shape)
        ynp=y[0].numpy().transpose(1,2,0)
        cv2.imwrite("test.png",ynp)
        break

    # for i in range(10):
    #     transform=transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip()])
    #     dataset=MyDataset("combined/train",transform=transform)
    #     train_data=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=4)
    #     t2i=dataset[0][0].numpy().transpose(1,2,0)
    #     cv2.imshow(f"{i}",t2i)
    #     cv2.waitKey(0)
    # print(len(dataset))
    # print(len(dataset))
    # print(dataset[0][0].shape)
    # t2i=(dataset[0][0].numpy().transpose(1,2,0)) #or t.permute(1,2,0).numpy() c,h,w to h,w,c 
    # t2i=(t2i*255).astype(np.uint8)
    # cv2.imshow("test",t2i)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(len(train_data))
    # for i,x in enumerate(train_data):
    #     print(type(x),len(x))