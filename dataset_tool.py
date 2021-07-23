import enum
import os
import random
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
import dlib
class BaseDataSet(Dataset):
    def __init__(self,dir_root,resize=None,transform=None):
        self.dir_root=dir_root
        self.landmark_dir=os.path.join(dir_root,"landmarks")
        self.reference_dir=os.path.join(dir_root,"crop_img")
        self.target_dir=os.path.join(dir_root,"crop_img")
        self.facemask_dir=os.path.join(dir_root,"face_mask")
        self.transform=transform
        self.landmark_list=[os.path.join(self.landmark_dir,filename) for filename in os.listdir(self.landmark_dir)]
        self.reference_list=[os.path.join(self.reference_dir,filename) for filename in os.listdir(self.reference_dir)]
        self.target_list=[os.path.join(self.target_dir,filename) for filename in os.listdir(self.target_dir)]
        self.facemask_list=[os.path.join(self.facemask_dir,filename) for filename in os.listdir(self.facemask_dir)]
        self.combinelist=self.rand_combine()
        self.resize=resize
    def __len__(self):
        return len(self.combinelist)
    def __getitem__(self, index):
        landmark_img=cv2.imread(self.combinelist[index][0],cv2.IMREAD_GRAYSCALE)
        reference_img=cv2.imread(self.combinelist[index][1])
        target_img=cv2.imread(self.combinelist[index][2])
        facemask_img=cv2.imread(self.combinelist[index][3],cv2.IMREAD_GRAYSCALE)
        # reference_img=reference_img[:,:,::-1].copy()
        # target_img=target_img[:,:,::-1].copy()
        if self.resize is not None:
            landmark_img=cv2.resize(landmark_img,(self.resize,self.resize))
            reference_img=cv2.resize(reference_img,(self.resize,self.resize))
            target_img=cv2.resize(target_img,(self.resize,self.resize))
            facemask_img=cv2.resize(facemask_img,(self.resize,self.resize))
        if self.transform is not None:
            landmark_img=self.transform(landmark_img)
            reference_img=self.transform(reference_img)
            target_img=self.transform(target_img)
            target_map=self.transform(facemask_img)
        return landmark_img,reference_img,target_img,target_map
    def rand_combine(self):
        combine=[]
        for idx,(landmark_img,target_img,facemask_img) in enumerate(zip(self.landmark_list,self.target_list,self.facemask_list)):
            for i in range(50):#產生隨機數量 x1 x2 y組合
                randidx=random.randint(0,len(self.landmark_list)-1)
            # for idx2,reference_img in enumerate(self.reference_list):
                while idx==randidx:
                    randidx=random.randint(0,len(self.landmark_list)-1)
                combine.append([landmark_img,self.reference_list[randidx],target_img,facemask_img])
        return combine

augment=transforms.Compose([transforms.ToTensor(),transforms.Resize(256)])

class TestDataSet(Dataset):
    def __init__(self, root):
        img_dir = glob.glob(f"{root}/img/*/")
        landmark_dir = glob.glob(f"{root}/landmarks/*/")
        face_mask_dir = glob.glob(f"{root}/face_mask/*/")
        self.img_list, self.landmark_list, self.face_mask_list = self.get_file_list(img_dir,landmark_dir,face_mask_dir)
        self.video_name=[item.split("/")[-2] for item in self.img_list]

    def __len__(self):
        return len(self.video_name)
    def __getitem__(self, index):
        rand_idx=self.random_get(index)
        landmark_img=cv2.imread(self.landmark_list[index],cv2.IMREAD_GRAYSCALE)
        reference_img=cv2.imread(self.img_list[rand_idx])
        face_mask_img=cv2.imread(self.face_mask_list[index],cv2.IMREAD_GRAYSCALE)
        target_img=cv2.imread(self.img_list[index])
        landmark_img=augment(landmark_img)
        reference_img=augment(reference_img)
        face_mask_img=augment(face_mask_img)
        target_img=augment(target_img)
        return landmark_img,reference_img,target_img,face_mask_img

    def landmark(img):
        pass
        detector=dlib.get_frontal_face_detector()
        predictor=dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
        dest=detector(img,1)
        for i,d in enumerate(dest):
            if i>0:
                break
            detected_landmark=predictor(img,d).parts()
            landmarks=[[p.x,p.y] for p in detected_landmark]
            face_contour=landmarks[0:17]
            forehead=[]
            for i in [78,74,79,73,72,69,68,76,75,77]:#[78,74,79,73,72,80,71,70,69,68,76,75,77]:
                forehead.append(landmarks[i])
            for item in forehead:
                face_contour.append(item)
            ih,iw,ic=img.shape
            landmarks_img=np.zeros((ih,iw).np.uint8)
            face_mask=np.zeros((ih,iw).np.uint8)
            cv2.fillPoly(face_mask,(np.array(face_contour)),(255,255,255),8,0)
            cv2.fillPoly(landmarks_img,np.array(landmarks).reshape(-1,1,2),(255,255,255),8,0)
            
    def get_file_list(self,img_dir,landmark_dir,face_mask_dir):
        all_img_list = []
        all_landmark_list = []
        all_face_mask_list = []
        for dir in img_dir:
            temp_list = [os.path.join(dir, file) for file in os.listdir(dir)]
            all_img_list.extend(temp_list)
        for dir in landmark_dir:
            temp_list = [os.path.join(dir, file) for file in os.listdir(dir)]
            all_landmark_list.extend(temp_list)
        for dir in face_mask_dir:
            temp_list = [os.path.join(dir, file) for file in os.listdir(dir)]
            all_face_mask_list.extend(temp_list)
        return all_img_list, all_landmark_list, all_face_mask_list
    def random_get(self,idx):
        min_idx=self.video_name.index(self.video_name[idx])
        max_idx=len(self.video_name)-1-self.video_name[::-1].index(self.video_name[idx])
        r_idx=random.randint(min_idx,max_idx)
        while idx==r_idx:
            r_idx=random.randint(min_idx,max_idx)
        return  r_idx



class BgMixerDataset(BaseDataSet):
    def __init__(self, dir_root, resize, transform):
        super().__init__(dir_root, resize=resize, transform=transform)
        self.root=dir_root
        self.x1_dir=os.path.join(self.root,"only_bg")
        self.x2_dir=os.path.join(self.root,"only_face")
        self.y_dir=os.path.join(self.root,"crop_img")
        x1_list=[os.path.join(self.x1_dir,filename) for filename in os.listdir(self.x1_dir)]
        x2_list=[os.path.join(self.x2_dir,filename) for filename in os.listdir(self.x2_dir)]
        y_list=[os.path.join(self.y_dir,filename) for filename in os.listdir(self.y_dir)]
        self.combinelist=[[x1,x2,y] for x1,x2,y in zip(x1_list,x2_list,y_list)]
        # print(x1_list)
        print(len(x1_list),len(x2_list),len(y_list))
        self.resize=resize
        self.transform=transform
    
if __name__=='__main__':
    # transform=transforms.Compose([transforms.ToTensor()])
    # dataset=BgMixerDataset("raw",256,transform)
    # print(len(dataset))
    # a = TestDataSet("/home/yuan/hdd/avspeech_preprocess/test/train")
    # print(len(a))
    a = TestDataSet("/home/yuan/hdd/avspeech_preprocess/train")



 