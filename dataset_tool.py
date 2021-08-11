import csv
import os
from random import randint
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
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

# augment=transforms.Compose([transforms.ToTensor(),transforms.Resize(256)])#transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
# augment=transforms.Compose([transforms.ToTensor(),transforms.Resize(256)])
augment=transforms.Compose([transforms.ToTensor()])
class TestDataSet(Dataset):
    def __init__(self, root,split,resize,landmark_gray=False):
        self.filelist,self.video_name= self.get_file_list(root,split)
        self.resize=(resize,resize)
        self.landmark_gray=landmark_gray
    def __len__(self):
        return len(self.video_name)
    def __getitem__(self, index):
        rand_idx=self.random_get(index,self.video_name)
        # landmark_img=cv2.imread(self.landmark_list[index],cv2.IMREAD_GRAYSCALE)
        if not self.landmark_gray:
            landmark_img=cv2.resize(cv2.imread(self.filelist[index][0]),self.resize,interpolation=cv2.INTER_AREA)
        else:
            landmark_img=cv2.resize(cv2.imread(self.filelist[index][0],cv2.IMREAD_GRAYSCALE),self.resize,interpolation=cv2.INTER_AREA)
        reference_img=cv2.resize(cv2.imread(self.filelist[rand_idx][1]),self.resize,interpolation=cv2.INTER_AREA)
        face_mask_img=cv2.resize(cv2.imread(self.filelist[index][2],cv2.IMREAD_GRAYSCALE),self.resize,interpolation=cv2.INTER_AREA)
        target_img=cv2.resize(cv2.imread(self.filelist[index][1]),self.resize,interpolation=cv2.INTER_AREA)
        landmark_img= augment(landmark_img)
        reference_img=augment(reference_img)
        target_img=augment(target_img)
        face_mask_img=augment(face_mask_img)
        # other_reference=self.random_feture(index,self.video_name)
        # other_reference=augment(other_reference)
        
        input_img=torch.cat([landmark_img,reference_img],dim=0)
        return input_img,target_img,face_mask_img#,other_reference
        # return landmark_img,reference_img,target_img,face_mask_img

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
            
    def get_file_list(self,root,split):
        filelist=[]
        videoname=[]
        with open(f"{root}/{split}_list.csv",newline='')as csvfile:
            rows = csv.reader(csvfile)
            path_join=os.path.join
            filelist_append=filelist.append
            videoname_append=videoname.append
            for row in rows:
                filelist_append([path_join(root,file)for file in row])
                videoname_append(row[0].split("/")[-2])
        return filelist,videoname
    def random_get(self,idx,video_name):
        min_idx=video_name.index(video_name[idx])
        max_idx=len(video_name)-1-video_name[::-1].index(video_name[idx])
        r_idx=randint(min_idx,max_idx)
        r_idx=randint(min_idx,max_idx)
        while idx==r_idx:
            r_idx=randint(min_idx,max_idx)
        return  r_idx
    def random_feture(self,idx,video_name):
        min_idx=video_name.index(video_name[idx])
        max_idx=len(video_name)-1-video_name[::-1].index(video_name[idx])
        r_idx=randint(min_idx,max_idx)
        featurelist=[]
        while len(featurelist)<8:
            if idx!=r_idx and r_idx not in featurelist:
                    featurelist.append(r_idx)
            r_idx=randint(min_idx,max_idx)
        feature_img=None
        for i in featurelist:
            img=cv2.imread(self.filelist[i][1])
            if feature_img is None:
                feature_img=img
            else:
                feature_img=np.concatenate((feature_img,img),axis=2)
        feature_img=cv2.resize(feature_img,self.resize)
        return  feature_img



class BgMixerDataset(TestDataSet):
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
    # a = TestDataSet("/home/yuan/hdd/avspeech_preprocess/train")
    train_dataset=TestDataSet("/home/yuan/hdd/avspeech_preprocess/preprocess4","train",resize=256)
    print(train_dataset[0][-1].shape)
 