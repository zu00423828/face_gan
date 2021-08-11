import os
import random
import glob
import shutil
import argparse
def main():
    landmarks=glob.glob(f"{root}/landmarks/*/")
    # random.seed(0)
    label_dirs=["landmarks","img","face_mask"]
    # label_dirs=["background","background_mask","face","face_mask","img","landmarks"]
    # count=0
    train_list=open(f"{output_dir}/train_list.csv","w+")
    val_list=open(f"{output_dir}/val_list.csv","w+")
    for item in landmarks:
        # count+=1
        # print(f"{count}/{len(landmarks)}")
        sub_dir="/".join(item.split("/")[-2:])
        filelist=os.listdir(item)
        filelist.sort()
        if len(filelist)<40:
            continue
        assignments=[]
        assignments.extend(["train"]*int(train_frac*len(filelist)))
        assignments.extend(["val"]*int((1-train_frac)*len(filelist)))
        random.shuffle(assignments)
        for file,assignment in zip(filelist,assignments):
            input_path=",".join([os.path.join(label_dir,sub_dir,file) for label_dir in label_dirs])
            if assignment=="train":
                train_list.write(input_path+"\n")
            else: 
                val_list.write(input_path+"\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_dir",dest="input_dir",type=str)
    parser.add_argument("--output_dir",dest="output_dir",type=str)
    parser.add_argument("--train_frac",dest="train_frac",default=0.75,type=float)
    args=parser.parse_args()
    root=args.input_dir
    output_dir=args.output_dir
    train_frac=args.train_frac
    main()
