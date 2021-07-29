import os
import random
import glob
import shutil
import argparse
def main():
    landmarks=glob.glob(f"{root}/landmarks/*/")
    random.seed(0)
    label_dirs=["background","background_mask","face","face_mask","img","landmarks"]
    count=0
    for item in landmarks:
        count+=1
        print(f"{count}/{len(landmarks)}")
        sub_dir="/".join(item.split("/")[-2:])
        filelist=os.listdir(item)
        filelist.sort()
        # print(len(filelist))
        if len(filelist)<40:
            continue
        assignments=[]
        assignments.extend(["train"]*int(train_frac*len(filelist)))
        assignments.extend(["val"]*int((1-train_frac)*len(filelist)))
        # assignments.extend(["val"]*int(0.25*len(filelist)))
        random.shuffle(assignments)
        for  assignment in ["train","val"]:
            for label_dir in label_dirs:
                os.makedirs(os.path.join(output_dir,assignment,label_dir,sub_dir),exist_ok=True)
        for file,assignment in zip(filelist,assignments):
            for label_dir in label_dirs:
                input_path=os.path.join(root,label_dir,sub_dir,file)
                output_path=os.path.join(output_dir,assignment,label_dir,sub_dir,file)
                shutil.copyfile(input_path,output_path)
            # print(input_path,output_path)


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
