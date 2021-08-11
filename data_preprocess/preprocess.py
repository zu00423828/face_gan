import os
import cv2
import numpy as np
import dlib
import argparse

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")




def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def face_split(frame):
        ih,iw,ic=frame.shape
        dest,score,idx=detector.run(frame,1)
        if not  score:
            return
        if score[0]<args.confidence:
            return
        for i,face in enumerate(dest):
            if i>0:
                break
            detected_landmarks=predictor(frame,face).parts()
            landmarks=[[p.x,p.y] for p in detected_landmarks]
            landmarks_array=np.array(landmarks)
            min_x=min(landmarks_array[:,0])
            min_y=min(landmarks_array[:,1])
            max_x=max(landmarks_array[:,0])
            max_y=max(landmarks_array[:,1])
            h,w=max_y-min_y,max_x-min_x
            # print("origin",h,w)
            if h<100:
                return
            if h!=w:
                min_x=min_x-((h-w)//2)-20
                max_x=max_x+((h-w)//2)+20
                min_y-=20
                max_y+=20
            # print(max_y-min_y,max_x-min_x)
            if min_y<=0 or max_y>=ih:
                #print("y> or y<")
                return
            if min_x<=0 or max_x>=iw :
                #print("x> or x<")
                return
            # for i in[78,74,79,73,72,80,71,70,69,68,76,75,77]:# [78,74,79,73,72,69,68,76,75,77]:
            face_contour_array=np.concatenate((landmarks_array[0:17],landmarks_array[[78,74,79,73,72,80,71,70,69,68,76,75,77]]))
            face_mask=np.zeros((ih,iw),np.uint8)
            landmark_img=np.zeros(frame.shape,np.uint8)
            cv2.fillPoly(face_mask,[face_contour_array],(255,255,255),8,0)
            # jaw = landmarks_array[0:17]
            left_eyebrow = landmarks_array[22:27]
            right_eyebrow =landmarks_array[17:22]
            nose_bridge = landmarks_array[27:31]
            lower_nose = landmarks_array[30:36]
            left_eye = landmarks_array[42:48]
            right_eye = landmarks_array[36:42]
            outer_lip = landmarks_array[48:60]
            inner_lip = landmarks_array[60:68]
            # cv2.fillPoly(face_mask, [left_eyebrow],(0,0,0), 16,0)
            # cv2.fillPoly(face_mask, [right_eyebrow], (0,0,0), 16,0)
            cv2.polylines(face_mask, [left_eyebrow], False,(0,0,0), 2)
            cv2.polylines(face_mask, [right_eyebrow], False, (0,0,0), 2)
            # cv2.fillPoly(face_mask, [nose_bridge], (0,0,0), 16,0)
            cv2.line(face_mask,nose_bridge[0],nose_bridge[-1],(0,0,0),2,16)
            cv2.polylines(face_mask, [nose_bridge], True, (0,0,0), 3)
            cv2.fillPoly(face_mask, [lower_nose], (0,0,0), 16,0)
            cv2.fillPoly(face_mask, [left_eye], (0,0,0), 16,0)
            cv2.fillPoly(face_mask, [right_eye], (0,0,0), 16,0)
            cv2.fillPoly(face_mask, [outer_lip],(0,0,0), 16,0)
            cv2.fillPoly(face_mask, [inner_lip],(255,255,255), 16,0)
            # cv2.polylines(landmark_img, [jaw], False, (255,255,255), 3)
            cv2.polylines(landmark_img, [face_contour_array], True, (255,255,255), 3)
            #---landmark color---
            if  args.color:
                cv2.polylines(landmark_img, [left_eyebrow], False,(0,255,0), 3)
                cv2.polylines(landmark_img, [right_eyebrow], False, (0,255,0), 3)
                cv2.polylines(landmark_img, [nose_bridge], False, (255,0,0), 3)
                cv2.polylines(landmark_img, [lower_nose], True, (255,0,0), 3)
                cv2.polylines(landmark_img, [left_eye], True, (0,0,255), 3)
                cv2.polylines(landmark_img, [right_eye], True, (0,0,255), 3)
                cv2.polylines(landmark_img, [outer_lip], True,(255,255,0), 3)
                cv2.polylines(landmark_img, [inner_lip], True,(0,255,255), 3)
            #---landmark gray---
            else:
                cv2.polylines(landmark_img, [left_eyebrow], False,(255,255,255), 3)
                cv2.polylines(landmark_img, [right_eyebrow], False, (255,255,255), 3)
                cv2.polylines(landmark_img, [nose_bridge], False, (255,255,255), 3)
                cv2.polylines(landmark_img, [lower_nose], True, (255,255,255), 3)
                cv2.polylines(landmark_img, [left_eye], True, (255,255,255), 3)
                cv2.polylines(landmark_img, [right_eye], True, (255,255,255), 3)
                cv2.polylines(landmark_img, [outer_lip], True,(255,255,255), 3)
                cv2.polylines(landmark_img, [inner_lip], True,(255,255,255), 3)
                # cv2.fillPoly(landmark_img,reshape_for_polyline(landmarks),(255,255,255),8,0)
                # cv2.polylines(landmark_img,reshape_for_polyline(landmarks),True,(255,255,255),3)
            #------------------------

            # bg_mask=cv2.bitwise_not(face_mask)
            # only_face=cv2.bitwise_and(frame,frame,mask=face_mask)
            # only_bg=cv2.bitwise_and(frame,frame,mask=bg_mask)
            # centroid_pos=landmarks[28]
            crop_img=frame[min_y:max_y,min_x:max_x]
            crop_landmark=landmark_img[min_y:max_y,min_x:max_x]
            crop_face_mask=face_mask[min_y:max_y,min_x:max_x]
            # crop_face=only_face[min_y:max_y,min_x:max_x]
            # crop_bg=only_bg[min_y:max_y,min_x:max_x]
            # crop_bg_mask=bg_mask[min_y:max_y,min_x:max_x]
            if crop_img.shape[0]!=256 or crop_img.shape[1]!=256:
                crop_img=cv2.resize(crop_img,(256,256))
                crop_landmark=cv2.resize(crop_landmark,(256,256))
                crop_face_mask=cv2.resize(crop_face_mask,(256,256))
                # crop_face=cv2.resize(crop_face,(256,256))
                # crop_bg=cv2.resize(crop_bg,(256,256))
                # crop_bg_mask=cv2.resize(crop_bg_mask,(256,256))
            # cv2.imwrite(f"{preprocess_original}/{count}.png",frame)
            cv2.imwrite(f"{preprocess_img}/{count}.png",crop_img)
            cv2.imwrite(f"{preprocess_landmark}/{count}.png",crop_landmark)
            cv2.imwrite(f"{preprocess_face_mask}/{count}.png",crop_face_mask)
            # cv2.imwrite(f"{preprocess_face}/{count}.png",crop_face)
            # cv2.imwrite(f"{preprocess_background}/{count}.png",crop_bg)
            # cv2.imwrite(f"{preprocess_background_mask}/{count}.png",crop_bg_mask)
# print(vildeolist[343])
# def main():
# for idx,videofilename in enumerate(vildeolist[a:a+1]):
# for idx,videofilename in enumerate(vildeolist[1000:1200]):


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_dir",dest="input_dir",required=True,type=str)
    ap.add_argument("--output_dir",dest="output_dir",required=True,type=str)
    ap.add_argument("--t",dest="threshold",default=20.,type=float)
    ap.add_argument("--c",dest="confidence",default=0.5,type=float)
    ap.add_argument("--color",dest="color",default=1,type=int)
    args=ap.parse_args()
    # vildeolist=os.listdir("avspeech")
    # output_root="preprocess4"
    vildeolist=os.listdir(args.inputdir)
    output_root=args.output_dir
    for idx,videofilename in enumerate(vildeolist[0:200]):
        videopath=os.path.join("avspeech",videofilename)
        video=cv2.VideoCapture(videopath)
        preprocess_landmark=os.path.join(f"{output_root}/landmarks",videofilename.rsplit(".")[0])
        preprocess_face_mask=os.path.join(f"{output_root}/face_mask",videofilename.rsplit(".")[0])
        # preprocess_face=os.path.join(f"{output_root}/face",videofilename.rsplit(".")[0])
        # preprocess_background=os.path.join(f"{output_root}/background",videofilename.rsplit(".")[0])
        # preprocess_background_mask=os.path.join(f"{output_root}/background_mask",videofilename.rsplit(".")[0])
        preprocess_img=os.path.join(f"{output_root}/img",videofilename.rsplit(".")[0])
        os.makedirs(preprocess_landmark,exist_ok=True)
        os.makedirs(preprocess_face_mask,exist_ok=True)
        # os.makedirs(preprocess_face,exist_ok=True)
        # os.makedirs(preprocess_background,exist_ok=True)
        # os.makedirs(preprocess_background_mask,exist_ok=True)
        os.makedirs(preprocess_img,exist_ok=True)
        # print(preprocess_dir)

        count=0
        while video.isOpened():
            ret,frame=video.read()
            if not ret:
                break
            if cv2.Laplacian(frame, cv2.CV_64F).var()>args.threshold:
                face_split(frame)
            count+=1
            # if cv2.waitKey(1) & 0xff==ord("q"):
            #     break
        print(f"pregress:{idx+1}/{len(vildeolist)}")
        cv2.destroyAllWindows()
        video.release()