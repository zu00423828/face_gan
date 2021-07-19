import os
import cv2
import numpy as np
import dlib




detector=dlib.get_frontal_face_detector()
# predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor=dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")





# origin_img_list=os.listdir("photos/original")
# for filename in origin_img_list:
#     filepath=os.path.join("photos/original",filename)
#     img=cv2.imread(filepath)
#     # cv2.imshow("img",img)
#     # cv2.waitKey(0)
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces=detector(gray,0)
#     for face in faces:
#         detected_landmarks=predictor(gray,face).parts()
#         landmarks=[[p.x,p.y]for p in detected_landmarks]
#         face_contour=landmarks[0:17]
#         eyebrow_pos=landmarks[26:16:-1]
#         forehead=[]
#         for i in [78,74,79,73,72,80,71,70,69,68,76,75,77]:
#             forehead.append(landmarks[i])
#         for item in eyebrow_pos:
#             face_contour.append(item)
#         ih,iw,ic=img.shape
#         mask_nobg=np.zeros((ih,iw),np.uint8)
#         cv2.fillPoly(mask_nobg,[np.array(face_contour)],(255,255,255),8,0)
#         mask_inv=cv2.bitwise_not(mask_nobg)
#         # cv2.imshow("frame",mask_inv)
#         # cv2.imwrite("bg_mask.png",mask_inv)
#         only_bg=cv2.bitwise_and(img,img,mask=mask_inv)
#         only_face=cv2.bitwise_and(img,img,mask=mask_nobg)
#         cv2.imwrite(f"photos/target_bg/{filename}",only_bg)
#         cv2.imwrite(f"photos/target_face/{filename}",only_face)
def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))

rawroot="raw/frame"
imglist=os.listdir(rawroot)
for idx,filename in enumerate(imglist):
    filepath=os.path.join(rawroot,filename)
    img=cv2.imread(filepath)
    dest=detector(img,1)
    for i,d in enumerate(dest):
        if i>0:
            break
        detected_landmarks=predictor(img,d).parts()
        landmarks=[[p.x,p.y]for p in detected_landmarks]
        landmark_reshape=reshape_for_polyline(landmarks)
        face_contour=landmarks[0:17]
        forehead=[]
        for i in [78,74,79,73,72,69,68,76,75,77]:#[78,74,79,73,72,80,71,70,69,68,76,75,77]:
            forehead.append(landmarks[i])
        for item in forehead:
            face_contour.append(item)
        ih,iw,ic=img.shape
        mask_nobg=np.zeros((ih,iw),np.uint8)
        landmarks_img=np.zeros((ih,iw),np.uint8)
        cv2.fillPoly(mask_nobg,[np.array(face_contour)],(255,255,255),8,0)
        # cv2.fillPoly(landmarks_img,landmark_reshape,(255,255,255),8,0)
        cv2.polylines(landmarks_img,landmark_reshape,True,(255,255,255),3)
        mask_bg=cv2.bitwise_not(mask_nobg)
        only_face=cv2.bitwise_and(img,img,mask=mask_nobg)
        only_bg=cv2.bitwise_and(img,img,mask=mask_bg)
        centroid_pos=landmarks[28]
        left_x=centroid_pos[0]-192
        right_x=centroid_pos[0]+192
        top_y=centroid_pos[1]-192
        botton_y=centroid_pos[1]+192
        crop_img=img[top_y:botton_y,left_x:right_x]
        crop_facemask=mask_nobg[top_y:botton_y,left_x:right_x]
        crop_bgmask=mask_bg[top_y:botton_y,left_x:right_x]
        crop_face=only_face[top_y:botton_y,left_x:right_x]
        crop_bg=only_bg[top_y:botton_y,left_x:right_x]
        crop_landmark=landmarks_img[top_y:botton_y,left_x:right_x]
        cv2.imwrite(f"raw/crop_img/{idx}.png",crop_img)
        cv2.imwrite(f"raw/face_mask/{idx}.png",crop_facemask)
        cv2.imwrite(f"raw/bg_mask/{idx}.png",crop_bgmask)
        cv2.imwrite(f"raw/only_face/{idx}.png",crop_face)
        cv2.imwrite(f"raw/landmarks/{idx}.png",crop_landmark)
        cv2.imwrite(f"raw/only_bg/{idx}.png",crop_bg)

        # cv2.imwrite(f"raw/face_mask/{idx}.png",mask_nobg)
        # cv2.imwrite(f"raw/only_face/{idx}.png",only_face)
        # cv2.imshow("img",mask_nobg)
        # if cv2.waitKey(1)& 0xFF==ord('q'): 
        #     break