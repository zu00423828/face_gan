import cv2


img=cv2.imread("frame0.png")
print(type(img))
if not img:
    pass
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
# ret,thresh = cv2.threshold(gray,127,255,0)
ret,thresh = cv2.threshold(gray,160,255,0)
cv2.imshow("thresh",thresh)
M=cv2.moments(thresh)
print(M)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
print(M["m00"])
# rect=cv2.minAreaRect(thresh)
# box=cv2.boxPoints(rect)
# cv2.drawContours(thresh,[box],0,(0,0,255),2)
print(cX,cY)
# cv2.circle()

cv2.circle(img,(cX,cY),1,(0,0,255),3)
cv2.imshow("img",img)
cv2.waitKey(0)



# video=cv2.VideoCapture("input.mp4")
# count=0
# while video.isOpened():
#     ret,frame=video.read()
#     if not ret:
#         break
#     # cv2.imshow("frame",frame)
#     cv2.imwrite(f"frame/{count}.png",frame)
#     print(count)
#     if cv2.waitKey(1)& 0xFF==ord('q'):
#         break
#     count+=1
# cv2.destroyAllWindows()
# video.release()