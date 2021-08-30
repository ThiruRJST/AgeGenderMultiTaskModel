# import the opencv library
import cv2
import numpy as np
import albumentations
from pl_model import AgeGendNet
import time
from albumentations.pytorch import ToTensorV2
# define a video capture object
vid = cv2.VideoCapture(-1,cv2.CAP_V4L)
model_file = "/home/lustbeast/office/dnn detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "/home/lustbeast/office/dnn detector/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file,model_file)
model = AgeGendNet()
model = model.load_from_checkpoint("/home/lustbeast/office/resnet18_256_Model_Fold-1-2/Resnet50_256_Checkpoint-ValLoss:val_loss=0.2497.ckpt")
model.eval()
model = model.to("cuda:0")
def val_augs():
	return albumentations.Compose([
		albumentations.Resize(256,256),
		albumentations.Normalize(),
		ToTensorV2()
	])

while(True):
	
	start_time = time.time()
	# Capture the video frame
	# by frame
	ret, frame = vid.read()
	h,w = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,117.0,123.0))
	net.setInput(blob)
	faces = net.forward()
	for i in range(0,faces.shape[2]):
		confidence = faces[0,0,i,2]
		#print(confidence)
		if confidence>0.5:
			box = faces[0,0,i,3:7] * np.array([w,h,w,h])
			(x,y,x1,y1) = box.astype("int")
			im = frame[y:y1,x:x1]
			im = val_augs()(image=im)['image']
			im = im.unsqueeze(0)
			im = im.to("cuda:0")
			out = model(im)
			if out[0].item()<0.5:
				#print("Male")
				cv2.putText(frame,f"M - {out[1].item()*116}",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.45,(0,0,255),2)
			else:
				#print("Female")
				cv2.putText(frame,f"F - {out[1].item()*116}",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.45,(0,0,255),2)
			#print(out[1]*116)
			cv2.rectangle(frame,(x,y),(x1,y1),(0,0,255),2)
	print(f"FPS:{1.0/ (time.time() - start_time)}")
	cv2.imshow("frame",frame)
	
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
