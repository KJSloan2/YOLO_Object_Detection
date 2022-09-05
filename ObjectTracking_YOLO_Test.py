import cv2
import argparse
import numpy as np
import time

imExport = False
class Args:
	def __init__(self,a0,a1,a2):
		self.config = a0
		self.weights = a1
		self.classes = a2
		
dirPath_args = r"PATH TO DIRECTORY OF SUPPORT FILES/"
dirPath_exportFrames = r"PATH TO DIRECTORY FOR FRAME EXPORTS"
args = Args(
	"%s%s" % (dirPath_args,"yolov3.cfg"),
	"%s%s" % (dirPath_args,"yolov3.weights"),
	"%s%s" % (dirPath_args,"yolov3.txt")
)

def main():
	frameCount = 0
	vcap = cv2.VideoCapture(0)
	back_sub_history = 200
	back_sub = cv2.createBackgroundSubtractorMOG2(history=back_sub_history, varThreshold=10, detectShadows=True)
	def get_output_layers(net):
		layer_names = net.getLayerNames()
		#Returns indexes of layers with unconnected outputs.
		output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
		return output_layers

	def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
		label = str(classes[class_id])
		print(label)
		color = COLORS[class_id]
		cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
		cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	while(True):
		ret, frame = vcap.read()
		frameCount +=1
		if frameCount > back_sub_history:
			fg_mask = back_sub.apply(frame)
			frame = cv2.medianBlur(fg_mask, 5)
		Width  = vcap.get(3)
		Height = vcap.get(4)
		scale = 0.00392
		classes = None

		with open(args.classes, 'r') as f:
			classes = [line.strip() for line in f.readlines()]

		COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

		net = cv2.dnn.readNet(args.weights, args.config)

		blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(get_output_layers(net))

		class_ids = []
		confidences = []
		boxes = []
		conf_threshold = 0.5
		nms_threshold = 0.4

		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					center_x = int(detection[0] * Width)
					center_y = int(detection[1] * Height)
					w = int(detection[2] * Width)
					h = int(detection[3] * Height)
					x = center_x - w / 2
					y = center_y - h / 2
					class_ids.append(class_id)
					confidences.append(float(confidence))
					boxes.append([x, y, w, h])

		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
		for i in indices:
			#i = i[0]
			box = boxes[i]
			x = box[0]
			y = box[1]
			w = box[2]
			h = box[3]

			draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
		cv2.imshow("object detection", frame)
		if imExport == True:
			cv2.imwrite("%s%s%s%s" % (dirPath_exportFrames,"object-detection_",frameCount,".jpg"), frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		# Go to the top of the while loop
		continue
	vcap.release()
	cv2.destroyAllWindows()
	cv2.waitKey()
if __name__ == '__main__':
	print(__doc__)
	main()
