import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2
import math

# 0 	nose
# 1 	leftEye
# 2 	rightEye
# 3 	leftEar
# 4 	rightEar
# 5 	leftShoulder
# 6 	rightShoulder
# 7 	leftElbow
# 8 	rightElbow
# 9 	leftWrist
# 10 	rightWrist
# 11 	leftHip
# 12 	rightHip
# 13 	leftKnee
# 14 	rightKnee
# 15 	leftAnkle
# 16 	rightAnkle

def parse_output(heatmap_data,offset_data, threshold):

  '''
  Input:
    heatmap_data - hetmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
  Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
  '''

  joint_num = heatmap_data.shape[-1]
  pose_kps = np.zeros((joint_num,3), np.uint32)

  for i in range(heatmap_data.shape[-1]):

      joint_heatmap = heatmap_data[...,i]
      max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
      pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
      pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
      max_prob = np.max(joint_heatmap)

      if max_prob > threshold:
        if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
          pose_kps[i,2] = 1

  return pose_kps


def draw_keypoints(show_img,kps, ratio=None):
    for i in range(5,kps.shape[0]):
      if kps[i,2]:
        if isinstance(ratio, tuple):
          cv2.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,255,255),round(int(1*ratio[1])))
          continue
        cv2.circle(show_img,(kps[i,1],kps[i,0]),2,(0,255,255),-1)
    return show_img



def calculateAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)
    """a = list(a)
    b=list(b)
    c=list(c)
    num = a[1]*(b[0]-c[0]) + b[1]*(c[0]-a[0])+c[1]*(a[0]-b[0])
    deno=(a[0]-b[0])*(b[0]-c[0]) + (a[1]-b[1])*(b[1]-c[1])
    print(num,deno)
    angrad = np.arctan(num/deno)
    angdeg = (angrad*180)/np.pi
    if(angdeg < 0):
        angdeg+=180
    
    return angdeg"""

def join_point(img, kps):

  body_parts = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),
                      (6,12),(11,13),(12,14),(13,15),(14,16)]

  for part in body_parts:
    cv2.line(img, (kps[part[0]][1], kps[part[0]][0]), (kps[part[1]][1], kps[part[1]][0]),
            color=(255,255,255), lineType=cv2.LINE_AA, thickness=3)
#left hand right hand left foot right foot 
joints = [(5,7,9),(6,8,10),(11,13,15),(12,14,16)]
parts_to_compare = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]
body_parts = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]

model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
file_path="person2.jpeg"

# Get input and output tensors information from the model file
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()



# Get input and output tensors information from the model file
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


image=cv2.imread(file_path)
image=cv2.resize(image,(width,height))
target_input = np.expand_dims(image.copy(), axis=0)

floating_model = input_details[0]['dtype'] == np.float32
if floating_model:
  #template_input = (np.float32(template_input) - 127.5) / 127.5
  target_input = (np.float32(target_input) - 127.5) / 127.5

interpreter.set_tensor(input_details[0]['index'], target_input)
interpreter.invoke()
target_output_data = interpreter.get_tensor(output_details[0]['index'])
target_offset_data = interpreter.get_tensor(output_details[1]['index'])
target_heatmaps = np.squeeze(target_output_data)
target_offsets = np.squeeze(target_offset_data)

target_show = np.squeeze((target_input.copy()*127.5+127.5)/255.0)
target_show = np.array(target_show*255,np.uint8)
target_kps = parse_output(target_heatmaps,target_offsets,0.3)
temp = np.asarray([ target_kps[i][0:2] for i in joints[2]] , dtype="float64")
print(calculateAngle(temp[0],temp[1],temp[2]))
cv2.imwrite('2.jpg',draw_keypoints(target_show.copy(),target_kps))



"""
while True:
    vid=cv2.VideoCapture(0)
    ret,frame=vid.read()
    frame = cv2.resize(frame, (width, height))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output",frame)
    cv2.resizeWindow("output",600,600)
    #plt.imshow(frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
"""
