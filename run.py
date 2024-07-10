import cv2
import numpy as np
from scipy.spatial.distance import cosine
import time
import models
from datetime import datetime

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("video.mp4")


xmin, xmax, ymin, ymax = [None] * 4

face_person_id = 0
person_id = 0

frame_count = 0

data = {}

faces = {}
faces_matrix = {}

persons = {}
persons_matrix = {}


def set_boundaries(detection,frame):
    initial_h, initial_w = frame.shape[:2]
    xmin = int(detection[3] * initial_w)
    ymin = int(detection[4] * initial_h)
    xmax = int(detection[5] * initial_w)
    ymax = int(detection[6] * initial_h)
    return xmin, ymin, xmax, ymax

def check_IOU(box1,box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

   
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight


    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea)
    return (iou > 0.05)

def get_age_gender(frame):
    age_gender_result = models.run_model(models.age_gender,frame)
    gender_prob = age_gender_result['prob'][0][1]
    gender = 'Male' if gender_prob > 0.5 else 'Female'
    age = (int(age_gender_result["fc3_a"][0][0][0][0]*100))
    return age,gender

def get_pose(frame):
    pose_results = models.run_model(models.pose_detection, frame)
    yaw = pose_results['fc_y'][0][0]
    roll = pose_results["fc_p"][0][0]
    if (-15<yaw<15 and 15>roll>-15 ):
        pose = "looking"
    else:
        pose = "not looking"   
    return pose 

def run_face(frame):
    global face_person_id
    face_result = models.run_model(models.face_detection,frame)
    face_detections = face_result[0][0]
    for detection in face_detections:
        confidence = detection[2]
        #if sure
        if confidence > 0.9:
            frame_entered_time = time.time()
            xmin,ymin,xmax,ymax = set_boundaries(detection, frame)
            if(xmin <0 or ymin <0 or xmin <0 or xmax <0 ):
                break
            
            face_crop = frame[ymin:ymax, xmin:xmax]
            age,gender = get_age_gender(face_crop)
            pose = get_pose(face_crop)
  
            face_min_distance = 1.0
            face_match_id = None

            # Face Reidentification
            face_reidentification_results = models.run_model(models.face_reidentification,face_crop)

            for matrix in faces_matrix:
                distance = cosine(faces_matrix[matrix], face_reidentification_results.flatten())
                if distance < face_min_distance:
                    face_min_distance = distance
                    face_match_id = matrix
            
            #if exist
            if face_min_distance < 0.8:
                cv2.putText(frame, f'id {face_match_id}',(xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                if(pose=="looking"):
                    faces[face_match_id]["looking_time"] += (time.time() - frame_entered_time)*10
                faces[face_match_id]["age"] = age
                faces[face_match_id]["gender"] = gender
                faces[face_match_id]["pose"] = pose
                faces[face_match_id]["boundaries"] = (xmin,ymin,xmax,ymax)

            #if not exist
            else:
                cv2.putText(frame, f'id {face_person_id}',(xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                faces[face_person_id] = {
                    "face_id": face_person_id,
                    "u_id": ("u"+str(face_person_id)),
                    "entered_time": (datetime.now().hour,datetime.now().minute, datetime.now().second),
                    "age": age,
                    "gender": gender,
                    "pose": pose,
                    "boundaries" : (xmin,ymin,xmax,ymax)

                }
                if (pose == "looking"):
                    faces[face_person_id]["looking_time"] = (time.time() - frame_entered_time)*10
                else:
                    faces[face_person_id]["looking_time"] = 0
                    
                faces_matrix[face_person_id] = face_reidentification_results.flatten()
                face_person_id += 1
            if (pose =="looking"):  
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

def run_body(frame):
    global person_id
    person_result = models.run_model(models.person_detection,frame)
    person_detections = person_result[0][0]
    for detection in person_detections:
        confidence = detection[2]
        if confidence > 0.9:  # Confidence threshold
            xmin,ymin,xmax,ymax = set_boundaries(detection,frame)
            if(xmin <0 or ymin <0 or xmin <0 or xmax <0 ):
                break
            
            
            person_crop = frame[ymin:ymax, xmin:xmax]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            min_distance = 1.0
            match_id = None
            person_reidentification_results = models.run_model(models.person_reidentification,person_crop)
            
            
            for matrix in persons_matrix:
                distance = cosine(persons_matrix[matrix], person_reidentification_results.flatten())
                if distance < min_distance:
                    min_distance = distance
                    match_id = matrix
            
            #if exist
            if min_distance < 0.8:
                cv2.putText(frame, f'id {match_id}',(xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                persons_matrix[match_id] = person_reidentification_results.flatten()
                persons[match_id]["boundaries"] = (xmin,ymin,xmax,ymax)

            #if not exist
            else:
                cv2.putText(frame, f'id {person_id}',(xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                persons[person_id] = {
                    "body_id": person_id,
                    "boundaries": (xmin,ymin,xmax,ymax)
                }

                persons_matrix[person_id] = person_reidentification_results.flatten()

                person_id += 1
            

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count+=1
    
    run_face(frame)
    run_body(frame)
    if(frame_count%100 == 0):
        for person in persons:
            for face in faces:
                if(check_IOU(persons[person]["boundaries"],faces[face]["boundaries"])):
                    persons[person]["u_id"]= faces[face]["u_id"]
        print(persons)
        
    
    cv2.putText(frame, f'face {face_person_id}',(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
            
    cv2.imshow('Person Reidentification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
