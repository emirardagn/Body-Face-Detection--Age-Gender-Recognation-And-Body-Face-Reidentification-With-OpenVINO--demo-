import cv2
from openvino.runtime import Core
import numpy as np
from scipy.spatial.distance import cosine
import time
core = Core()

# Load person detection and reidentification models
person_detection_model_xml = "models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
face_detection_model_xml = "models/face-detection-retail-0005/FP32/face-detection-retail-0005.xml"
reidentification_model_xml = "models/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml"
age_gender_model_xml = "models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml"
pose_detection_model_xml = "models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"


#Person Detection
person_detection_model = core.read_model(model=person_detection_model_xml)
compiled_person_detection_model = core.compile_model(model=person_detection_model, device_name="CPU")
input_layer_detection = compiled_person_detection_model.input()
output_layer_detection = compiled_person_detection_model.output()

#ReID
reidentification_model = core.read_model(model=reidentification_model_xml)
compiled_reidentification_model = core.compile_model(model=reidentification_model, device_name="CPU")
input_layer_reidentification = compiled_reidentification_model.input()
output_layer_reidentification = compiled_reidentification_model.output()

#Face Detection
face_detection_model = core.read_model(model=face_detection_model_xml)
compiled_face_detection_model = core.compile_model(model=face_detection_model, device_name="CPU")
input_layer_face_detection = compiled_face_detection_model.input()
output_layer_face_detection = compiled_face_detection_model.output()

# Pose Est
pose_detection_model = core.read_model(model=pose_detection_model_xml)
compiled_pose_detection_model = core.compile_model(model=pose_detection_model, device_name="CPU")
input_layer_pose_detection = compiled_pose_detection_model.input()
#output_layer_pose_detection = compiled_pose_detection_model.output()

#Age Gender
age_gender_model = core.read_model(model=age_gender_model_xml)
compiled_age_gender_model = core.compile_model(model=age_gender_model, device_name="CPU")
input_layer_age_gender = compiled_age_gender_model.input()



cap = cv2.VideoCapture(0)
cap = cv2.resize(cap,700,700)
tracked_persons = {}
total_people=0
tracked_persons_duration = {}

next_person_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    initial_h, initial_w = frame.shape[:2]
    input_shape_detection = input_layer_detection.shape

    # Preprocess the frame for person detection
    image_resized = cv2.resize(frame, (input_shape_detection[3], input_shape_detection[2]))
    image_transposed = image_resized.transpose((2, 0, 1))
    image_reshaped = image_transposed.reshape(1, 3, input_shape_detection[2], input_shape_detection[3])

    # Perform person detection inference
    detection_results = compiled_person_detection_model([image_reshaped])[output_layer_detection]
    detected_objects = detection_results[0][0]
    
    for result in detected_objects:
        
        if result[2] > 0.95:  # Confidence threshold
            frame_entered_time = time.time()
            xmin = int(result[3] * initial_w)
            ymin = int(result[4] * initial_h)
            xmax = int(result[5] * initial_w)
            ymax = int(result[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            if(xmin <0 or ymin <0 or xmin <0 or xmax <0 ):
                break


            # Person Reidentification
            person_crop = frame[ymin:ymax, xmin:xmax]
            input_shape_reidentification = input_layer_reidentification.shape
            person_crop_resized = cv2.resize(person_crop, (input_shape_reidentification[3], input_shape_reidentification[2]))
            person_crop_resized = person_crop_resized.transpose((2, 0, 1))
            person_crop_reshaped = person_crop_resized.reshape(1, 3, input_shape_reidentification[2], input_shape_reidentification[3])
            reidentification_results = compiled_reidentification_model([person_crop_reshaped])[output_layer_reidentification]
            

            # Face Detection
            face_crop = frame[ymin:ymax,xmin:xmax]
            input_shape_face_detection = input_layer_face_detection.shape
            face_crop_resized = cv2.resize(face_crop, (input_shape_face_detection[3], input_shape_face_detection[2]))
            face_crop_resized = face_crop_resized.transpose((2, 0, 1))
            face_crop_reshaped = face_crop_resized.reshape(1, 3, input_shape_face_detection[2], input_shape_face_detection[3])
            face_results = compiled_face_detection_model([face_crop_reshaped])[output_layer_face_detection]
            face_detections = face_results[0][0]
            for detection in face_detections:
                confidence = detection[2]
                if confidence > 0.95:  # Confidence threshold
                    face_xmin = int(detection[3] * face_crop.shape[1])
                    face_ymin = int(detection[4] * face_crop.shape[0])
                    face_xmax = int(detection[5] * face_crop.shape[1])
                    face_ymax = int(detection[6] * face_crop.shape[0])
                    cv2.rectangle(face_crop, (face_xmin, face_ymin), (face_xmax, face_ymax), (255, 0, 0), 2)


            # Head Pose Est
            pose_crop = frame[ymin:ymax,xmin:xmax]
            input_shape_pose_detection = input_layer_pose_detection.shape
            pose_crop_resized = cv2.resize(pose_crop, (input_shape_pose_detection[3], input_shape_pose_detection[2]))
            pose_crop_resized = pose_crop_resized.transpose((2, 0, 1))
            pose_crop_reshaped = pose_crop_resized.reshape(1, 3, input_shape_pose_detection[2], input_shape_pose_detection[3])
            pose_results = compiled_pose_detection_model([pose_crop_reshaped])


            yaw = pose_results['angle_y_fc'][0][0] #Yan Kayma)
            pitch = pose_results['angle_p_fc'][0][0]  #(Baş Eğimi)
            roll = pose_results['angle_r_fc'][0][0] # (Dönüş)
            print(yaw, pitch, roll)


            # Gender Part
            gender_crop = frame[ymin:ymax, xmin:xmax]
            input_shape_gender = input_layer_age_gender.shape
            age_gender_resize = cv2.resize(frame,(input_shape_gender[3], input_shape_gender[2]))
            age_gender_resize = age_gender_resize.transpose((2, 0, 1))
            age_gender_reshaped = age_gender_resize.reshape(1, 3, input_shape_gender[3], input_shape_gender[2])
            age_gender_result = compiled_age_gender_model([age_gender_reshaped])
            gender_prob = age_gender_result['prob'][0][1]  # Gender probability (0 - female, 1 - male)
            gender = 'Male' if gender_prob > 0.5 else 'Female'
            age = (int(age_gender_result["age_conv3"][0][0][0][0]*100))
            # Initialize variables for ID assignment
            found_match = False
            min_distance = 1.0  # Initialize with a value larger than possible cosine similarity
            match_id = None
            # Compare with tracked persons
            for person_id, features in tracked_persons.items():
                distance = cosine(features, reidentification_results.flatten())
                if distance < min_distance:
                    min_distance = distance
                    match_id = person_id
            
            #CHECK WHETHER ID MATCH OR NOT

            # If a match is found within a threshold, assign the match ID
            if min_distance < 0.5:  # You may adjust this threshold based on your application
                time_passed = time.time() - frame_entered_time
                tracked_persons_duration.setdefault(match_id, 0)
                tracked_persons_duration[match_id] += time_passed*10

                cv2.putText(frame, f'id {match_id}, {tracked_persons_duration[match_id]:.2f}, {gender}, {age}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                tracked_persons[match_id] = reidentification_results.flatten()  # Update features
            else:
                # Assign a new ID for a new person
                total_people+=1
                tracked_persons[next_person_id] = reidentification_results.flatten()
                time_passed = time.time() - frame_entered_time
                tracked_persons_duration.setdefault(match_id, 0)
                tracked_persons_duration[match_id] += time_passed*10
                cv2.putText(frame, f'id {next_person_id}, {tracked_persons_duration[match_id]:.2f}, {gender}, {age}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                next_person_id += 1
    
    
    cv2.putText(frame,f"Total number: {total_people}",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    # Display the frame with detections and IDs
    cv2.imshow('Person Reidentification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
