from openvino.runtime import Core
core = Core()
import cv2

# Load models
person_detection = "models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
face_detection = "models/face-detection-retail-0005/FP32/face-detection-retail-0005.xml"
person_reidentification = "models/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml"
age_gender = "models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml"
pose_detection = "models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
face_reidentification = "models/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml"

def compiled_model(model_name):
    model = core.read_model(model=(model_name))
    compiled_model = core.compile_model(model=model, device_name="CPU")
    return compiled_model

person_detection = compiled_model(person_detection)
face_detection = compiled_model(face_detection)
person_reidentification = compiled_model(person_reidentification)
age_gender = compiled_model(age_gender)
pose_detection = compiled_model(pose_detection)
face_reidentification = compiled_model(face_reidentification)



def image_processing(model, image):
    input_shape = model.input().shape
    image_resized = cv2.resize(image, (input_shape[3], input_shape[2]))
    image_resized = image_resized.transpose((2, 0, 1))
    image_reshaped = image_resized.reshape(1, 3, input_shape[2], input_shape[3])
    return image_reshaped

def run_model(model,image):
    image_reshaped = image_processing(model,image)
    
    try:
        results = model([image_reshaped])[model.output()]
    except:
        results = model([image_reshaped])
    return results
