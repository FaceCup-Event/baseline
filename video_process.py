import cv2
import torch
from torchvision.transforms import functional as F
from facenet_pytorch import MTCNN,InceptionResnetV1
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np



print(torch.cuda.is_available())
frame_rate = 10
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()
face_ids = set()
model = load_model('./models/model.h5')
# Load pre-trained Haar Cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load pre-trained Haar Cascade classifier for detecting eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect(videoname):
    print(videoname)
    # Initialize variables
    mask_count = 0
    acculotion = 0
    glasses_count = 0
    count = 0
    total_result = []
    multi_face_result = []
    mask_prob=[]

    cap = cv2.VideoCapture(videoname)
    while True:
        ret, frame = cap.read()
        multi_face = 0
        if not ret:
            break
        count += 1
        if count % frame_rate != 0:
            continue

################################################### spoof detect ############################################
        

########################################## multiface & multi id ########################################
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces in frame
        boxes, _ = mtcnn.detect(frame_rgb)
        if boxes is not None:
            multi_face_result.append(len(boxes))

        # Loop through detected faces
        if boxes is not None:
            for box in boxes:
                # Extract face from frame
                face = F.to_tensor(frame_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
                if (face.shape[1] >= 160 & face.shape[2] >= 160):
                    face = F.resize(face, (160, 160))

                    face = torch.unsqueeze(face, dim=0).float()
                    # Normalize face
                    face = (face - 127.5) / 128.0
                    face = face.cuda()
                    # Calculate face embeddings using InceptionResnetV1 model
                    embeddings = resnet(face)

                    # Convert embeddings to numpy array
                    embeddings_np = embeddings.detach().cpu().numpy()

                    # Convert embeddings to string
                    embeddings_str = embeddings_np.tostring()

                    # Calculate face ID based on embeddings
                    face_id = hash(embeddings_str)

                    # Check if face ID is new
                    if face_id not in face_ids:
                        # Add face ID to set
                        face_ids.add(face_id)

########################################## mask detection ############################################
        # Preprocess frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)

        # Predict mask probability
        pred = model.predict(np.expand_dims(frame, axis=0))[0]
        mask_prob = pred[0]
        no_mask_prob = pred[1]

        # Check if wearing a mask
        if mask_prob > no_mask_prob:
            mask_count += 1

################################################## sunglasses  ####################################################
        faces = face_cascade.detectMultiScale(frame_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
        for (x, y, w, h) in faces:
        # Extract face region of interest
            face_roi = frame_rgb[y:y + h, x:x + w]

        # Detect eyes in face region
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if wearing glasses
            if len(eyes) >= 2:
                glasses_count += 1

##################################################################################################
    for item in multi_face_result:
        if item > 1:
            multi_face=1
        else:
            multi_face=0
            

    if len(face_ids)>1:
        multi_id=1
    else:
        multi_id=0

    # acculotion
    if mask_count>0 and glasses_count>0:
        acculotion=1

    return multi_face,acculotion,multi_id


def process(videoname):
    result = []
    multi_face,acculotion,multi_id=detect(videoname)
    result.append(0)
    result.append(multi_face)
    result.append(acculotion)
    result.append(multi_id)

    my_list = [0] * 15
    result.extend(my_list)

    return result
