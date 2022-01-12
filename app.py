######################################################################################################################################
# Imports
from flask import Flask, render_template, Response , request ,send_from_directory
import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms ,models
import torch.optim as optim
import os
import time
#####################################################################################################################################
# Initiate some variables
app = Flask(__name__)

camera = cv2.VideoCapture(0)
device = torch.device("cpu")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
#####################################################################################################################################  
# Prepare class Model
class ModeL(nn.Module):
    def __init__(self, n_classes,device= 'cuda'):

        super(ModeL, self).__init__()
        self.model = self._creat_Model(n_classes).to(device)  

    def _creat_Model(self,out_features, pretrained = False):

        model= models.efficientnet_b2(pretrained=pretrained)
        model.classifier =  nn.Sequential(
            nn.Linear(in_features=1408, out_features=out_features, bias=True),
            nn.LogSoftmax(dim = 1))

        return model

    def forward(self, x):
        x = self.model(x)
        return x
######################################################################################################################################
# Your transformation
transform= transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224,224)),
                                transforms.Normalize((0.5694, 0.4460, 0.3912), (0.2734, 0.2435, 0.2370))
                                                            ])
#####################################################################################################################################
# create and load the model
model =ModeL(4).to(device) 

state = torch.load("AFF_4-Class_Acc equal 90.6 train and 76.8 val .pt",map_location=device)
model.load_state_dict(state['state_dict'],strict=True)
model.eval()
######################################################################################################################################
# Function to predect the emotion
def img_ER (img,num_of_emos = 4):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    inputs = transform(img)

    final_image = torch.unsqueeze(inputs, 0)
    dataa=final_image.type(torch.FloatTensor).to(device)
    outputs = model(dataa)
    pred = F.softmax(outputs,dim=1)
    prediction = torch.argmax(pred)
    
    if num_of_emos == 4:
        if ((prediction) == 0):
            status = "Angry >:("
        elif ((prediction) == 1):
            status = "Happy ^^"  
        elif ((prediction) == 2):
            status = "Sad :( "   
        elif ((prediction) == 3): 
            status = "Surprise o_o' "
            
    else:
        if ((prediction) == 0):
            status = "Not Smiling T^T"
        elif ((prediction) == 1):
            status = "Smiling ^^"  
        elif ((prediction) == 2):
            status = "Not Smiling T^T"   
        elif ((prediction) == 3): 
            status = "Not Smiling T^T"
            
    return (status)
######################################################################################################################################
# Flask function
def gen_frames(num_of_emos):  
    pTime = 0
    with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.8) as face_detection:
        
        while True:
            success, image = camera.read()  # read the camera frame
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime
            cv2.putText(image,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
            if not success:
                print("Ignoring empty camera frame.")
                break
###################################################################################################################################### 
            else:

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                image_h = image.shape[0]
                image_w = image.shape[1]


                # Draw the face detection annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for face_no, detection in enumerate(results.detections):
                        try:
                            face_bounding = detection.location_data.relative_bounding_box
                            h = int((face_bounding.height) * image_h)
                            xmin = int(((face_bounding.xmin) * image_w))
                            w = int((face_bounding.width) * image_w)
                            ymin = int(((face_bounding.ymin) *image_h))

                            face_for_emo = image[ymin:ymin+h , xmin:xmin+w]
                            emo = img_ER(face_for_emo,num_of_emos)

                            cv2.putText(image,emo,(xmin+int(w/10),ymin-int(h/10)) ,
                                        cv2.FONT_HERSHEY_SIMPLEX, (face_bounding.width)+0.1 , (0,255,255) ,2+round(face_bounding.width)) 

#                             mp_drawing.draw_detection(image, detection)
                            cv2.rectangle(image, (xmin,ymin,w,h),(255,255,0),2+2*round(face_bounding.width + 0.1))
                        except:
                            pass
######################################################################################################################################
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
######################################################################################################################################
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico')#send_from_directory

@app.route('/')
def index():
    return render_template('index.html')
#############################################################################################

@app.route('/four_Emos',methods=["GET", "POST"])
def four_Emos():
    if request.method == "POST":
        if request.form['submit_button'] == "Check your emotion":
            return render_template('four_Emos.html') 

@app.route('/four_Emos_vid',methods=["GET", "POST"])
def four_Emos_vid():
    return (Response(gen_frames(4), mimetype='multipart/x-mixed-replace; boundary=frame'))
        
################################################################################################
@app.route('/two_Emos',methods=["GET", "POST"])
def two_Emos():
    if request.method == "POST":
        if request.form['submit_button'] == "Are you smiling":
            return render_template('two_Emos.html')

@app.route('/two_Emos_vid',methods=["GET", "POST"])
def two_Emos_vid():
    return (Response(gen_frames(2), mimetype='multipart/x-mixed-replace; boundary=frame'))

######################################################################################################################################
if __name__=='__main__':
    app.run(debug=True)
######################################################################################################################################