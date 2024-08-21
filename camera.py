from datetime import datetime 
import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from twilio.rest import Client
import geocoder  # to fetch the location

# Twilio configuration
account_sid = 'AC7c4b5f430b35bf82b3e7059c594b485a'
auth_token = '66d141afc01ca8e706c66ad955df1d95'
client = Client(account_sid, auth_token)

# Cloudinary configuration       
cloudinary.config( 
    cloud_name = "dw9gzgnzl", 
    api_key = "196663271749338", 
    api_secret = "BoSfgs8vf6fLGmN4b5QP6d8mMuc",  
    secure=True
)

# Load the accident detection model
model = AccidentDetectionModel("./model/model.json", './model/model_weights.keras')
font = cv2.FONT_HERSHEY_SIMPLEX

def upload_image_to_cloudinary(image):
    # Upload image to Cloudinary
    upload_result = cloudinary.uploader.upload(image)
    return upload_result["secure_url"]

def fetch_location():
    # Fetch the current location of the device
    g = geocoder.ip('me')
    return g.latlng

def send_sms(url, location,detection_time,probability):
    message_body = f"🚨 Accident Detected with {probability}% probability\nLocation: {location}\nImage: {url}\nTime: {detection_time}"
    message = client.messages.create(
        body=message_body,
        from_='whatsapp:+14155238886',
        to='whatsapp:+918449035579'
    )
    print(f"Message SID: {message.sid}")

def startapplication():
    video = cv2.VideoCapture('./video/cr.mp4') 
    # For camera use 
    # video = cv2.VideoCapture(0)
    above_threshold_duration = 0  # Timer in seconds
    check_interval = 1  # Interval to check in seconds

    while True:
        ret, frame = video.read()

        if not ret:
            print("Failed to grab frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        
        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)
            
            
            if prob > 95:
                above_threshold_duration += check_interval 
                os.system("say beep")
                if above_threshold_duration >= 10:
                    # Save the frame to a file
                    frame_filename = 'accident_frame.jpg'
                    cv2.imwrite(frame_filename, frame)
                    
                    # Upload the frame to Cloudinary
                    image_url = upload_image_to_cloudinary(frame_filename)
                    probability = prob
                    
                    # Fetch location
                    location = fetch_location()
                    detection_time = datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
                    
                    # Send SMS with the location and image URL
                    send_sms(image_url, location, detection_time,probability)
                    above_threshold_duration = 5


            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

        cv2.imshow('Video', frame)  

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()
