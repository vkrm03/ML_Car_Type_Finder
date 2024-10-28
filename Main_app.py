from flask import Flask, render_template, request, redirect, Response, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and cascade classifier
model = load_model('car_type_model.h5')
car_cascade = cv2.CascadeClassifier('car_cascade_train.xml')

# Preprocess frame for model input
def preprocess_frame(frame):
    img_resized = cv2.resize(frame, (150, 150))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict car type
def predict_car_type(frame):
    img_array = preprocess_frame(frame)
    prediction = model.predict(img_array)
    prediction_value = prediction[0][0]
    car_type = "SUV" if prediction_value > 0.56 else "Sedan"
    return car_type, prediction_value

# Generate frames for live video feed
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=1)

        if len(cars) > 0:
            x, y, w, h = cars[0]
            car_roi = frame[y:y + h, x:x + w]
            car_type, prediction_probability = predict_car_type(car_roi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{car_type}: {prediction_probability:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/file_upload')
def file_upload():
    return render_template('file_upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        image = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=1)

        if len(cars) == 0:
            return render_template('result.html', car_type="No car detected", probability=0)

        for (x, y, w, h) in cars:
            car_roi = frame[y:y + h, x:x + w]
            car_type, prediction_probability = predict_car_type(car_roi)
            break

        return render_template('result.html', car_type=car_type, probability=prediction_probability)

@app.route('/cam')
def cam():
    return render_template('cam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
