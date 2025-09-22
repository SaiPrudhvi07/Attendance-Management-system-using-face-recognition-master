# app.py (Flask Backend)
from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import csv
import datetime
import time
import base64
import io
from PIL import Image
import pandas as pd

app = Flask(__name__)

# Paths configuration
haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "TrainingImageLabel/Trainner.yml"
trainimage_path = "TrainingImage"
studentdetail_path = "StudentDetails/studentdetails.csv"
attendance_path = "Attendance"

# Ensure directories exist
os.makedirs(trainimage_path, exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)
os.makedirs("StudentDetails", exist_ok=True)
os.makedirs(attendance_path, exist_ok=True)

# Initialize face detector
face_detector = cv2.CascadeClassifier(haarcasecade_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    # Simple authentication (in production, use proper authentication)
    if username == 'teacher123' and password == 'password123':
        return jsonify({'success': True, 'message': 'Login successful'})
    elif username == 'admin456' and password == 'admin123':
        return jsonify({'success': True, 'message': 'Login successful', 'isAdmin': True})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/get_classes')
def get_classes():
    # Read classes from student details
    classes = set()
    if os.path.exists(studentdetail_path):
        df = pd.read_csv(studentdetail_path)
        if 'Class' in df.columns:
            classes = sorted(df['Class'].unique())
    return jsonify(list(classes))

@app.route('/get_students')
def get_students():
    class_name = request.args.get('class')
    students = []
    
    if os.path.exists(studentdetail_path):
        df = pd.read_csv(studentdetail_path)
        if class_name and 'Class' in df.columns:
            df = df[df['Class'] == class_name]
        
        for _, row in df.iterrows():
            students.append({
                'id': row.get('Enrollment', ''),
                'name': row.get('Name', ''),
                'class': row.get('Class', ''),
                'roll': row.get('Roll', '')
            })
    
    return jsonify(students)

@app.route('/register_student', methods=['POST'])
def register_student():
    try:
        data = request.json
        enrollment = data.get('enrollment')
        name = data.get('name')
        class_name = data.get('class')
        roll = data.get('roll')
        
        # Save student details to CSV
        fieldnames = ['Enrollment', 'Name', 'Class', 'Roll']
        file_exists = os.path.isfile(studentdetail_path)
        
        with open(studentdetail_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'Enrollment': enrollment,
                'Name': name,
                'Class': class_name,
                'Roll': roll
            })
        
        return jsonify({'success': True, 'message': 'Student registered successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/capture_images', methods=['POST'])
def capture_images():
    try:
        data = request.json
        enrollment = data.get('enrollment')
        name = data.get('name')
        
        # Initialize webcam
        cam = cv2.VideoCapture(0)
        sampleNum = 0
        
        # Create directory for student images
        student_image_path = os.path.join(trainimage_path, enrollment)
        os.makedirs(student_image_path, exist_ok=True)
        
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                sampleNum += 1
                # Save the captured image
                cv2.imwrite(f"{student_image_path}/{name}.{enrollment}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Display countdown
                cv2.putText(img, f"Capturing sample {sampleNum}/50", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Show video feed
            cv2.imshow('Capturing Faces', img)
            
            # Wait for 100 milliseconds or 'q' key to stop
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 50:  # Take 50 face samples
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        return jsonify({'success': True, 'message': 'Images captured successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/train_model')
def train_model():
    try:
        # Training logic from the GitHub repo
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        faces = []
        ids = []
        
        # Get the training images and labels
        image_paths = [os.path.join(trainimage_path, f) for f in os.listdir(trainimage_path)]
        
        for image_path in image_paths:
            if os.path.isdir(image_path):
                enrollment = os.path.basename(image_path)
                for image_file in os.listdir(image_path):
                    if image_file.endswith('.jpg'):
                        img_path = os.path.join(image_path, image_file)
                        pil_image = Image.open(img_path).convert('L')
                        image_np = np.array(pil_image, 'uint8')
                        
                        # Extract enrollment from filename: name.enrollment.number.jpg
                        try:
                            id = int(enrollment)
                            faces.append(image_np)
                            ids.append(id)
                        except:
                            continue
        
        # Train the model
        recognizer.train(faces, np.array(ids))
        recognizer.save(trainimagelabel_path)
        
        return jsonify({'success': True, 'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    try:
        data = request.json
        class_name = data.get('class')
        subject = data.get('subject')
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainimagelabel_path)
        
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Initialize attendance dictionary
        attendance = {}
        
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
                Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                if confidence < 50:
                    # Get student name from CSV
                    df = pd.read_csv(studentdetail_path)
                    student = df[df['Enrollment'] == Id]
                    if not student.empty:
                        name = student['Name'].values[0]
                        attendance[Id] = {
                            'name': name,
                            'status': 'Present',
                            'time': datetime.datetime.now().strftime("%H:%M:%S")
                        }
                        cv2.putText(im, name, (x, y+h), font, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(im, "Unknown", (x, y+h), font, 1, (0, 0, 255), 2)
                
                cv2.putText(im, f"{str(confidence)}%", (x, y+h+30), font, 1, (255, 255, 0), 1)
            
            cv2.imshow('Taking Attendance', im)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Save attendance to CSV
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        attendance_filename = f"{attendance_path}/Attendance_{subject}_{date_str}_{time_str}.csv"
        
        with open(attendance_filename, 'w', newline='') as csvfile:
            fieldnames = ['Enrollment', 'Name', 'Class', 'Status', 'Time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for enrollment, data in attendance.items():
                # Get class from student details
                df = pd.read_csv(studentdetail_path)
                student = df[df['Enrollment'] == enrollment]
                class_name = student['Class'].values[0] if not student.empty else 'Unknown'
                
                writer.writerow({
                    'Enrollment': enrollment,
                    'Name': data['name'],
                    'Class': class_name,
                    'Status': data['status'],
                    'Time': data['time']
                })
        
        return jsonify({'success': True, 'message': 'Attendance taken successfully', 'attendance': attendance})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_attendance_reports')
def get_attendance_reports():
    class_name = request.args.get('class')
    date = request.args.get('date')
    subject = request.args.get('subject')
    
    reports = []
    
    # Get all attendance files
    attendance_files = [f for f in os.listdir(attendance_path) if f.endswith('.csv')]
    
    for file in attendance_files:
        file_path = os.path.join(attendance_path, file)
        df = pd.read_csv(file_path)
        
        # Filter by class if specified
        if class_name and 'Class' in df.columns:
            df = df[df['Class'] == class_name]
        
        # Filter by date if specified (file name contains date)
        if date and date in file:
            pass  # File already filtered by date in filename
        
        # Filter by subject if specified (file name contains subject)
        if subject and subject in file:
            pass  # File already filtered by subject in filename
        
        if not df.empty:
            reports.append({
                'filename': file,
                'date': file.split('_')[2],  # Extract date from filename
                'subject': file.split('_')[1],  # Extract subject from filename
                'total_students': len(df),
                'present': len(df[df['Status'] == 'Present']),
                'absent': len(df[df['Status'] == 'Absent']) if 'Absent' in df['Status'].values else 0
            })
    
    return jsonify(reports)

@app.route('/get_attendance_details')
def get_attendance_details():
    filename = request.args.get('filename')
    file_path = os.path.join(attendance_path, filename)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return jsonify(df.to_dict('records'))
    else:
        return jsonify({'error': 'File not found'})

if __name__ == '__main__':
    app.run(debug=True)