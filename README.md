# 🏫 Rural School Attendance System
An open-source, facial recognition-based attendance management system designed specifically for the unique challenges of rural schools. This system provides a simple, accurate, and efficient way for teachers to track student attendance.

✨ Features
This project automates and streamlines the attendance process with a user-friendly interface and robust backend functionality.

👥 Face-Based Attendance: Utilizes a webcam and a trained facial recognition model to automatically mark student attendance in real-time.

🧑‍🎓 Student Management: Allows for the registration of new students, including capturing multiple face samples for accurate model training.

👨‍🏫 Teacher & Admin Panels: A secure login system with separate dashboards for teachers and administrators to manage classes, students, and reports.

✍️ Manual Entry: Offers a manual attendance option for situations where facial recognition may not be feasible.

📊 Reporting & Analytics: Generates detailed attendance reports, including class-wise comparisons, daily summaries, and individual student records.

💾 Offline-Ready: The system is designed to function and store data locally, making it resilient to internet connectivity issues common in rural areas.

⚙️ Technology Stack
This project is built using a combination of popular Python libraries and web technologies.

Backend:

Python: The core programming language.

Flask: A lightweight and flexible web framework for the backend server.

OpenCV (cv2): A powerful library for computer vision tasks, including face detection and recognition.

Numpy: Used for numerical operations, especially with image data.

Pandas: For efficient handling and analysis of student and attendance data stored in CSV files.

Frontend:

HTML, CSS, JavaScript: The foundation for the web interface.

Font Awesome: Provides the icons used in the UI for a clean, intuitive design.

🚀 Getting Started
Follow these steps to set up and run the project locally.

Prerequisites
Make sure you have Python 3.x and pip installed on your system.

Installation
Clone the repository:

Bash

git clone [repository-url]
cd Rural-School-Attendance-System
Install the required Python packages:

Bash

pip install -r requirements.txt
Note: You will need to create a requirements.txt file listing the dependencies from app.py (e.g., Flask, opencv-python, numpy, pandas).

Usage
Place the Haar Cascade file:
Download the haarcascade_frontalface_default.xml file and place it in the project directory. This file is essential for the facial recognition process.

Run the Flask application:

Bash

python app.py
Access the application:
Open your web browser and navigate to http://127.0.0.1:5000 to access the login page.

Demo Credentials
Teacher: teacher123 / password123

Admin: admin456 / admin123

🤝 Contributing
Contributions are welcome! Please feel free to open a pull request or report issues. Before contributing, please read our Contribution Guidelines (if you create one).
