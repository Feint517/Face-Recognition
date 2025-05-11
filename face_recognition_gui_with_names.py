import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import PIL.Image, PIL.ImageTk
from face_database import FaceDatabase

class FaceRecognitionApp:
    def __init__(self, window, window_title):
        # Initialize the main window
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="#f0f0f0")
        self.window.resizable(True, True)
        self.window.minsize(1000, 700)
        
        # Set window icon if available
        try:
            self.window.iconbitmap("face_icon.ico")
        except:
            pass
        
        # Initialize variables
        self.is_running = False
        self.thread = None
        self.stopEvent = None
        self.is_training = False
        self.current_name = ""
        self.training_faces = []
        self.training_count = 0
        self.max_training_faces = 30
        
        # Initialize face database
        self.face_db = FaceDatabase()
        
        # Initialize face detector and recognizer
        self.init_face_recognition()
        
        # Create UI elements
        self.create_ui()
        
        # Set window close handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Center window on screen
        self.center_window()
        
    
    def center_window(self):
        """Center the window on the screen"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry('{}x{}+{}+{}'.format(width, height, x, y))


    def init_face_recognition(self):
        """Initialize face recognition components"""
        # Get the current directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the pre-trained face detection model (Haar Cascade)
        haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        
        # Initialize face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,           # Controls the LBP operator radius
            neighbors=8,        # Number of sample points for the LBP operator
            grid_x=8,           # Number of cells in horizontal direction
            grid_y=8,           # Number of cells in vertical direction
            threshold=100.0     # Confidence threshold for recognition
        )
        
        self.model_trained = False
        
        # Load recognizer if exists
        self.recognizer_path = os.path.join(self.current_dir, 'face_recognizer.yml')
        self.load_recognizer()
        
        # Settings
        self.detection_confidence = 45  # Confidence threshold for recognition (0-100)
        self.show_fps = True
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.face_size = (100, 100)  # Size to normalize faces
        
        # Label mapping
        self.label_to_id = {}
        
        # Train recognizer with existing database
        self.train_from_database()

    
    def load_recognizer(self):
        """Load the face recognizer if it exists"""
        try:
            if os.path.exists(self.recognizer_path):
                self.recognizer.read(self.recognizer_path)
            
                # Load the label mapping
                label_map_path = os.path.join(self.current_dir, 'label_mapping.pkl')
                if os.path.exists(label_map_path):
                    with open(label_map_path, 'rb') as f:
                        self.label_to_id = pickle.load(f)
                    print("Face recognizer model and label mapping loaded successfully")
                    self.model_trained = True  # Set flag when model is successfully loaded
                else:
                    print("Warning: Label mapping not found, recognition may not work correctly")
            else:
                print("No existing face recognizer model found")
        except Exception as e:
            print(f"Error loading face recognizer: {e}")
            print("Starting with a new recognizer")
            self.model_trained = False  # Ensure flag is reset on error


    def save_recognizer(self):
        """Save the face recognizer"""
        try:
            if hasattr(self, 'recognizer'):
                self.recognizer.write(self.recognizer_path)
            
                # Save the label mapping alongside the model
                label_map_path = os.path.join(self.current_dir, 'label_mapping.pkl')
                with open(label_map_path, 'wb') as f:
                    pickle.dump(self.label_to_id, f)
            
                print(f"Face recognizer and label mapping saved successfully")
                return True
            else:
                print("No recognizer to save")
                return False
        except Exception as e:
            print(f"Error saving face recognizer: {e}")
            return False   
    
    
    def train_from_database(self):
        """Train the recognizer using the face database"""
        faces, labels, label_ids = self.face_db.get_training_data()
        
        if len(faces) == 0 or len(labels) == 0:
            print("No training data available in database")
            return False
        
        try:
            # Convert to numpy arrays
            faces = np.array(faces)
            labels = np.array(labels)
            
            # Train the recognizer
            self.recognizer.train(faces, labels)
            
            # Store label to ID mapping
            self.label_to_id = {v: k for k, v in label_ids.items()}
            
            print(f"Trained recognizer with {len(faces)} faces from {len(label_ids)} people")
            print(f"Label to ID mapping: {self.label_to_id}")  # Debug print
            self.model_trained = True
            return True
        except Exception as e:
            print(f"Error training from database: {e}")
            self.model_trained = False
            return False
    
    
    def create_ui(self):
        """Create the user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for video display
        self.video_frame = ttk.Frame(main_frame, borderwidth=2, relief="groove")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for video display
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create right panel for controls
        control_frame = ttk.Frame(main_frame, padding="10", borderwidth=2, relief="groove", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Title label
        title_label = ttk.Label(control_frame, text="Face Recognition", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Start/Stop button
        self.btn_start_stop = ttk.Button(control_frame, text="Start Camera", command=self.toggle_camera)
        self.btn_start_stop.pack(fill=tk.X, pady=5)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Mode frame
        mode_frame = ttk.LabelFrame(control_frame, text="Mode")
        mode_frame.pack(fill=tk.X, pady=5)
        
        # Mode buttons
        self.btn_recognition = ttk.Button(mode_frame, text="Recognition Mode", 
                                            command=lambda: self.set_mode("recognition"))
        self.btn_recognition.pack(fill=tk.X, padx=5, pady=2)
        self.btn_recognition.config(state=tk.DISABLED)
        
        self.btn_training = ttk.Button(mode_frame, text="Training Mode", 
                                        command=lambda: self.set_mode("training"))
        self.btn_training.pack(fill=tk.X, padx=5, pady=2)
        self.btn_training.config(state=tk.DISABLED)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(control_frame, text="Settings")
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Show FPS checkbox
        self.show_fps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Show FPS", variable=self.show_fps_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Confidence threshold slider
        ttk.Label(settings_frame, text="Recognition Confidence Threshold:").pack(anchor=tk.W, padx=5, pady=2)
        self.confidence_var = tk.IntVar(value=45)
        confidence_scale = ttk.Scale(settings_frame, from_=50, to=95, variable=self.confidence_var, 
                                    orient=tk.HORIZONTAL, length=200)
        confidence_scale.pack(anchor=tk.W, padx=5, pady=2)
        
        # Confidence value label
        self.confidence_label = ttk.Label(settings_frame, text="80%")
        self.confidence_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Update confidence label when slider changes
        self.confidence_var.trace_add("write", self.update_confidence_label)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Database management frame
        db_frame = ttk.LabelFrame(control_frame, text="Database Management")
        db_frame.pack(fill=tk.X, pady=5)
        
        # Database buttons
        self.btn_add_person = ttk.Button(db_frame, text="Add New Person", 
                                        command=self.add_new_person)
        self.btn_add_person.pack(fill=tk.X, padx=5, pady=2)
        
        self.btn_view_people = ttk.Button(db_frame, text="View All People", 
                                            command=self.view_all_people)
        self.btn_view_people.pack(fill=tk.X, padx=5, pady=2)
        
        self.btn_remove_person = ttk.Button(db_frame, text="Remove Person", 
                                            command=self.remove_person)
        self.btn_remove_person.pack(fill=tk.X, padx=5, pady=2)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(control_frame, text="Status")
        status_frame.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Camera not started")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, wraplength=280)
        self.status_label.pack(padx=5, pady=5, fill=tk.X)
        
        # Face count label
        self.face_count_var = tk.StringVar(value="Faces detected: 0")
        self.face_count_label = ttk.Label(status_frame, textvariable=self.face_count_var)
        self.face_count_label.pack(padx=5, pady=5)
        
        # Training progress frame (initially hidden)
        self.training_frame = ttk.LabelFrame(control_frame, text="Training Progress")
        
        # Training progress labels
        self.training_name_var = tk.StringVar(value="")
        self.training_name_label = ttk.Label(self.training_frame, textvariable=self.training_name_var)
        self.training_name_label.pack(padx=5, pady=5)
        
        self.training_progress_var = tk.StringVar(value="")
        self.training_progress_label = ttk.Label(self.training_frame, textvariable=self.training_progress_var)
        self.training_progress_label.pack(padx=5, pady=5)
        
        # Training progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.training_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(padx=5, pady=5, fill=tk.X)
        
        # Cancel training button
        self.btn_cancel_training = ttk.Button(self.training_frame, text="Cancel Training", 
                                                command=self.cancel_training)
        self.btn_cancel_training.pack(padx=5, pady=5, fill=tk.X)
        
        # About button at the bottom
        self.btn_about = ttk.Button(control_frame, text="About", command=self.show_about)
        self.btn_about.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    
    
    def start_training(self):
        """Start training mode"""
        if self.is_training:
            return
        
        # Prompt for name
        name = simpledialog.askstring("Training", "Enter name for training:", parent=self.window)
        if not name or name.strip() == "":
            return
        
        self.is_training = True
        self.current_name = name.strip()
        self.training_faces = []
        self.training_count = 0
        
        # Update UI
        self.btn_recognition.config(state=tk.DISABLED)
        self.btn_training.config(state=tk.DISABLED)
        self.training_name_var.set(f"Training: {self.current_name}")
        self.training_progress_var.set(f"Progress: 0/{self.max_training_faces}")
        self.progress_var.set(0)
        self.training_frame.pack(fill=tk.X, pady=5, before=self.btn_about)
        self.status_var.set(f"Training mode: Collecting faces for {self.current_name}")
        
        print(f"Started training for: {self.current_name}")
    
    
    def add_training_face(self, face_roi):
        """Add a face for training"""
        if not self.is_training:
            return False
        
        try:
            # Resize face to standard size
            face_roi_resized = cv2.resize(face_roi, self.face_size)
            
            # Add to training set
            self.training_faces.append(face_roi_resized)
            self.training_count += 1
            
            # Update UI
            progress_percent = (self.training_count / self.max_training_faces) * 100
            self.training_progress_var.set(f"Progress: {self.training_count}/{self.max_training_faces}")
            self.progress_var.set(progress_percent)
            
            # Check if we have enough faces
            if self.training_count >= self.max_training_faces:
                self.complete_training()
                
            return True
        except Exception as e:
            print(f"Error adding training face: {e}")
            return False
    
    
    def complete_training(self):
        """Complete the training process"""
        if not self.is_training or len(self.training_faces) == 0:
            self.is_training = False
            return False
        
        try:
            # Update status
            self.status_var.set(f"Processing training data for {self.current_name}...")
            self.window.update()
            
            # Check if person already exists in database
            person_id, existing_data = self.face_db.get_person_by_name(self.current_name)
            
            # If not, add new person
            if person_id is None:
                person_id = self.face_db.add_person(self.current_name)
            
            # Add face images to database
            for face in self.training_faces:
                self.face_db.add_face_image(person_id, face)
            
            # Retrain the recognizer with the updated database
            success = self.train_from_database()
            if not success:
                raise Exception("Failed to train the model with new data")
            
            # Save the model
            self.save_recognizer()
            
            # Update UI
            self.status_var.set(f"Training completed for {self.current_name}")
            messagebox.showinfo("Training Complete", 
                                f"Successfully trained {self.current_name} with {len(self.training_faces)} faces.")
            
            print(f"Training completed for {self.current_name} with {len(self.training_faces)} faces")
            
            # Reset training state
            self.is_training = False
            self.current_name = ""
            self.training_faces = []
            self.training_count = 0
            
            # Update UI
            self.training_frame.pack_forget()
            self.btn_recognition.config(state=tk.NORMAL)
            self.btn_training.config(state=tk.NORMAL)
            
            return True
        except Exception as e:
            print(f"Error completing training: {e}")
            self.status_var.set(f"Error during training: {str(e)}")
            self.is_training = False
            self.training_frame.pack_forget()
            self.btn_recognition.config(state=tk.NORMAL)
            self.btn_training.config(state=tk.NORMAL)
            return False
    
    
    def cancel_training(self):
        """Cancel the current training session"""
        if self.is_training:
            self.is_training = False
            self.current_name = ""
            self.training_faces = []
            self.training_count = 0
            
            # Update UI
            self.training_frame.pack_forget()
            self.btn_recognition.config(state=tk.NORMAL)
            self.btn_training.config(state=tk.NORMAL)
            self.status_var.set("Training cancelled")
            
            print("Training cancelled")
            return True
        return False
    
    
    def video_loop(self):
        """Main video processing loop"""
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Error: Could not open camera")
            self.btn_start_stop.config(text="Start Camera")
            self.is_running = False
            return
        
        self.status_var.set("Camera running")
        self.prev_frame_time = time.time()
        
        try:
            while not self.stopEvent.is_set():
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret:
                    self.status_var.set("Error: Failed to capture image")
                    break
                
                # Mirror the frame horizontally (selfie view)
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces, gray = self.detect_faces(frame)
                
                # Update face count
                self.face_count_var.set(f"Faces detected: {len(faces)}")
                
                # Process faces
                frame = self.process_faces(frame, faces, gray)
                
                # Calculate and display FPS if enabled
                if self.show_fps_var.get():
                    frame = self.calculate_fps(frame)
                
                # Convert to RGB for tkinter
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL format
                pil_image = PIL.Image.fromarray(cv2image)
                
                # Resize to fit canvas if needed
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
                    img_width, img_height = pil_image.size
                    
                    # Calculate scaling factor to fit in canvas
                    scale_width = canvas_width / img_width
                    scale_height = canvas_height / img_height
                    scale = min(scale_width, scale_height)
                    
                    # Resize image
                    if scale < 1:  # Only resize if image is larger than canvas
                        new_width = int(img_width * scale)
                        new_height = int(img_height * scale)
                        pil_image = pil_image.resize((new_width, new_height), PIL.Image.LANCZOS)
                
                # Convert to ImageTk format
                self.current_image = PIL.ImageTk.PhotoImage(image=pil_image)
                
                # Update canvas
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                        image=self.current_image, anchor=tk.CENTER)
                
                # Update the window
                self.window.update_idletasks()
                self.window.update()
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error in video loop: {e}")
        finally:
            # Ensure resources are released but DON'T call stop_camera again
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()
            print("Video loop ended")
    
    
    def detect_faces(self, frame):
        """Detect faces in the given frame"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces, gray
    
    
    def process_faces(self, frame, faces, gray):
        """Process detected faces (recognize or train)"""
        for (x, y, w, h) in faces:
            # Extract face region of interest
            face_roi = gray[y:y+h, x:x+w]
            
            # Apply histogram equalization for better lighting normalization
            face_roi = cv2.equalizeHist(face_roi)
            
            # If in training mode, add this face to training set
            if self.is_training:
                if self.add_training_face(face_roi):
                    # Draw green rectangle for training faces
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add training progress
                    progress_text = f"Training: {self.training_count}/{self.max_training_faces}"
                    cv2.putText(frame, progress_text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Recognize the face
                name, confidence, person_id = self.recognize_face(face_roi)
                
                # Choose color based on recognition (green for known, red for unknown)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Prepare text for display
                text = f"{name} ({confidence:.0f}%)"
                
                # Increase text size and thickness
                font_scale = 0.8  # Increased from 0.5
                thickness = 2     # Increased from 1
            
                # Get text size to create background
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
                # Draw background rectangle for text
                cv2.rectangle(frame, 
                            (x, y-text_height-baseline-10), 
                            (x+text_width, y-5), 
                            (0, 0, 0), -1)  # -1 fills the rectangle
            
                # Add name and confidence with larger font
                cv2.putText(frame, text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        
        return frame


    def recognize_face(self, face_roi):
        """Recognize a face using the trained model"""
        try:
            if not hasattr(self, 'model_trained') or not self.model_trained:
                print("Warning: Face recognizer is not trained yet!")
                return "Not Trained", 0, None
        
            # Resize face to standard size
            face_roi_resized = cv2.resize(face_roi, self.face_size)
        
            # Predict using the recognizer
            label, distance = self.recognizer.predict(face_roi_resized)
        
            # Convert LBPH distance to confidence percentage (better formula)
            # In LBPH, distance of 0 is perfect match, typically anything under 50-80 is good
            # Maximum reasonable distance is around 200
            MAX_DISTANCE = 200.0
            confidence_score = max(0, min(100, 100 * (1 - distance / MAX_DISTANCE)))
        
            # Debug info
            print(f"Recognition: label={label}, raw_distance={distance:.2f}, confidence={confidence_score:.2f}%")
        
            if confidence_score >= self.detection_confidence and label in self.label_to_id:
                person_id = self.label_to_id[label]
                person_data = self.face_db.get_person_by_id(person_id)
            
                if person_data:
                    # Update last seen timestamp
                    self.face_db.update_last_seen(person_id)
                    return person_data['name'], confidence_score, person_id
        
            return "Unknown", confidence_score, None
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return "Error", 0, None
    
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    
    def start_camera(self):
        """Start the camera and detection process"""
        if self.is_running:
            return
        
        # Update UI
        self.btn_start_stop.config(text="Stop Camera")
        self.status_var.set("Starting camera...")
        self.window.update()
        
        # Create a stop event for the thread
        self.stopEvent = threading.Event()
        
        # Start camera in a separate thread
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.daemon = True
        self.thread.start()
        
        # Enable mode buttons
        self.btn_recognition.config(state=tk.NORMAL)
        self.btn_training.config(state=tk.NORMAL)
        
        self.is_running = True
    
    
    def stop_camera(self):
        """Stop the camera and detection process"""
        if not self.is_running:
            return
    
        # Signal the thread to stop
        if self.stopEvent is not None:
            self.stopEvent.set()
    
        # Update UI immediately
        self.btn_start_stop.config(text="Start Camera")
        self.status_var.set("Camera stopping...")
        self.face_count_var.set("Faces detected: 0")
    
        # Clear canvas
        self.canvas.delete("all")
        self.canvas.create_text(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, 
                            text="Camera Off", fill="white", font=("Arial", 20))
    
        # Disable mode buttons
        self.btn_recognition.config(state=tk.DISABLED)
        self.btn_training.config(state=tk.DISABLED)
    
        # Hide training frame if visible
        if self.is_training:
            self.cancel_training()
    
        self.is_running = False
    
        # Start a timer to periodically check if the thread has ended
        # and do the cleanup once it's done
        self.window.after(100, self.check_thread_ended)
    
    
    def check_thread_ended(self):
        """Check if the camera thread has ended and release resources if it has"""
        if self.thread is not None and not self.thread.is_alive():
            # Thread has ended, release resources
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.status_var.set("Camera stopped")
            self.thread = None
        elif self.thread is not None:
            # Thread still running, check again later
            self.window.after(100, self.check_thread_ended)
    
    
    def set_mode(self, mode):
        """Set the current mode (recognition or training)"""
        if mode == "training" and not self.is_training:
            self.start_training()
        elif mode == "recognition" and self.is_training:
            self.cancel_training()
            
        if mode == "recognition" and not self.model_trained:
            # If there's training data, train the model
            if self.train_from_database():
                print("Trained model before switching to recognition mode")
            else:
                messagebox.showwarning("Recognition Mode", 
                                        "No trained model exists. Add people to the database first.")
    
    
    def update_confidence_label(self, *args):
        """Update the confidence threshold label"""
        self.confidence_label.config(text=f"{self.confidence_var.get()}%")
        self.detection_confidence = self.confidence_var.get()
    
    
    def add_new_person(self):
        """Add a new person to the database"""
        if not self.is_running:
            messagebox.showinfo("Camera Required", "Please start the camera first.")
            return
        
        if self.is_training:
            messagebox.showinfo("Training in Progress", "Please complete or cancel current training first.")
            return
        
        self.start_training()
    
    
    def view_all_people(self):
        """View all people in the database"""
        people = self.face_db.list_all_people()
        
        if not people:
            messagebox.showinfo("Database Empty", "No people in the database.")
            return
        
        # Create a new window to display people
        people_window = tk.Toplevel(self.window)
        people_window.title("People Database")
        people_window.geometry("500x400")
        people_window.resizable(True, True)
        
        # Create a frame for the list
        frame = ttk.Frame(people_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a treeview to display people
        columns = ("id", "name", "images")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        # Define headings
        tree.heading("id", text="ID")
        tree.heading("name", text="Name")
        tree.heading("images", text="Images")
        
        # Define column widths
        tree.column("id", width=50)
        tree.column("name", width=200)
        tree.column("images", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add people to the treeview
        for person_id, name, image_count in people:
            tree.insert("", tk.END, values=(person_id, name, image_count))
    
    
    def remove_person(self):
        """Remove a person from the database"""
        people = self.face_db.list_all_people()
        
        if not people:
            messagebox.showinfo("Database Empty", "No people in the database.")
            return
        
        # Create a list of names for the dropdown
        names = [f"{name} (ID: {person_id})" for person_id, name, _ in people]
        
        # Create a new window for removal
        remove_window = tk.Toplevel(self.window)
        remove_window.title("Remove Person")
        remove_window.geometry("400x200")
        remove_window.resizable(False, False)
        
        # Create a frame
        frame = ttk.Frame(remove_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add label
        ttk.Label(frame, text="Select person to remove:").pack(pady=10)
        
        # Add dropdown
        person_var = tk.StringVar()
        dropdown = ttk.Combobox(frame, textvariable=person_var, values=names, state="readonly", width=30)
        dropdown.pack(pady=10)
        dropdown.current(0)
        
        # Add remove button
        def do_remove():
            if not person_var.get():
                return
                
            # Extract ID from selection
            selection = person_var.get()
            person_id = selection.split("(ID: ")[1].split(")")[0]
            
            # Confirm removal
            if messagebox.askyesno("Confirm Removal", f"Are you sure you want to remove {selection}?"):
                if self.face_db.remove_person(person_id):
                    messagebox.showinfo("Success", f"Successfully removed {selection}")
                    # Retrain recognizer
                    self.train_from_database()
                    self.save_recognizer()
                    remove_window.destroy()
                else:
                    messagebox.showerror("Error", f"Failed to remove {selection}")
        
        ttk.Button(frame, text="Remove", command=do_remove).pack(pady=10)
        ttk.Button(frame, text="Cancel", command=remove_window.destroy).pack(pady=5)
    
    
    def calculate_fps(self, frame):
        """Calculate and display FPS on frame"""
        self.new_frame_time = time.time()
        fps = 1/(self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        fps = int(fps)
        
        cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Face Recognition with Names

This application uses computer vision to detect and recognize faces in real-time using your webcam.

Features:
• Real-time face detection and recognition
• Training mode to add new faces
• Database management for known faces
• Confidence threshold adjustment
• Performance monitoring

Created with OpenCV and Tkinter

© 2025"""
        messagebox.showinfo("About Face Recognition", about_text)
    
    
    def on_closing(self):
        """Handle window close event"""
        if self.is_running:
            self.stop_camera()
        self.window.destroy()

if __name__ == "__main__":
    # Create tkinter window
    root = tk.Tk()
    
    # Create app
    app = FaceRecognitionApp(root, "Face Recognition with Names")
    
    # Start the main loop
    root.mainloop()
