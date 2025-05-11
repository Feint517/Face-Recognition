import os
import pickle
import cv2
import numpy as np
import shutil
from datetime import datetime

class FaceDatabase:
    def __init__(self, database_dir='face_database'):
        # Get the current directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set up database directory
        self.database_dir = os.path.join(self.current_dir, database_dir)
        self.images_dir = os.path.join(self.database_dir, 'images')
        self.data_file = os.path.join(self.database_dir, 'face_data.pkl')
        
        # Create directories if they don't exist
        os.makedirs(self.database_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize database
        self.face_data = self.load_database()
    
    def load_database(self):
        """Load the face database from disk"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    face_data = pickle.load(f)
                print(f"Loaded face database with {len(face_data)} entries")
                return face_data
            except Exception as e:
                print(f"Error loading face database: {e}")
                return {}
        else:
            print("No existing face database found. Creating new database.")
            return {}
    
    def save_database(self):
        """Save the face database to disk"""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.face_data, f)
            print(f"Saved face database with {len(self.face_data)} entries")
            return True
        except Exception as e:
            print(f"Error saving face database: {e}")
            return False
    
    def add_person(self, name, face_image=None):
        """Add a new person to the database"""
        if not name or name.strip() == "":
            print("Error: Name cannot be empty")
            return False
        
        name = name.strip()
        
        # Generate a unique ID for the person
        person_id = str(len(self.face_data) + 1).zfill(3)
        while person_id in self.face_data:
            person_id = str(int(person_id) + 1).zfill(3)
        
        # Create entry in database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.face_data[person_id] = {
            'name': name,
            'added_on': timestamp,
            'last_seen': timestamp,
            'images': []
        }
        
        # Save face image if provided
        if face_image is not None:
            self.add_face_image(person_id, face_image)
        
        # Save database
        self.save_database()
        
        print(f"Added person: {name} (ID: {person_id})")
        return person_id
    
    def add_face_image(self, person_id, face_image):
        """Add a face image for a person"""
        if person_id not in self.face_data:
            print(f"Error: Person ID {person_id} not found in database")
            return False
        
        try:
            # Create directory for this person if it doesn't exist
            person_dir = os.path.join(self.images_dir, person_id)
            os.makedirs(person_dir, exist_ok=True)
            
            # Generate filename for the image
            image_count = len(self.face_data[person_id]['images']) + 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_id}_{timestamp}_{image_count}.jpg"
            filepath = os.path.join(person_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, face_image)
            
            # Add to database
            self.face_data[person_id]['images'].append(filename)
            self.face_data[person_id]['last_seen'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save database
            self.save_database()
            
            print(f"Added face image for {self.face_data[person_id]['name']} (ID: {person_id})")
            return True
        except Exception as e:
            print(f"Error adding face image: {e}")
            return False
    
    def get_person_by_id(self, person_id):
        """Get person data by ID"""
        if person_id in self.face_data:
            return self.face_data[person_id]
        else:
            print(f"Error: Person ID {person_id} not found in database")
            return None
    
    def get_person_by_name(self, name):
        """Get person data by name"""
        for person_id, data in self.face_data.items():
            if data['name'].lower() == name.lower():
                return person_id, data
        return None, None
    
    def update_person_name(self, person_id, new_name):
        """Update a person's name"""
        if person_id not in self.face_data:
            print(f"Error: Person ID {person_id} not found in database")
            return False
        
        if not new_name or new_name.strip() == "":
            print("Error: Name cannot be empty")
            return False
        
        self.face_data[person_id]['name'] = new_name.strip()
        self.save_database()
        
        print(f"Updated name for ID {person_id} to {new_name}")
        return True
    
    def remove_person(self, person_id):
        """Remove a person from the database"""
        if person_id not in self.face_data:
            print(f"Error: Person ID {person_id} not found in database")
            return False
        
        # Remove person's image directory
        person_dir = os.path.join(self.images_dir, person_id)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        
        # Remove from database
        name = self.face_data[person_id]['name']
        del self.face_data[person_id]
        self.save_database()
        
        print(f"Removed person: {name} (ID: {person_id})")
        return True
    
    def list_all_people(self):
        """List all people in the database"""
        if not self.face_data:
            print("Database is empty")
            return []
        
        people = []
        print("Face Database Contents:")
        print("----------------------")
        for person_id, data in self.face_data.items():
            print(f"ID: {person_id}, Name: {data['name']}, Images: {len(data['images'])}")
            print(f"  Added: {data['added_on']}, Last seen: {data['last_seen']}")
            people.append((person_id, data['name'], len(data['images'])))
        
        return people
    
    def update_last_seen(self, person_id):
        """Update the last seen timestamp for a person"""
        if person_id not in self.face_data:
            return False
        
        self.face_data[person_id]['last_seen'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_database()
        return True
    
    def get_training_data(self):
        """Get training data for face recognition"""
        faces = []
        labels = []
        label_ids = {}
        
        # Sort keys to ensure consistent ordering
        sorted_person_ids = sorted(self.face_data.keys())
        
        for idx, person_id in enumerate(sorted_person_ids):
            label_ids[person_id] = idx
            person_dir = os.path.join(self.images_dir, person_id)
            
            if not os.path.exists(person_dir):
                continue
                
            for image_name in self.face_data[person_id]['images']:
                image_path = os.path.join(person_dir, image_name)
                if os.path.exists(image_path):
                    face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if face_img is not None:
                        faces.append(face_img)
                        labels.append(idx)
        
        return faces, labels, label_ids
