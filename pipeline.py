import cv2
import numpy as np
import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import mediapipe as mp
import json
import os
import subprocess
from tqdm import tqdm

class H2VProcessor:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_detection_model = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.mp_pose_model = self.mp_pose.Pose(min_detection_confidence=0.5)
        
        # Tracking parameters
        self.prev_focus_point = None
        self.focus_points_data = []
        
    def detect_scenes(self, video_path):
        """Split video into scenes using PySceneDetect"""
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30))
        
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        scene_list = scene_manager.get_scene_list()
        return scene_list
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        results = self.mp_face_detection_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                faces.append((x, y, w, h))
        return faces
    
    def detect_people(self, frame):
        """Detect people using pose estimation"""
        results = self.mp_pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        people = []
        if results.pose_landmarks:
            h, w, _ = frame.shape
            # Use nose landmark as person center
            if results.pose_landmarks.landmark[0].visibility > 0.5:  # Check if nose is visible
                nose_x = int(results.pose_landmarks.landmark[0].x * w)
                nose_y = int(results.pose_landmarks.landmark[0].y * h)
                people.append((nose_x, nose_y))
        return people
    
    def detect_motion(self, prev_frame, curr_frame):
        """Detect motion between frames using optical flow"""
        if prev_frame is None:
            return None
        
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Get magnitude and angle of flow
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Find area with most motion
        if np.max(magnitude) > 0.5:  # Only if significant motion exists
            y, x = np.unravel_index(magnitude.argmax(), magnitude.shape)
            return (x, y)
        
        return None
    
    def get_focus_point(self, frame, prev_frame=None, is_speaking=False):
        """Determine the main focus point in the frame"""
        h, w, _ = frame.shape
        
        # Priority 1: Speaking person
        if is_speaking:
            faces = self.detect_faces(frame)
            if faces:
                # Return center of the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, fw, fh = largest_face
                return (x + fw // 2, y + fh // 2)
        
        # Priority 2: People in frame
        people = self.detect_people(frame)
        if people:
            # Return position of first detected person
            return people[0]
        
        # Priority 3: Motion detection
        motion_point = self.detect_motion(prev_frame, frame)
        if motion_point:
            return motion_point
        
        # Fallback: use previous focus point or center of frame
        if self.prev_focus_point:
            return self.prev_focus_point
        
        return (w // 2, h // 2)
    
    def smooth_focus_point(self, new_point, alpha=0.3):
        """Apply smoothing to prevent jittery camera movement"""
        if self.prev_focus_point is None:
            self.prev_focus_point = new_point
            return new_point
        
        # Exponential smoothing
        smoothed_x = int(alpha * new_point[0] + (1 - alpha) * self.prev_focus_point[0])
        smoothed_y = int(alpha * new_point[1] + (1 - alpha) * self.prev_focus_point[1])
        
        self.prev_focus_point = (smoothed_x, smoothed_y)
        return self.prev_focus_point
    
    def crop_to_vertical(self, frame, focus_point):
        """Crop the frame to vertical 9:16 aspect ratio centered on focus point"""
        h, w = frame.shape[:2]
        
        # Calculate crop dimensions for 9:16 aspect ratio
        target_width = int(h * 9 / 16)
        
        # Center the crop on the focus point
        x_center, y_center = focus_point
        
        # Calculate crop boundaries
        left = max(0, x_center - target_width // 2)
        right = min(w, left + target_width)
        
        # Adjust if we hit the edge of the frame
        if right == w:
            left = max(0, w - target_width)
        if left == 0:
            right = min(w, target_width)
        
        # Perform the crop
        cropped = frame[:, left:right]
        
        # Return both the cropped frame and focus point data
        return cropped, {
            "time": None,  # Will be set later
            "original_frame": {"width": w, "height": h},
            "focus_point": {"x": x_center, "y": y_center},
            "crop": {"left": left, "right": right, "top": 0, "bottom": h}
        }
    
    def process_video(self, input_path, output_path, fps=None, temp_dir="temp_frames"):
        """Process the whole video and generate vertical version with tracking data"""
        # Create temp directory for frames if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use original fps if not specified
        if fps is None:
            fps = original_fps
        
        # Detect scenes
        scenes = self.detect_scenes(input_path)
        if not scenes:
            # If no scenes detected, treat the whole video as one scene
            scenes = [(0, total_frames)]
        
        # Output dimensions for 9:16 aspect ratio based on original height
        out_width = int(height * 9 / 16)
        
        # Process each scene
        prev_frame = None
        frame_count = 0
        
        # Reset tracking for new video
        self.prev_focus_point = None
        self.focus_points_data = []
        
        # Create temporary video path for frames only
        temp_video_path = os.path.splitext(output_path)[0] + "_temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (out_width, height))
        
        with tqdm(total=total_frames, desc="Processing frames") as progress_bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate current time in seconds
                current_time = frame_count / original_fps
                
                # Get focus point
                focus_point = self.get_focus_point(frame, prev_frame)
                
                # Apply smoothing
                smooth_point = self.smooth_focus_point(focus_point)
                
                # Crop frame
                cropped_frame, focus_data = self.crop_to_vertical(frame, smooth_point)
                
                # Add timestamp to focus data
                focus_data["time"] = current_time
                self.focus_points_data.append(focus_data)
                
                # Write frame to output video
                out.write(cropped_frame)
                
                # Update for next iteration
                prev_frame = frame.copy()
                frame_count += 1
                progress_bar.update(1)
        
        # Clean up video capture and writer
        cap.release()
        out.release()
        
        # Combine the processed video with the original audio using FFmpeg
        self.combine_video_with_audio(input_path, temp_video_path, output_path)
        
        # Remove temporary video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        # Export focus points data as JSON
        tracking_data_path = os.path.splitext(output_path)[0] + "_tracking.json"
        with open(tracking_data_path, 'w') as f:
            json.dump({"focus_points": self.focus_points_data}, f, indent=2)
        
        return output_path, tracking_data_path
    
    def combine_video_with_audio(self, original_video, processed_video, output_path):
        """Combine the processed video with the original audio using FFmpeg"""
        try:
            print("Adding audio track to the processed video...")
            # Command to extract audio from original video and combine with new video
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', processed_video,  # Input processed video (no audio)
                '-i', original_video,   # Input original video (for audio)
                '-c:v', 'copy',         # Copy video stream without re-encoding
                '-c:a', 'aac',          # Use AAC codec for audio
                '-map', '0:v:0',        # Use video from first input
                '-map', '1:a:0',        # Use audio from second input
                '-shortest',            # Finish encoding when the shortest input stream ends
                '-y',                   # Overwrite output file if it exists
                output_path
            ]
            
            # Run the FFmpeg command
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Successfully added audio to: {output_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error adding audio: {e}")
            print(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'No error output'}")
            # If FFmpeg fails, just use the video without audio
            import shutil
            shutil.copy(processed_video, output_path)
            print(f"Using video without audio: {output_path}")
        except Exception as e:
            print(f"Unexpected error adding audio: {e}")
            # If another error occurs, just use the video without audio
            import shutil
            shutil.copy(processed_video, output_path)
            print(f"Using video without audio: {output_path}")

# Example usage
if __name__ == "__main__":
    processor = H2VProcessor()
    
    input_video = "input_video.mp4"  # Replace with your video path
    output_video = "output_vertical_video.mp4"
    
    video_path, tracking_data = processor.process_video(input_video, output_video)
    print(f"Processing complete. Vertical video saved to: {video_path}")
    print(f"Tracking data saved to: {tracking_data}")
