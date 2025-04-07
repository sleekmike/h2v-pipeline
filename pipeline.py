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
import collections
import time
from virtual_camera import VirtualCamera
from scene_analyzer import SceneContentAnalyzer
from smooth_cropper import SmoothCropper




class ImprovedH2VProcessor:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_detection_model = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.mp_pose_model = self.mp_pose.Pose(min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)

        # Initialize advanced components
        self.virtual_camera = VirtualCamera()
        self.scene_analyzer = SceneContentAnalyzer(self, self)  # Pass self as face/pose detector
        self.smooth_cropper = SmoothCropper(target_aspect_ratio=9/16)

        # Scene tracking
        self.current_shot_id = 0
        self.shot_boundaries = []
        self.current_scene_frames = []
        self.current_scene_analysis = None
        self.scenes_without_people = []
        
        # Tracking data
        self.focus_points_data = []
        self.prev_frame = None
        
        # Stabilization history
        self.prev_focus_point = None
        self.focus_point_history = collections.deque(maxlen=30)
        
        # Enhanced stability parameters
        self.temporal_smoothing_window = 15  # Frames for temporal averaging
        self.min_confidence_threshold = 0.7  # Minimum confidence for detection
        self.static_scene_counter = 0  # Counter for static scenes
        self.is_static_scene = False
        self.static_scene_focus_point = None

    def detect_scenes(self, video_path):
        """Split video into scenes using PySceneDetect with higher threshold for stability"""
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        # Higher threshold means fewer scene changes, more stability
        scene_manager.add_detector(ContentDetector(threshold=40))

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()
        # Store scene boundaries for reference during processing
        self.shot_boundaries = [(scene[0].frame_num, scene[1].frame_num) for scene in scene_list]
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
                # Add confidence score to help with stability
                confidence = detection.score[0]
                faces.append((x, y, w, h, confidence))
        return faces

    def detect_people(self, frame):
        """Detect people using pose estimation"""
        results = self.mp_pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        people = []
        if results.pose_landmarks:
            h, w, _ = frame.shape
            # Get upper body landmarks for better stability
            landmarks = results.pose_landmarks.landmark

            # Check if key points are visible (nose, shoulders, hips)
            key_points = [0, 11, 12, 23, 24]  # Indices for nose, shoulders, and hips
            visible_confidence = sum(landmarks[i].visibility for i in key_points) / len(key_points)

            if visible_confidence > 0.7:  # Only track if confident
                # Calculate center of mass from upper body landmarks for stability
                upper_body_x = np.mean([landmarks[i].x for i in key_points if landmarks[i].visibility > 0.5]) * w
                upper_body_y = np.mean([landmarks[i].y for i in key_points if landmarks[i].visibility > 0.5]) * h

                people.append((int(upper_body_x), int(upper_body_y), visible_confidence))

        return people

    def detect_motion(self, prev_frame, curr_frame):
        """Detect significant motion between frames using optical flow"""
        if prev_frame is None:
            return None

        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate sparse optical flow using Lucas-Kanade method
        # This is more stable than dense optical flow
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        if prev_points is None or len(prev_points) == 0:
            return None

        # Use Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)

        if next_points is None:
            return None

        # Select good points
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        if len(good_new) == 0:
            return None

        # Calculate the movement vectors
        movements = good_new - good_old

        # Calculate magnitudes for each point
        magnitudes = np.sqrt(np.sum(movements**2, axis=1))

        # If no significant motion, return None
        if np.max(magnitudes) < 5:  # Threshold for significant motion
            return None

        # Find the point with maximum movement
        max_idx = np.argmax(magnitudes)
        motion_center = (int(good_new[max_idx][0][0]), int(good_new[max_idx][0][1]))
        
        return motion_center

    def _get_weighted_focus_point(self, frame, is_new_shot=False):
        """
        Calculate focus point using weighted detection results
        with temporal smoothing for stability
        """
        h, w, _ = frame.shape
        frame_center = (w // 2, h // 2)
        
        # Reset for new shots
        if is_new_shot:
            self.focus_point_history.clear()
            self.prev_focus_point = None
            self.is_static_scene = False
            self.static_scene_counter = 0
            self.static_scene_focus_point = None
            # Return frame center for first frame of a new shot
            return frame_center
        
        # Detect faces with higher priority
        faces = self.detect_faces(frame)
        faces = [f for f in faces if f[4] > self.min_confidence_threshold]  # Filter by confidence
        
        # If faces found, use them as primary focus
        if faces:
            # Use the largest/most confident face
            face_centers = [(f[0] + f[2]//2, f[1] + f[3]//2, f[4]) for f in faces]
            
            # Sort by confidence
            face_centers.sort(key=lambda x: x[2], reverse=True)
            
            # If multiple faces, use weighted average of top faces
            if len(face_centers) > 1:
                # Use top 2 faces only to avoid jumping between many faces
                top_faces = face_centers[:2]
                total_weight = sum(face[2] for face in top_faces)
                focus_x = sum(face[0] * face[2] for face in top_faces) / total_weight
                focus_y = sum(face[1] * face[2] for face in top_faces) / total_weight
                focus_point = (int(focus_x), int(focus_y))
            else:
                focus_point = (face_centers[0][0], face_centers[0][1])
                
            # Add to history with high confidence
            focus_confidence = 1.0
                
        else:
            # No faces, try pose detection
            people = self.detect_people(frame)
            
            if people:
                # Sort by confidence
                people.sort(key=lambda x: x[2], reverse=True)
                
                if len(people) > 1:
                    # Weighted average of top 2 people
                    top_people = people[:2]
                    total_weight = sum(person[2] for person in top_people)
                    focus_x = sum(person[0] * person[2] for person in top_people) / total_weight
                    focus_y = sum(person[1] * person[2] for person in top_people) / total_weight
                    focus_point = (int(focus_x), int(focus_y))
                else:
                    focus_point = (people[0][0], people[0][1])
                    
                # Add to history with medium confidence
                focus_confidence = 0.8
                
            else:
                # No people, try motion detection only if not already in static scene mode
                if not self.is_static_scene and self.prev_frame is not None:
                    motion_center = self.detect_motion(self.prev_frame, frame)
                    
                    if motion_center:
                        focus_point = motion_center
                        # Motion is less reliable, use lower confidence
                        focus_confidence = 0.6
                        # Reset static scene counter when motion detected
                        self.static_scene_counter = 0
                        self.is_static_scene = False
                    else:
                        # No motion detected, increment static counter
                        self.static_scene_counter += 1
                        
                        # If static for multiple frames, switch to static scene mode
                        if self.static_scene_counter > 30:  # ~1 second at 30fps
                            self.is_static_scene = True
                            
                            # Use scene analysis to find compositional focus point
                            if not self.static_scene_focus_point:
                                # Analyze frame for visual saliency
                                self.static_scene_focus_point = self.scene_analyzer.find_salient_region([frame])
                            
                            focus_point = self.static_scene_focus_point
                            focus_confidence = 0.7  # Medium-high confidence once established
                        elif self.prev_focus_point:
                            # Use previous focus point with decay
                            focus_point = self.prev_focus_point
                            focus_confidence = 0.5  # Lower confidence for continued use
                        else:
                            # Fallback to center
                            focus_point = frame_center
                            focus_confidence = 0.3  # Low confidence
                else:
                    # Already in static scene mode or no previous frame
                    if self.is_static_scene and self.static_scene_focus_point:
                        focus_point = self.static_scene_focus_point
                        focus_confidence = 0.7
                    elif self.prev_focus_point:
                        focus_point = self.prev_focus_point
                        focus_confidence = 0.5
                    else:
                        focus_point = frame_center
                        focus_confidence = 0.3
        
        # Apply temporal smoothing for stability
        if self.prev_focus_point and not is_new_shot:
            # Add confidence-weighted current detection to history
            self.focus_point_history.append((focus_point, focus_confidence))
            
            # Calculate temporally smoothed position using weighted average
            # Weight more recent frames higher for responsiveness
            if len(self.focus_point_history) >= 3:
                total_weight = 0
                weighted_x = 0
                weighted_y = 0
                
                # Apply exponential weighting to favor more recent points
                for i, (point, conf) in enumerate(self.focus_point_history):
                    # More recent points get exponentially higher weights
                    recency_weight = np.exp(i / 10)  # Exponential growth factor
                    weight = conf * recency_weight
                    weighted_x += point[0] * weight
                    weighted_y += point[1] * weight
                    total_weight += weight
                
                smoothed_x = int(weighted_x / total_weight)
                smoothed_y = int(weighted_y / total_weight)
                smoothed_focus = (smoothed_x, smoothed_y)
                
                # Additional hysteresis: Move only part way to new position
                if self.prev_focus_point:
                    # Calculate distance
                    dist = np.sqrt((smoothed_focus[0] - self.prev_focus_point[0])**2 + 
                                 (smoothed_focus[1] - self.prev_focus_point[1])**2)
                    
                    # Apply stronger smoothing for small movements
                    if dist < 20:  # Small movement
                        # Move only 30% of the way to new position for small changes
                        lerp_factor = 0.3
                    elif dist < 50:  # Medium movement
                        # Move 50% of the way
                        lerp_factor = 0.5
                    else:  # Large movement - likely intentional
                        # Move 70% of the way
                        lerp_factor = 0.7
                    
                    final_x = int(self.prev_focus_point[0] + lerp_factor * (smoothed_focus[0] - self.prev_focus_point[0]))
                    final_y = int(self.prev_focus_point[1] + lerp_factor * (smoothed_focus[1] - self.prev_focus_point[1]))
                    smoothed_focus = (final_x, final_y)
            else:
                smoothed_focus = focus_point
        else:
            smoothed_focus = focus_point
            self.focus_point_history.append((focus_point, focus_confidence))
        
        # Update previous focus point
        self.prev_focus_point = smoothed_focus
        return smoothed_focus

    def process_video(self, input_path, output_path, output_data_path=None):
        """
        Process video using optimized H2V algorithm with improved stability.
        
        Args:
            input_path: Path to input horizontal video
            output_path: Path to output vertical video
            output_data_path: Path to save focus point data (optional)
        """
        # Detect scenes first
        scenes = self.detect_scenes(input_path)
        print(f"Detected {len(scenes)} scenes")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output video
        output_height = height
        output_width = int(height * 9/16)  # 9:16 aspect ratio
        
        # Use lossless intermediate codec
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        temp_output_path = output_path + ".temp.avi"
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (output_width, output_height))
        
        # Track current scene
        current_scene_idx = 0
        current_frame_idx = 0
        buffer_frames = []  # For analyzing scene content before processing
        scene_buffer_size = min(30, int(fps))  # Buffer up to 1 second or 30 frames
        is_new_shot = True
        is_people_scene = True
        
        print("Processing video...")
        progress_bar = tqdm(total=total_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame_idx += 1
            progress_bar.update(1)
            
            # Check if we're entering a new scene
            if (current_scene_idx < len(scenes) and 
                current_frame_idx >= scenes[current_scene_idx][1].frame_num):
                current_scene_idx += 1
                is_new_shot = True
                buffer_frames = []
                
                # Reset stabilization
                self.virtual_camera = VirtualCamera()
                self.focus_point_history.clear()
                self.prev_focus_point = None
                self.is_static_scene = False
                
                # Print scene transition
                print(f"\nTransitioning to scene {current_scene_idx}")
                
            # Buffer frames for scene analysis
            if len(buffer_frames) < scene_buffer_size:
                buffer_frames.append(frame.copy())
                
                # If we have enough frames, analyze scene content
                if len(buffer_frames) == scene_buffer_size:
                    scene_analysis = self.scene_analyzer.analyze_scene_frames(buffer_frames)
                    is_people_scene = scene_analysis["content_type"] == "people"
                    
                    # Skip scenes without people if requested
                    if not is_people_scene:
                        print(f"Scene {current_scene_idx} has no people, marking for potential removal")
                        self.scenes_without_people.append(current_scene_idx)
            
            # Store previous frame for motion detection
            self.prev_frame = frame.copy()
            
            # Skip processing for scenes without people if desired
            # This is where you'd implement your second objective
            if not is_people_scene:
                # Option 1: Skip frame entirely (will make video shorter)
                # continue
                
                # Option 2: Use center crop for scenes without people
                focus_point = (width // 2, height // 2)
                is_new_shot = False
            else:
                # Get focus point with enhanced stability
                focus_point = self._get_weighted_focus_point(frame, is_new_shot)
            
            # Use virtual camera to smooth movement and reduce jitter
            camera_center = self.virtual_camera.update(focus_point, force_immediate=is_new_shot)
            
            # Apply smooth cropping
            cropped_frame, crop_data = self.smooth_cropper.apply_smooth_crop(
                frame, camera_center, force_update=is_new_shot, 
                content_type="people" if is_people_scene else "other", 
                new_shot=is_new_shot
            )
            
            # Reset new shot flag
            if is_new_shot:
                is_new_shot = False
                
            # Apply long-term stabilization occasionally during stable scenes
            if current_frame_idx % 30 == 0 and not is_new_shot:
                self.smooth_cropper.long_term_stabilization()
            
            # Resize to target output resolution if needed
            if cropped_frame.shape[1] != output_width or cropped_frame.shape[0] != output_height:
                cropped_frame = cv2.resize(cropped_frame, (output_width, output_height))
                
            # Store focus point data
            self.focus_points_data.append({
                "frame": current_frame_idx,
                "scene": current_scene_idx,
                "timestamp": current_frame_idx / fps,
                "focus_point": {"x": camera_center[0], "y": camera_center[1]},
                "crop": {
                    "left": crop_data["left"],
                    "right": crop_data["right"],
                    "top": crop_data["top"],
                    "bottom": crop_data["bottom"]
                }
            })
            
            # Write frame
            out.write(cropped_frame)
        
        # Release resources
        cap.release()
        out.release()
        progress_bar.close()
        
        # Convert to H.264 with appropriate bitrate
        #output_bitrate = "5M"  # Adjust as needed for quality
        #self._convert_to_mp4(temp_output_path, output_path, output_bitrate)
        
        # Combine the processed video with the original audio using FFmpeg
        self.combine_video_with_audio(input_path, temp_output_path, output_path)
        
        # Clean up temp file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            
        # Save focus point data if requested
        if output_data_path:
            with open(output_data_path, 'w') as f:
                json.dump({
                    "focus_points": self.focus_points_data,
                    "scenes_without_people": self.scenes_without_people
                }, f, indent=2)
                
        print(f"Processing complete. Output saved to {output_path}")
        if output_data_path:
            print(f"Focus point data saved to {output_data_path}")
            
        return output_data_path

    def _convert_to_mp4(self, input_path, output_path, bitrate="5M"):
        """Convert video to mp4 with specified bitrate using ffmpeg"""
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-preset", "slow", 
            "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
            "-pix_fmt", "yuv420p", "-y", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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


    def process_batch(self, input_dir, output_dir, threads=1):
        """Process multiple videos in batch mode with optional multithreading"""
        os.makedirs(output_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"Found {len(video_files)} videos to process")
        
        if threads > 1:
            # TODO: Implement parallel processing if needed
            pass
        else:
            for video_file in video_files:
                input_path = os.path.join(input_dir, video_file)
                output_name = f"{os.path.splitext(video_file)[0]}_vertical.mp4"
                output_path = os.path.join(output_dir, output_name)
                output_data_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_data.json")
                
                print(f"Processing {video_file}...")
                self.process_video(input_path, output_path, output_data_path)


# Example usage
if __name__ == "__main__":
    processor = ImprovedH2VProcessor()

    input_video = "input_video.mp4"  # Replace with your video path
    output_video = "output_vertical_video.mp4"
    # Export focus points data as JSON
    #tracking_data_path = os.path.splitext(output_video)[0] + "_tracking.json"
    tracking_data = 'output_vertical_video_tracking.json'
    focus_points_tracking_data = processor.process_video(input_video, output_video, tracking_data)
    #print(f"Processing complete. Vertical video saved to: {video_path}")
    print(f"Tracking data saved to: {focus_points_tracking_data }")
