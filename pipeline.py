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

class H2VProcessor:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_detection_model = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.mp_pose_model = self.mp_pose.Pose(min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)

        # Anti-flickering parameters
        self.focus_point_history = collections.deque(maxlen=30)  # Store more history for better smoothing
        self.crop_history = collections.deque(maxlen=30)  # Store crop region history
        self.prev_focus_point = None
        self.focus_points_data = []
        self.stability_weights = np.array([0.05, 0.1, 0.15, 0.2, 0.5])  # More weight on recent frames
        self.stability_weights = self.stability_weights / np.sum(self.stability_weights)  # Normalize

        # Shot detection state
        self.current_shot_id = 0
        self.shot_boundaries = []
        self.current_shot_stable_point = None

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

      # Calculate magnitudes for each point (fix the indexing issue)
      magnitudes = np.sqrt(np.sum(movements**2, axis=1))

      # If no significant motion, return None
      if np.max(magnitudes) < 5:  # Threshold for significant motion
        return None

      # Find the point with maximum movement
      max_idx = np.argmax(magnitudes)
      x, y = good_new[max_idx].ravel()  # Use ravel to flatten the point

      return (int(x), int(y))

    def is_new_shot(self, frame_idx):
        """Check if current frame is at a scene boundary"""
        for start, end in self.shot_boundaries:
            if frame_idx == start:
                self.current_shot_id += 1
                self.current_shot_stable_point = None
                return True
        return False

    def get_focus_point(self, frame, prev_frame=None, frame_idx=0, is_speaking=False):
        """Determine the main focus point in the frame with stability measures"""
        h, w, _ = frame.shape

        # Check if we're at a scene boundary
        if self.is_new_shot(frame_idx):
            # Reset tracking for new shot
            self.focus_point_history.clear()
            print(f"New shot detected at frame {frame_idx}")

        # Get candidate focus points with confidence scores
        candidates = []

        # Priority 1: Speaking person
        if is_speaking:
            faces = self.detect_faces(frame)
            if faces:
                # Sort by confidence * size
                faces.sort(key=lambda face: face[4] * (face[2] * face[3]), reverse=True)
                for x, y, fw, fh, conf in faces:
                    candidates.append(((x + fw // 2, y + fh // 2), conf * 1.5))  # Boost confidence for speakers

        # Priority 2: People in frame
        people = self.detect_people(frame)
        if people:
            for x, y, conf in people:
                candidates.append(((x, y), conf * 1.2))  # Slightly boost people detection

        # Priority 3: Motion detection (least reliable, lowest confidence)
        if len(candidates) == 0:  # Only use motion if no people detected
            motion_point = self.detect_motion(prev_frame, frame)
            if motion_point:
                candidates.append((motion_point, 0.5))  # Lower confidence for motion

        # If we have candidates, select the highest confidence one
        best_candidate = None
        best_confidence = 0

        for point, confidence in candidates:
            if confidence > best_confidence:
                best_candidate = point
                best_confidence = confidence

        # If no candidates found, use center of frame or previous point
        if best_candidate is None:
            if self.prev_focus_point:
                return self.prev_focus_point
            else:
                return (w // 2, h // 2)

        return best_candidate

    def adaptive_smooth_focus_point(self, new_point, frame_idx, shot_id):
        """
        Advanced smoothing algorithm with adaptive parameters based on content
        - More smoothing for static scenes
        - Less smoothing for fast-moving content
        - Scene-aware stabilization
        """
        # Initialize if this is the first point
        if self.prev_focus_point is None:
            self.prev_focus_point = new_point
            self.focus_point_history.append(new_point)
            return new_point

        # Add new point to history
        self.focus_point_history.append(new_point)

        # Different smoothing approach based on history length
        if len(self.focus_point_history) < 5:
            # Not enough history, use simple exponential smoothing
            alpha = 0.3  # Lower alpha = more smoothing
            smoothed_x = int(alpha * new_point[0] + (1 - alpha) * self.prev_focus_point[0])
            smoothed_y = int(alpha * new_point[1] + (1 - alpha) * self.prev_focus_point[1])
        else:
            # Calculate motion variance to determine how much smoothing to apply
            recent_points = list(self.focus_point_history)[-5:]
            x_coords = [p[0] for p in recent_points]
            y_coords = [p[1] for p in recent_points]

            x_variance = np.var(x_coords)
            y_variance = np.var(y_coords)

            # Adaptive smoothing - more stable for low variance, more responsive for high variance
            if x_variance < 100 and y_variance < 100:
                # Low movement - strong smoothing
                weights = np.array([0.1, 0.1, 0.2, 0.2, 0.4])  # More weight on recent frames
            elif x_variance > 1000 or y_variance > 1000:
                # High movement - minimal smoothing
                weights = np.array([0.05, 0.05, 0.1, 0.3, 0.5])  # Much more weight on current frame
            else:
                # Medium movement - moderate smoothing
                weights = np.array([0.05, 0.1, 0.15, 0.3, 0.4])

            # Normalize weights
            weights = weights / np.sum(weights)

            # Apply weighted average to recent points
            smoothed_x = int(np.sum([p[0] * w for p, w in zip(recent_points, weights)]))
            smoothed_y = int(np.sum([p[1] * w for p, w in zip(recent_points, weights)]))

        # Determine if we should "lock" onto a stable subject
        # For example, in static shots with minimal movement
        if len(self.focus_point_history) >= 10:
            recent_x = [p[0] for p in list(self.focus_point_history)[-10:]]
            recent_y = [p[1] for p in list(self.focus_point_history)[-10:]]

            # If movement is very small over multiple frames, lock onto the average position
            if np.std(recent_x) < 20 and np.std(recent_y) < 20:
                if self.current_shot_stable_point is None:
                    # We found a stable point, lock onto it for this shot
                    self.current_shot_stable_point = (int(np.mean(recent_x)), int(np.mean(recent_y)))
                    print(f"Locking onto stable point {self.current_shot_stable_point} for shot {shot_id}")

                # Use the stable point with slight adjustment towards current point
                if self.current_shot_stable_point:
                    smoothed_x = int(0.9 * self.current_shot_stable_point[0] + 0.1 * smoothed_x)
                    smoothed_y = int(0.9 * self.current_shot_stable_point[1] + 0.1 * smoothed_y)

        # Update previous focus point for next iteration
        self.prev_focus_point = (smoothed_x, smoothed_y)
        return self.prev_focus_point

    def crop_with_stability(self, frame, focus_point, aspect_ratio=9/16):
        """Crop the frame with temporal stability to prevent border flickering"""
        h, w = frame.shape[:2]

        # Calculate crop dimensions for target aspect ratio
        target_width = int(h * aspect_ratio)

        # Center the crop on the focus point
        x_center, y_center = focus_point

        # Calculate initial crop boundaries
        left = max(0, x_center - target_width // 2)
        right = min(w, left + target_width)

        # Adjust if we hit the edge of the frame
        if right == w:
            left = max(0, w - target_width)
        if left == 0:
            right = min(w, target_width)

        # Add current crop to history
        current_crop = (left, right)
        self.crop_history.append(current_crop)

        # Stabilize crop boundaries if we have history
        if len(self.crop_history) >= 5:
            # Get recent crop boundaries
            recent_lefts = [c[0] for c in list(self.crop_history)[-5:]]
            recent_rights = [c[1] for c in list(self.crop_history)[-5:]]

            # Smooth the boundaries with weighted average
            # More weight on recent frames but not too aggressive
            weights = [0.1, 0.15, 0.2, 0.25, 0.3]
            smooth_left = int(sum(l * w for l, w in zip(recent_lefts, weights)))
            smooth_right = int(sum(r * w for r, w in zip(recent_rights, weights)))

            # Ensure we maintain the correct width
            if smooth_right - smooth_left != target_width:
                center = (smooth_left + smooth_right) // 2
                smooth_left = max(0, center - target_width // 2)
                smooth_right = min(w, smooth_left + target_width)

                # Adjust again if needed
                if smooth_right == w:
                    smooth_left = max(0, w - target_width)
                if smooth_left == 0:
                    smooth_right = min(w, target_width)

            left, right = smooth_left, smooth_right

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
        """Process the whole video with anti-flickering techniques"""
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

        # Detect scenes and store scene boundaries
        print("Detecting scene boundaries...")
        scenes = self.detect_scenes(input_path)
        if not scenes:
            # If no scenes detected, treat the whole video as one scene
            scenes = [(0, total_frames)]
            self.shot_boundaries = [(0, total_frames)]

        print(f"Detected {len(scenes)} scenes in the video")

        # Output dimensions for 9:16 aspect ratio based on original height
        out_width = int(height * 9 / 16)

        # Reset tracking for new video
        self.current_shot_id = 0
        self.prev_focus_point = None
        self.focus_point_history.clear()
        self.crop_history.clear()
        self.focus_points_data = []

        # Create temporary video path for frames only
        temp_video_path = os.path.splitext(output_path)[0] + "_temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (out_width, height))

        # Process frames
        prev_frame = None
        frame_count = 0

        with tqdm(total=total_frames, desc="Processing frames") as progress_bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate current time in seconds
                current_time = frame_count / original_fps

                # Get focus point
                focus_point = self.get_focus_point(frame, prev_frame, frame_count)

                # Apply advanced smoothing
                smooth_point = self.adaptive_smooth_focus_point(focus_point, frame_count, self.current_shot_id)

                # Apply stabilized cropping
                cropped_frame, focus_data = self.crop_with_stability(frame, smooth_point)

                # Add timestamp to focus data
                focus_data["time"] = current_time
                focus_data["frame"] = frame_count
                focus_data["shot_id"] = self.current_shot_id
                self.focus_points_data.append(focus_data)

                # Add visual indicator of focus point (for debugging)
                #cv2.circle(frame, smooth_point, 10, (0, 255, 0), -1)
                #cv2.rectangle(frame, (focus_data["crop"]["left"], 0),
                #             (focus_data["crop"]["right"], height), (0, 0, 255), 2)

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