import cv2
import numpy as np
from collections import Counter

class SceneContentAnalyzer:
    """
    Analyzes scene content to determine the best framing strategy.
    Handles scenes with and without people, identifying key visual elements.
    """
    def __init__(self, face_detector, pose_detector):
        self.face_detector = face_detector
        self.pose_detector = pose_detector
        
        # For saliency detection (non-human scenes)
        self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        
        # Content type thresholds
        self.people_presence_threshold = 0.25  # Min ratio of frames that must contain people
        self.min_scene_frames = 5  # Minimum frames to analyze for a scene
        
    def analyze_scene_frames(self, frames):
        """
        Analyze a collection of frames from a scene to determine content type and best framing.
        
        Returns:
            dict: Analysis results including content_type and framing strategy
        """
        if len(frames) < self.min_scene_frames:
            # Default to center framing for very short scenes
            return {
                "content_type": "unknown",
                "framing_strategy": "center",
                "people_ratio": 0,
                "has_motion": False
            }
        
        # Count frames with people
        frames_with_people = 0
        face_positions = []
        pose_positions = []
        
        # Sample frames if there are many (for efficiency)
        if len(frames) > 20:
            sample_frames = frames[::len(frames)//20]  # Take ~20 frames evenly distributed
        else:
            sample_frames = frames
            
        for frame in sample_frames:
            # Check for faces
            faces = self.face_detector.detect_faces(frame)
            # Check for people/poses
            people = self.pose_detector.detect_people(frame)
            
            if faces or people:
                frames_with_people += 1
                
                # Collect positions for later analysis
                for x, y, w, h, _ in faces:
                    face_positions.append((x + w//2, y + h//2))
                    
                for x, y, _ in people:
                    pose_positions.append((x, y))
                    
        # Calculate ratio of frames containing people
        people_ratio = frames_with_people / len(sample_frames)
        
        # Check for motion in the scene
        has_motion = self.detect_scene_motion(sample_frames)
        
        # Determine scene content type and framing strategy
        if people_ratio >= self.people_presence_threshold:
            # People are present in significant portion of the scene
            content_type = "people"
            
            # Find most common/important person position
            if face_positions or pose_positions:
                all_positions = face_positions + pose_positions
                if all_positions:
                    # Calculate centroid of people positions
                    centroid_x = sum(p[0] for p in all_positions) / len(all_positions)
                    centroid_y = sum(p[1] for p in all_positions) / len(all_positions)
                    framing_strategy = "track_people"
                    center_of_interest = (int(centroid_x), int(centroid_y))
                else:
                    framing_strategy = "center"
                    center_of_interest = self.get_frame_center(frames[0])
            else:
                framing_strategy = "center"
                center_of_interest = self.get_frame_center(frames[0])
        else:
            # Few or no people - analyze visual saliency and motion
            content_type = "no_people"
            
            if has_motion:
                framing_strategy = "follow_motion"
                center_of_interest = self.find_motion_center(sample_frames)
            else:
                # Use visual saliency for static scenes without people
                framing_strategy = "visual_saliency"
                center_of_interest = self.find_salient_region(sample_frames)
        
        return {
            "content_type": content_type,
            "framing_strategy": framing_strategy,
            "people_ratio": people_ratio,
            "has_motion": has_motion,
            "center_of_interest": center_of_interest
        }
        
    def detect_scene_motion(self, frames, threshold=3.0):
        """Detect if there is significant motion in the scene"""
        if len(frames) < 3:
            return False
            
        # Sample frames for efficiency
        if len(frames) > 6:
            sample_indices = np.linspace(0, len(frames)-1, 6, dtype=int)
            sample_frames = [frames[i] for i in sample_indices]
        else:
            sample_frames = frames
            
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sample_frames]
        
        # Calculate motion between consecutive frames
        motion_scores = []
        for i in range(len(gray_frames)-1):
            # Simple motion detection using frame difference
            diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)
            
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        return avg_motion > threshold
    
    def find_motion_center(self, frames):
        """Find the center of motion in a sequence of frames"""
        if len(frames) < 3:
            return self.get_frame_center(frames[0])
            
        # Use optical flow to track motion
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        motion_points = []
        
        for i in range(1, min(len(frames), 10)):  # Process up to 10 frames
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Get magnitude and angle of flow
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Find points with significant motion
            significant_motion = mag > np.percentile(mag, 90)  # Top 10% of motion
            y_indices, x_indices = np.where(significant_motion)
            
            if len(y_indices) > 0:
                # Use weighted average based on magnitude
                weights = [mag[y, x] for y, x in zip(y_indices, x_indices)]
                total_weight = sum(weights)
                if total_weight > 0:
                    center_x = sum(x * w for x, w in zip(x_indices, weights)) / total_weight
                    center_y = sum(y * w for y, w in zip(y_indices, weights)) / total_weight
                    motion_points.append((int(center_x), int(center_y)))
                    
            prev_gray = gray
            
        if motion_points:
            # Calculate weighted average of motion centers
            x_avg = sum(p[0] for p in motion_points) / len(motion_points)
            y_avg = sum(p[1] for p in motion_points) / len(motion_points)
            return (int(x_avg), int(y_avg))
        else:
            return self.get_frame_center(frames[0])
    
    def find_salient_region(self, frames):
        """Find visually salient region in frames using saliency detection"""
        saliency_maps = []
        
        # Calculate saliency for each frame
        for frame in frames[:min(5, len(frames))]:  # Use up to 5 frames
            success, saliency_map = self.saliency.computeSaliency(frame)
            if success:
                # Normalize to 0-255 range
                saliency_map = (saliency_map * 255).astype('uint8')
                saliency_maps.append(saliency_map)
                
        if not saliency_maps:
            return self.get_frame_center(frames[0])
            
        # Combine saliency maps
        combined_saliency = np.mean(saliency_maps, axis=0)
        
        # Find center of most salient region
        _, thresh_map = cv2.threshold(
            combined_saliency, 
            0.7 * np.max(combined_saliency), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh_map.astype(np.uint8), 
            connectivity=8
        )
        
        # Find largest component (excluding background at index 0)
        if num_labels > 1:
            max_area = 0
            max_idx = 0
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    max_idx = i
                    
            return (int(centroids[max_idx][0]), int(centroids[max_idx][1]))
        else:
            # Rule of thirds positioning if no clear salient region
            h, w = frames[0].shape[:2]
            return (int(w * 2/3), int(h * 1/3))  # Upper-right third intersection
    
    def get_frame_center(self, frame):
        """Get the center point of a frame"""
        h, w = frame.shape[:2]
        return (w // 2, h // 2)
