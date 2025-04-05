import numpy as np
from collections import deque

class SmoothCropper:
    """
    Creates stable, smooth crop windows with cinematic movement.
    Handles different aspect ratios while maintaining visual stability.
    """
    def __init__(self, target_aspect_ratio=9/16, transition_frames=30):
        self.target_aspect_ratio = target_aspect_ratio
        self.transition_frames = transition_frames
        
        # Crop window history
        self.crop_history = deque(maxlen=60)  # Store up to 2 seconds at 30fps
        
        # Current crop state
        self.current_crop = None
        self.target_crop = None
        
        # Transition state
        self.in_transition = False
        self.transition_progress = 0
        self.transition_start_crop = None
        self.transition_end_crop = None
        
        # Anti-jitter settings
        self.min_movement_threshold = 5  # Pixels
        self.hysteresis_ratio = 0.2  # Move only if change is significant
        
        # Content-aware framing options
        self.headroom_ratio = 0.1  # Extra space above detected faces
    
    def calculate_crop(self, frame, focus_point, content_type="people"):
        """
        Calculate the optimal crop window centered on the focus point.
        
        Args:
            frame: The video frame
            focus_point: (x, y) center point to focus on
            content_type: Type of content ("people", "motion", etc.)
            
        Returns:
            dict: Crop parameters (left, right, top, bottom)
        """
        h, w = frame.shape[:2]
        
        # Calculate target width for given aspect ratio
        target_width = int(h * self.target_aspect_ratio)
        if target_width > w:
            # Handle case where target width exceeds frame width
            target_width = w
            
        # Adjust focus point based on content type
        adjusted_focus_x, adjusted_focus_y = focus_point
        
        if content_type == "people":
            # Add headroom for people (shift focus point down slightly)
            headroom = int(h * self.headroom_ratio)
            adjusted_focus_y = min(h - 1, adjusted_focus_y + headroom)
        
        # Calculate initial crop boundaries centered on adjusted focus point
        half_width = target_width // 2
        left = adjusted_focus_x - half_width
        right = adjusted_focus_x + half_width
        
        # Adjust if crop goes out of bounds
        if left < 0:
            left = 0
            right = target_width
        elif right > w:
            right = w
            left = max(0, w - target_width)
            
        # Ensure exact width needed for aspect ratio
        if right - left != target_width:
            right = min(w, left + target_width)
            
        crop = {
            "left": int(left),
            "right": int(right),
            "top": 0,
            "bottom": h
        }
        
        return crop
    
    def apply_smooth_crop(self, frame, focus_point, force_update=False, 
                         content_type="people", new_shot=False):
        """
        Apply a smoothed crop to the frame based on focus point.
        
        Args:
            frame: The video frame
            focus_point: (x, y) center point to focus on
            force_update: Force update even if movement is small
            content_type: Type of content for content-aware framing
            new_shot: True if this is the start of a new shot
            
        Returns:
            tuple: (cropped_frame, crop_data)
        """
        h, w = frame.shape[:2]
        
        # Calculate target crop based on current focus point
        target_crop = self.calculate_crop(frame, focus_point, content_type)
        
        # For first frame or new shot
        if self.current_crop is None or new_shot:
            self.current_crop = target_crop
            self.crop_history.clear()
            self.crop_history.append(self.current_crop)
            
            # Apply crop
            cropped = frame[:, target_crop["left"]:target_crop["right"]]
            return cropped, target_crop
            
        # Determine if we need to transition to a new crop
        if self.in_transition:
            # Continue existing transition
            self.transition_progress += 1
            progress_ratio = self.transition_progress / self.transition_frames
            
            if self.transition_progress >= self.transition_frames:
                # Transition complete
                self.in_transition = False
                self.current_crop = self.transition_end_crop
            else:
                # Apply easing function for smooth transition
                t = progress_ratio
                # Cubic easing
                t = t * t * (3 - 2 * t)
                
                # Interpolate crop values
                left = int(self.transition_start_crop["left"] + 
                          (self.transition_end_crop["left"] - self.transition_start_crop["left"]) * t)
                right = int(self.transition_start_crop["right"] + 
                           (self.transition_end_crop["right"] - self.transition_start_crop["right"]) * t)
                
                self.current_crop = {
                    "left": left,
                    "right": right,
                    "top": 0,
                    "bottom": h
                }
        else:
            # Check if the target crop is significantly different from current crop
            current_center = (self.current_crop["left"] + self.current_crop["right"]) // 2
            target_center = (target_crop["left"] + target_crop["right"]) // 2
            
            movement = abs(current_center - target_center)
            
            # Determine if we should start a new transition
            if force_update or movement > self.min_movement_threshold:
                significant_change = movement > self.hysteresis_ratio * (target_crop["right"] - target_crop["left"])
                
                if significant_change or force_update:
                    # Start new transition
                    self.in_transition = True
                    self.transition_progress = 0
                    self.transition_start_crop = self.current_crop.copy()
                    self.transition_end_crop = target_crop.copy()
                    
        # Add current crop to history
        self.crop_history.append(self.current_crop)
        
        # Create cropped frame
        cropped = frame[:, self.current_crop["left"]:self.current_crop["right"]]
        
        return cropped, self.current_crop
        
    def long_term_stabilization(self):
        """
        Apply long-term stabilization by analyzing the history of crop windows.
        Helps reduce slow drift and maintain more stable framing.
        
        Should be called periodically during stable scenes.
        """
        if len(self.crop_history) < 15:
            return
            
        # Extract centers of recent crops
        recent_crops = list(self.crop_history)[-15:]
        centers = [(c["left"] + c["right"]) // 2 for c in recent_crops]
        
        # Check if centers are relatively stable
        center_std = np.std(centers)
        
        if center_std < 10:  # Very stable
            # Calculate average center position
            avg_center = int(np.mean(centers))
            crop_width = self.current_crop["right"] - self.current_crop["left"]
            
            # Create a stabilized crop around this center
            stable_left = avg_center - crop_width // 2
            stable_right = avg_center + crop_width // 2
            
            # Apply small correction toward stable position
            current_center = (self.current_crop["left"] + self.current_crop["right"]) // 2
            diff = avg_center - current_center
            
            if abs(diff) > 3:  # Only correct if difference is noticeable
                correction = int(diff * 0.2)  # Apply 20% correction
                self.current_crop["left"] += correction
                self.current_crop["right"] += correction
