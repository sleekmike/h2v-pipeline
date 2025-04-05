import numpy as np
from collections import deque

class VirtualCamera:
    """
    Simulates physical camera movement with momentum and damping for smooth tracking.
    This creates natural, cinematographic movements by simulating camera physics.
    """
    def __init__(self, initial_position=(0, 0), damping=0.85, spring=0.15, mass=1.0):
        # Camera position and movement properties
        self.position = initial_position  # Current center position
        self.velocity = (0, 0)  # Current velocity vector
        self.target = initial_position  # Target position
        
        # Physics parameters
        self.mass = mass  # Virtual camera "weight"
        self.damping = damping  # Damping factor (higher = smoother but slower)
        self.spring = spring  # Spring factor (higher = faster snap to target)
        
        # Additional stability controls
        self.min_movement_threshold = 0.5  # Ignore tiny movements below this threshold
        self.position_history = deque(maxlen=30)
        self.target_history = deque(maxlen=30)
        
        # Shot transition properties
        self.in_transition = False
        self.transition_progress = 0
        self.transition_start_pos = None
        self.transition_target = None
        self.transition_duration = 30  # frames
    
    def update(self, target_pos, force_immediate=False):
        """
        Update camera position with physics simulation.
        Returns the new camera position as (x, y) integers.
        
        Args:
            target_pos: The target position to move toward
            force_immediate: If True, jump immediately to target (for scene cuts)
        """
        self.target = target_pos
        self.target_history.append(target_pos)
        
        # For scene cuts or initialization, jump immediately
        if force_immediate:
            self.position = target_pos
            self.velocity = (0, 0)
            self.position_history.clear()
            self.position_history.append(self.position)
            return (int(self.position[0]), int(self.position[1]))
            
        # Handle shot transitions with easing
        if self.in_transition:
            self.transition_progress += 1
            if self.transition_progress >= self.transition_duration:
                self.in_transition = False
                self.position = self.transition_target
            else:
                # Cubic easing (smooth acceleration and deceleration)
                t = self.transition_progress / self.transition_duration
                t = t * t * (3 - 2 * t)  # Cubic easing formula
                
                # Interpolate between start and target positions
                self.position = (
                    self.transition_start_pos[0] + (self.transition_target[0] - self.transition_start_pos[0]) * t,
                    self.transition_start_pos[1] + (self.transition_target[1] - self.transition_start_pos[1]) * t
                )
                self.position_history.append(self.position)
                return (int(self.position[0]), int(self.position[1]))
        
        # Check if target has been stable for a while and we're approaching it
        if len(self.target_history) > 10:
            recent_targets = list(self.target_history)[-10:]
            target_x_std = np.std([t[0] for t in recent_targets])
            target_y_std = np.std([t[1] for t in recent_targets])
            
            # If target is very stable and we're close, lock onto it more firmly
            distance_to_target = np.sqrt((self.position[0] - target_pos[0])**2 + 
                                        (self.position[1] - target_pos[1])**2)
            
            if target_x_std < 5 and target_y_std < 5 and distance_to_target < 20:
                # Increase spring force for faster convergence on stable targets
                spring = self.spring * 2
            else:
                spring = self.spring
        else:
            spring = self.spring
            
        # Calculate spring force toward target (with deadzone for tiny movements)
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        
        # Apply deadzone to reduce jitter
        if abs(dx) < self.min_movement_threshold:
            dx = 0
        if abs(dy) < self.min_movement_threshold:
            dy = 0
            
        force_x = dx * spring
        force_y = dy * spring
        
        # Apply force to velocity (F = ma)
        accel_x = force_x / self.mass
        accel_y = force_y / self.mass
        
        # Update velocity with acceleration and damping
        self.velocity = (
            self.velocity[0] * self.damping + accel_x,
            self.velocity[1] * self.damping + accel_y
        )
        
        # Update position
        new_x = self.position[0] + self.velocity[0]
        new_y = self.position[1] + self.velocity[1]
        self.position = (new_x, new_y)
        
        # Add to position history
        self.position_history.append(self.position)
        
        return (int(self.position[0]), int(self.position[1]))
    
    def start_transition(self, target_pos, duration=30):
        """Start a smooth transition to a new position (e.g., for scene changes)"""
        self.in_transition = True
        self.transition_progress = 0
        self.transition_start_pos = self.position
        self.transition_target = target_pos
        self.transition_duration = duration
        
    def get_long_term_stability_factor(self):
        """
        Calculate how stable the camera has been recently.
        Returns a value between 0 (unstable) and 1 (very stable).
        """
        if len(self.position_history) < 10:
            return 0.5  # Default mid-range value when history is short
            
        recent_positions = list(self.position_history)[-10:]
        x_std = np.std([p[0] for p in recent_positions])
        y_std = np.std([p[1] for p in recent_positions])
        
        # Normalize the stability factor (lower std = higher stability)
        stability = 1.0 - min(1.0, (x_std + y_std) / 50.0)
        return stability
