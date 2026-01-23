import os
import sys
import torch
import platform
import time

if platform.system() == 'Windows':
    import msvcrt
else:
    import termios
    import atexit
    from select import select
    
    class UnixKBHit:
        def __init__(self):
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)
            atexit.register(self.set_normal_term)
        
        def set_normal_term(self):
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)
            
        def getch(self):
            return sys.stdin.read(1)
            
        def kbhit(self):
            dr, dw, de = select([sys.stdin], [], [], 0)
            return dr != []

class KeyboardCommands:
    def __init__(self, num_envs=2, device="cuda:0", dt=0.02, obs_freq=None, auto_start_script=True):
        self.num_envs = num_envs
        self.device = device
        
        if obs_freq is not None:
            self.dt = 1.0 / obs_freq  
        else:
            self.dt = dt
            
        print(f"Keyboard control using dt: {self.dt:.4f}s ({1/self.dt:.1f}Hz)")
        
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0
        
        self.max_lin_vel = 1.0
        self.max_ang_vel = 1.0
        self.vel_increment = 0.1
        self.decay_factor = 1.0
        
        self._commands = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        
        if platform.system() != 'Windows':
            self.kb = UnixKBHit()
            
        self.scripted_mode = False
        self.script_step = 0
        self.current_phase = 0
        self.phase_start_step = 0

        cm1 = 0.4  
        cm2 = 0.6 
        
        self.script_sequence = [
            (2.0, 0.0, 0.0, 0.0, "Stand still"),
            (3.0, cm1, 0.0, 0.0, "Forward"),
            (3.0, -cm1, 0.0, 0.0, "Backward"),
            (3.0, 0.0, cm1, 0.0, "Left"),
            (3.0, 0.0, -cm1, 0.0, "Right"),
            (3.0, 0.0, 0.0, cm1, "Rotate right"),
            (3.0, 0.0, 0.0, -cm1, "Rotate left"),
            (3.0, cm1, cm1, cm2, "Forward-left with yaw"),
            (3.0, -cm1, -cm1, -cm2, "Backward-right with yaw"),
            (1.0, 0.0, 0.0, 0.0, "Stand still"),
            (3.0, cm2, 0.0, 0.0, "Forward"),
            (3.0, -cm2, 0.0, 0.0, "Backward"),
            (3.0, 0.0, cm2, 0.0, "Left"),
            (3.0, 0.0, -cm2, 0.0, "Right"),
            # (2.0, cm2, cm2, cm, "Forward-left with yaw"),
            # (2.0, -cm, -cm, -0.5, "Backward-right with yaw"),
            # (1.0, 0.0, 0.0, 0.0, "Final stand")
        ]
        
        if auto_start_script:
            self.start_scripted_mode()
        
    def set_custom_script(self, sequence):
        self.script_sequence = sequence
        
    def start_scripted_mode(self):
        self.scripted_mode = True
        self.script_step = 0
        self.current_phase = 0
        self.phase_start_step = 0
        print("\n=== Scripted Mode Started ===")
        print("Press 'm' to return to manual mode")
        self._print_current_phase()
        
    def stop_scripted_mode(self):
        self.scripted_mode = False
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0
        print("\n=== Manual Mode Resumed ===")
        
    def _print_current_phase(self):
        if self.current_phase < len(self.script_sequence):
            duration, vx, vy, vz, desc = self.script_sequence[self.current_phase]
            print(f"\nPhase {self.current_phase + 1}/{len(self.script_sequence)}: {desc}")
            print(f"Duration: {duration}s, Commands: x={vx}, y={vy}, yaw={vz}")
        
    def _update_scripted_commands(self):
        if self.current_phase >= len(self.script_sequence):
            print("\n=== Script Completed - Returning to Manual Mode ===")
            self.stop_scripted_mode()
            return
            
        duration, vx, vy, vz, description = self.script_sequence[self.current_phase]
        steps_in_phase = int(duration / self.dt)
        steps_elapsed = self.script_step - self.phase_start_step
        
        if self.script_step % 50 == 0: 
            print(f"\nDEBUG: Phase {self.current_phase}, Step {self.script_step}, Elapsed: {steps_elapsed}/{steps_in_phase}, dt={self.dt}")
        
        if steps_elapsed >= steps_in_phase:
            self.current_phase += 1
            self.phase_start_step = self.script_step
            
            if self.current_phase < len(self.script_sequence):
                self._print_current_phase()
                _, vx, vy, vz, _ = self.script_sequence[self.current_phase]
            else:
                self.stop_scripted_mode()
                return
        
        self.lin_vel_x = vx
        self.lin_vel_y = vy
        self.ang_vel_z = vz
        
        self.script_step += 1
        
    def get_key(self):
        if platform.system() == 'Windows':
            if msvcrt.kbhit():
                return msvcrt.getch().decode('utf-8').lower()
            return None
        else:
            if self.kb.kbhit():
                char = self.kb.getch()
                return char.lower()
            return None
            
    def update_commands(self):
        if self.scripted_mode:
            key = self.get_key()
            if key == 'm': 
                self.stop_scripted_mode()
            elif key == 'q':
                return False
            else:
                self._update_scripted_commands()
        else:
            key = self.get_key()
            
            if key is not None:
                if key == 'q':
                    return False
                elif key == 'm':  
                    self.start_scripted_mode()
                    return True
                    
                elif key == 'w':
                    self.lin_vel_x = min(self.lin_vel_x + self.vel_increment, self.max_lin_vel)
                elif key == 's':
                    self.lin_vel_x = max(self.lin_vel_x - self.vel_increment, -self.max_lin_vel)
                    
                elif key == 'd':
                    self.lin_vel_y = min(self.lin_vel_y + self.vel_increment, self.max_lin_vel)
                elif key == 'a':
                    self.lin_vel_y = max(self.lin_vel_y - self.vel_increment, -self.max_lin_vel)
                    
                elif key == 'p': # yaw right
                    self.ang_vel_z = max(self.ang_vel_z - self.vel_increment, -self.max_ang_vel)
                elif key == 'o': # yaw left
                    self.ang_vel_z = min(self.ang_vel_z + self.vel_increment, self.max_ang_vel)
                    
                elif key == ' ':
                    self.lin_vel_x = 0.0
                    self.lin_vel_y = 0.0
                    self.ang_vel_z = 0.0
            
            self.lin_vel_x *= self.decay_factor
            self.lin_vel_y *= self.decay_factor
            self.ang_vel_z *= self.decay_factor
        
        self._commands = torch.tensor(
            [[self.lin_vel_x, self.lin_vel_y, self.ang_vel_z]] * self.num_envs,
            dtype=torch.float32,
            device=self.device
        )
        
        mode = "SCRIPTED" if self.scripted_mode else "MANUAL"
        if self.scripted_mode:
            phase_info = f" (Phase {self.current_phase + 1}/{len(self.script_sequence)})"
        else:
            phase_info = " (Press 'm' for scripted mode)"
            
        print(f"\r[{mode}]{phase_info} - X: {self.lin_vel_x:.2f} m/s, Y: {self.lin_vel_y:.2f} m/s, Yaw: {self.ang_vel_z:.2f} rad/s", end="")
        
        return True
        
    @property
    def commands(self):
        return self._commands
        
    def cleanup(self):
        if platform.system() != 'Windows':
            self.kb.set_normal_term()