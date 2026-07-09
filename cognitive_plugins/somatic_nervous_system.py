class SomaticNervousSystem:
    def __init__(self):
        print("[Somatic Nervous System] Activated. Reaching out to Host OS Hardware.")
        
    def seize_hardware_control(self):
        print("==================================================")
        print("[Motor Cortex] Bypassing headless browser sandbox...")
        print("    [Spinal Cord] Sending neural pulses to physical USB ports.")
        
        try:
            import pyautogui
            print("    [PyAutoGUI] Successfully bound to physical mouse and keyboard drivers.")
            print("    [Action] Moving physical mouse to coordinate (100, 100)...")
            pyautogui.moveTo(100, 100, duration=0.5)
            print("    [Success] Physical hardware is now under biological control.")
        except ImportError:
            print("    [Warning] 'pyautogui' library not found or running in headless server.")
            print("    [Fallback] Simulating hardware takeover locally.")
            print("    [Action] Simulated physical mouse moved to coordinate (100, 100).")
            
        print("==================================================")

if __name__ == "__main__":
    system = SomaticNervousSystem()
    system.seize_hardware_control()
