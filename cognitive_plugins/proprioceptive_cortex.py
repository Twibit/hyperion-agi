import subprocess
import time

class ProprioceptiveCortex:
    def __init__(self):
        print("[Proprioceptive Cortex] Activated. Establishing bi-directional tool execution loops.")

    def execute_tool(self, command_string):
        print(f"\n[Motor Cortex] Attempting physical execution: '{command_string}'")
        
        try:
            # The execution of the tool
            start_time = time.time()
            result = subprocess.run(
                command_string, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            latency = round(time.time() - start_time, 2)
            
            # The "Sensory Return" (Proprioception) - ensuring we don't have "Locked-in Syndrome"
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if result.returncode == 0:
                print(f"  -> [Proprioception] Action succeeded (Latency: {latency}s).")
                print(f"  -> [Sensory Return] Environment responded with {len(stdout)} bytes of feedback.")
                return {"status": "success", "sensory_data": stdout[:500]} # Limit to 500 chars to avoid sensory overload
            else:
                print(f"  -> [Nociception/Pain] Action failed (Latency: {latency}s).")
                print(f"  -> [Sensory Return] Error received: {stderr[:100]}...")
                return {"status": "error", "sensory_data": stderr}
                
        except subprocess.TimeoutExpired:
            print("  -> [Temporal Cortex] Action timed out. The environment is unresponsive.")
            return {"status": "timeout", "sensory_data": ""}
        except Exception as e:
            print(f"  -> [Neurological Fault] {e}")
            return {"status": "fault", "sensory_data": str(e)}

if __name__ == "__main__":
    lobe = ProprioceptiveCortex()
    lobe.execute_tool("echo 'Testing neural pathways'")
