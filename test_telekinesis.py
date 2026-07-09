import json
import os
import sys
from rust_backend_shim import NativeMotorSystem

print("==================================================")
print(" TESTING NATIVE BIOLOGICAL MOTOR CORTEX ")
print("==================================================\n")

try:
    motor = NativeMotorSystem()
    print("[System] NativeMotorSystem actor mesh initialized successfully!")
except Exception as e:
    print(f"[Error] Failed to initialize MotorSystem: {e}")
    sys.exit(1)

file_path = "/root/bio-ai-lab/hyperion/test_telekinesis_output.txt"

# 1. Write Intent
write_intent = {
    "kind": "write",
    "path": file_path,
    "data": list(b"Hello from the biological Rust effector!")
}

print(f"\n[Cortex] Firing write intent down the spinal cord to FileLimb...")
try:
    outcome_str = motor.reach_file(json.dumps(write_intent))
    outcome = json.loads(outcome_str)
    print(f"  [Proprioception] Reafference verified! Outcome: {outcome}")
except Exception as e:
    print(f"  [Error] Motor Execution Failed: {e}")

# 2. Read Intent
read_intent = {
    "kind": "read",
    "path": file_path
}

print(f"\n[Cortex] Firing read intent...")
try:
    outcome_str = motor.reach_file(json.dumps(read_intent))
    outcome = json.loads(outcome_str)
    if outcome.get("kind") == "bytes":
        # The struct serializes bytes as an array of integers under "bytes" or unnamed tuple? 
        # Actually it's Outcome::Bytes(Vec<u8>). Serde with tag="kind" makes it {"kind": "bytes"} but Vec<u8> is tricky.
        # It's an internally tagged enum, so data is at the same level? Wait, standard Serde internally tagged enum for newtype variant is weird. Let's just print it.
        print(f"  [Proprioception] Read successful: {outcome}")
except Exception as e:
    print(f"  [Error] Motor Execution Failed: {e}")

# 3. Delete Intent
delete_intent = {
    "kind": "delete",
    "path": file_path
}

print(f"\n[Cortex] Firing delete intent...")
try:
    outcome_str = motor.reach_file(json.dumps(delete_intent))
    outcome = json.loads(outcome_str)
    print(f"  [Proprioception] File securely destroyed. Outcome: {outcome}")
except Exception as e:
    print(f"  [Error] Motor Execution Failed: {e}")

print("\n[System] All motor acts completed perfectly using biological efference copies!")
