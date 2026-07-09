import sys
from cognitive_plugins.omni_optical_nerve import OmniOpticalNerve

print("==================================================")
print(" TESTING COMBINED AGI EYE + AI_EYE (RUST) ")
print("==================================================\n")

eye = OmniOpticalNerve()
url = "https://example.com"

print(f"[System] Directing OmniOpticalNerve to look at: {url}")
connectome = eye.perceive(url)

print("\n==================================================")
print(" VISUAL CORTEX OUTPUT ")
print("==================================================")

print(f"\nInteractive UI Primitives Found: {len(connectome['interactive_nodes'])}")
for node in connectome['interactive_nodes']:
    print(f"  - [{node['type']}] '{node['label']}' at bbox {node['bbox']}")

print(f"\nSemantic Text Found: {len(connectome['semantic_nodes'])}")
for node in connectome['semantic_nodes']:
    print(f"  - {node['text']}")

print("\n[System] Vision test complete.")
