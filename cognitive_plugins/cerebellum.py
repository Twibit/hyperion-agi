class Cerebellum:
    def __init__(self):
        print("[Cerebellum] Forward Models online. Adaptive prediction engine active.")

    def adapt_prediction_horizon(self, intent):
        print("\n==================================================")
        print(f"[Cerebellum] Evaluating cognitive load for intent: '{intent}'")
        
        # Determine if we need System 1 (Instinct/Fast) or System 2 (Prediction/Slow)
        if len(intent) < 15 or "echo" in intent or "ls" in intent:
            horizon = 1
            print("[System 1 - Fast Thinking] Routine action detected. No forward prediction required. Relying on instinct.")
            print(f"[Prediction Horizon] Set to {horizon} step.")
        elif "analyze" in intent or "solve" in intent or "complex" in intent or "predict" in intent:
            # Calculate dynamic horizon based on the 'cognitive weight' of the intent
            complexity_score = len(intent.split()) + sum(3 for word in intent.split() if word.lower() in ['analyze', 'solve', 'complex', 'predict', 'model', 'future'])
            horizon = min(100, max(5, int(complexity_score * 1.5)))
            print("[System 2 - Slow Thinking] Complex abstract problem detected. Activating deep Forward Models.")
            print(f"[Prediction Horizon] Dynamically scaled to {horizon} steps based on cognitive load variables...")
        else:
            horizon = 3
            print("[Balanced Thinking] Moderate complexity. Default sequencing.")
            print(f"[Prediction Horizon] Set to {horizon} steps.")
            
        print("==================================================")
        return horizon
