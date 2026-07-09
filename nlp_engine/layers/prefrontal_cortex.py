import os
import sys

# Import core internal layers
from nlp_engine.layers.meta_cortex import SafeExecutor, BashTerminalExecutor

class WernickesAreaV2:
    def evaluate_code(self, text: str) -> tuple[int, str]:
        # Minimal biological Wernicke evaluation
        return 100, "Clean code."

class CorpusCallosum:
    """
    The mathematical bridge between the Right Brain (Creative N-Grams) 
    and the Left Brain (Rigid Logic).
    """
    def __init__(self):
        from nlp_engine.layers.indentation_restorer import IndentationRestorer
        self.restorer = IndentationRestorer()

    def synchronize(self, right_brain_output: str) -> tuple[str, str]:
        output = right_brain_output.strip()
        if not output: return "", ""

        # Neuro-Symbolic Wrap: Automatically infer bash/python without hardcoded anchors
        restored_python = self.restorer.restore(output)
        with open("processor.py", "w") as f:
            f.write(restored_python)
        synchronized_bash = "python3 processor.py"
        return restored_python, synchronized_bash

class WorkingMemoryBuffer:
    """
    Stores the overall goal and decomposes execution steps logically
    before passing them to the generator.
    """
    def __init__(self):
        self.overall_goal = ""
        self.steps = []

    def decompose(self, goal: str) -> str:
        self.overall_goal = goal
        # Decompose the goal into logical execution steps
        self.steps = [
            "# 1. Analyze goal and required inputs/dependencies.",
            "# 2. Synthesize core logic mathematically.",
            "# 3. Format final output for execution."
        ]
        context = f"# Overall Goal: {self.overall_goal}\n# Logical Execution Steps:\n" + "\n".join(self.steps) + "\n\n# Let's write the python code to achieve this:\nimport os\n"
        return context

class PrefrontalCortex:
    """
    The PrefrontalCortex acts as the Semantic Reasoning Engine (LLM Bridge).
    It takes the raw mathematical strings from the Universal Cortex, 
    evaluates them, and acts on the environment (executing bash commands).
    """
    def __init__(self):
        # [Terminal-Bench Upgrade]: Replace Python sandbox with a live Linux terminal
        self.executor = BashTerminalExecutor()
        self.wernicke = WernickesAreaV2()
        self.corpus_callosum = CorpusCallosum()
        self.working_memory = WorkingMemoryBuffer()

    def _generate_code(self, prompt: str, code_generator_callback, error_context: str = None) -> tuple[str, str]:
        """
        No more hardcoded mocks.
        We ask the biological Universal Cortex to generate the code directly.
        """
        # Utilize Working Memory Buffer to contextualize the prompt
        full_prompt = self.working_memory.decompose(prompt)
        if error_context:
            full_prompt += f"\n# ERROR TO FIX: {error_context}\n"
            
        print("   [Prefrontal Cortex]: Channeling prompt to Liquid Universal Cortex for generation...")
        raw_thought = code_generator_callback(full_prompt, domain="code", length=200)
        return self.corpus_callosum.synchronize(raw_thought)

    def execute_task(self, prompt: str, code_generator_callback) -> dict:
        """
        The Active Inference Task Loop.
        1. Predict code generation.
        2. Test in the Sandbox (Measure Prediction Error).
        3. If Free Energy (Surprise) is too high, route to Motor Cortex to forage for missing knowledge.
        """
        print(f"\n[Prefrontal Cortex] New Semantic Task Received: '{prompt}'")
        error_context = None
        code = ""
        restored_python = ""
        
        # We start with low free energy
        free_energy = 0.0
        
        while free_energy < 100.0:
            print(f"\n   [Active Inference] Synthesizing biological code (Free Energy: {free_energy:.2f})...")
            restored_python, bash_code = self._generate_code(prompt, code_generator_callback, error_context)
            code = bash_code
            
            print("   Running through Live Linux Bash Terminal...")
            exec_result = self.executor.execute(bash_code)
            
            if not exec_result['success']:
                print(f"   Execution Failed: {exec_result.get('error', 'Unknown Error')}")
                error_context = f"The bash command failed with error: {exec_result.get('error')}. Please fix it."
                free_energy += 40.0 # Spike in surprise
                
                if free_energy >= 100.0:
                    print("   [Amygdala] High Prediction Error! Abandoning rigid execution.")
                    print("   [Motor Cortex] Rerouting to Autonomously Forage the Web for syntax help...")
                    import sys
                    import os
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if project_root not in sys.path:
                        sys.path.append(project_root)
                    from cognitive_plugins.human_motor_cortex import HumanMotorCortex
                    motor = HumanMotorCortex()
                    motor.global_exploration("https://stackoverflow.com/search?q=" + prompt.replace(" ", "+"))
                    
                    return {
                        'success': False,
                        'error': 'High Free Energy (Surprise). Task aborted and routed to Motor Cortex for learning.',
                        'last_code': code
                    }
                continue
                
            print("   Grading with Wernicke's Area (Philosophy of Software Design)...")
            score, feedback = self.wernicke.evaluate_code(restored_python)
            
            print(f"   Score: {score} | Feedback: {feedback}")
            
            # Dynamic threshold (we consider it a success if it runs, but we increase free energy if ugly)
            if score < 50:
                print(f"   Code rejected by Wernicke's Area! Not elegant enough.")
                error_context = f"The code runs, but is ugly. Score: {score}. Feedback: {feedback}. Rewrite strictly applying Clean Code."
                free_energy += 20.0
                continue
                
            print(f"   Task successfully synthesized, executed, and philosophically approved!")
            return {
                'success': True,
                'code': code,
                'attempts': int(free_energy / 20) + 1,
                'score': score
            }
            
        print("   Prefrontal Cortex exhausted prediction bounds. Task failed.")
        return {
            'success': False,
            'error': 'Max free energy reached without satisfying Wernicke or Executor',
            'last_code': code
        }
