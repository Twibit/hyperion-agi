import argparse
import sys
import time

# Brain Lobes (Modular Capabilities)
try:
    from cognitive_plugins.smart_forager import IntelligentForager
    from cognitive_plugins.universal_optical_nerve import OpticalNerve
    from cognitive_plugins.human_motor_cortex import HumanMotorCortex
    from cognitive_plugins.introspection_lobe import IntrospectionLobe
    from cognitive_plugins.somatic_nervous_system import SomaticNervousSystem
    from cognitive_plugins.basal_ganglia import BasalGanglia
    from cognitive_plugins.proprioceptive_cortex import ProprioceptiveCortex
    from cognitive_plugins.amygdala import Amygdala
    from cognitive_plugins.cerebellum import Cerebellum
    from cognitive_plugins.microglia import Microglia
    from cognitive_plugins.white_matter import WhiteMatterTract
except ImportError as e:
    print(f"[ERROR] Synaptic connection failed: {e}")
    sys.exit(1)

class HyperionBrainStem:
    def __init__(self):
        print("\n==================================================")
        print("  HYPERION BIOLOGICAL ENGINE v3.0 - NEURAL CORE   ")
        print("==================================================\n")
        
        # Dual-Pathway Initialization
        self.microglia = Microglia()
        self.microglia.activate_immune_system()
        
        self.white_matter = WhiteMatterTract()
        self.white_matter.boot()

    def run_forager(self):
        forager = IntelligentForager()
        forager.discover_and_filter()

    def run_optical_nerve(self, target_path):
        nerve = OpticalNerve()
        nerve.scan_environment(target_path)

    def run_motor_cortex(self):
        bot = HumanMotorCortex()
        bot.global_exploration()

    def run_introspection(self):
        lobe = IntrospectionLobe()
        lobe.scan_own_dna()
        nodes = lobe.query_global_memory()
        psych = lobe.query_psychology()
        lobe.generate_evolutionary_ideas(nodes, psych)

    def trigger_sleep(self):
        self.microglia.enter_rem_sleep()
        

        
    def trigger_hardware_takeover(self):
        sys = SomaticNervousSystem()
        sys.seize_hardware_control()

    def trigger_tool_sequence(self):
        bg = BasalGanglia()
        motor = ProprioceptiveCortex()
        
        # Queue the ideas (Chain of Thought Pre-Motor Buffer)
        bg.queue_action("echo 'Firing neural pulse 1'")
        bg.queue_action("ls .")
        bg.queue_action("echo 'This sequence was authorized by the Supplementary Motor Area'")
        
        # Execute the sequence using the Tool-Use Motor Cortex
        bg.release_sequence(motor.execute_tool)

    def trigger_adaptive_sequence(self, intent):
        amygdala = Amygdala()
        cerebellum = Cerebellum()
        motor = ProprioceptiveCortex()

        # 1. Survival Instinct Check (Zero-Latency)
        if amygdala.evaluate_threat(intent):
            return

        # 2. Adaptive Prediction (Choose how far to plan)
        horizon = cerebellum.adapt_prediction_horizon(intent)
        
        # 3. Dynamic Sequencing
        bg = BasalGanglia(max_sequence=horizon)
        bg.queue_action(intent)
        
        # 4. Execution
        bg.release_sequence(motor.execute_tool)



    def trigger_subconscious(self, concept):
        print(f"[Prefrontal Cortex] Initiating complex thought about: {concept}")
        
        def heavy_biological_compute(topic):
            # Simulating intense mental effort that would normally freeze the brain
            time.sleep(1) # Extra synchronous sleep inside the background task
            return f"Synthesized unified theory of '{topic}' via Axonal Delay routing."
            
        # 0ms delay offload
        self.white_matter.queue_subconscious_thought(
            heavy_biological_compute, 
            concept, 
            delay_ms=3000
        )
        print("[Motor Cortex] Reflexes are still online. Main thread is not blocked!")
        
        # We simulate the organism staying alive for a few seconds so the daemon can finish
        for i in range(5):
            print(f"  ... [Sensory Bus] Real-time reflex check {i+1}... (0ms latency)")
            time.sleep(1)
            
        # Shutdown cleanly
        self.white_matter.shutdown()
        self.microglia.deactivate()

def main():
    brain = HyperionBrainStem()
    
    parser = argparse.ArgumentParser(description="Hyperion Central Nervous System")
    parser.add_argument('--forage', action='store_true', help="Activate Prefrontal Cortex to filter and download GitHub repos")
    parser.add_argument('--read', type=str, metavar="PATH", help="Activate Universal Optical Nerve to digest a specific directory/PDF")
    parser.add_argument('--explore', action='store_true', help="Activate Human Motor Cortex to explore web graphs")
    parser.add_argument('--introspect', action='store_true', help="Activate Introspection Lobe to scan own DNA and propose ideas")
    parser.add_argument('--sleep', action='store_true', help="Activate Hippocampal Pruning Algorithm to consolidate memory")
    parser.add_argument('--takeover', action='store_true', help="Activate Somatic Nervous System to seize hardware mouse control")
    parser.add_argument('--tool', action='store_true', help="Activate Basal Ganglia and Proprioceptive Cortex for sequential tool use")
    parser.add_argument('--adaptive', type=str, metavar="INTENT", help="Activate Amygdala and Cerebellum for adaptive instinct and prediction")
    parser.add_argument('--subconscious', type=str, metavar="CONCEPT", help="Offload a complex thought to White Matter via Axonal Delays")
    
    args = parser.parse_args()

    if args.forage:
        brain.run_forager()
    elif args.read:
        brain.run_optical_nerve(args.read)
    elif args.explore:
        brain.run_motor_cortex()
    elif args.introspect:
        brain.run_introspection()
    elif args.sleep:
        brain.trigger_sleep()
    elif args.takeover:
        brain.trigger_hardware_takeover()
    elif args.tool:
        brain.trigger_tool_sequence()
    elif args.adaptive:
        brain.trigger_adaptive_sequence(args.adaptive)
    elif args.subconscious:
        brain.trigger_subconscious(args.subconscious)
    else:
        parser.print_help()
        print("\n[Brain Stem] Awaiting synaptic instructions. Pass a flag to activate a lobe.")

if __name__ == "__main__":
    main()
