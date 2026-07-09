"""
Debugger Lobe (Automated Program Repair)
Gives Hyperion the ability to autonomously fix bugs in Python code using 
Abstract Syntax Tree (AST) mutation and genetic evolution.
"""
import ast
import random
import copy
from nlp_engine.layers.meta_cortex import SafeExecutor

class ASTMutator(ast.NodeTransformer):
    """Mutates Python code at the structural level to find bug fixes."""
    
    def __init__(self, mutation_rate=0.1):
        self.mutation_rate = mutation_rate
        self.mutations_applied = 0
        
    def visit_Compare(self, node):
        """Swap comparison operators (e.g., < to <=, == to !=)"""
        self.generic_visit(node)
        if random.random() < self.mutation_rate:
            for i, op in enumerate(node.ops):
                if isinstance(op, ast.Lt): node.ops[i] = ast.LtE()
                elif isinstance(op, ast.LtE): node.ops[i] = ast.Lt()
                elif isinstance(op, ast.Gt): node.ops[i] = ast.GtE()
                elif isinstance(op, ast.GtE): node.ops[i] = ast.Gt()
                elif isinstance(op, ast.Eq): node.ops[i] = ast.NotEq()
                elif isinstance(op, ast.NotEq): node.ops[i] = ast.Eq()
                self.mutations_applied += 1
        return node
        
    def visit_BinOp(self, node):
        """Swap binary operators (e.g., + to -, * to /)"""
        self.generic_visit(node)
        if random.random() < self.mutation_rate:
            if isinstance(node.op, ast.Add): node.op = ast.Sub()
            elif isinstance(node.op, ast.Sub): node.op = ast.Add()
            elif isinstance(node.op, ast.Mult): node.op = ast.FloorDiv()
            elif isinstance(node.op, ast.FloorDiv): node.op = ast.Mult()
            self.mutations_applied += 1
        return node
        
    def visit_Constant(self, node):
        """Mutate numbers (off-by-one errors)"""
        self.generic_visit(node)
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            if random.random() < self.mutation_rate:
                node.value += random.choice([-1, 1])
                self.mutations_applied += 1
        return node


class AutonomousDebugger:
    """The autonomous bug-fixing engine."""
    
    def __init__(self):
        self.executor = SafeExecutor()
        
    def debug(self, buggy_code: str, test_code: str, max_generations: int = 100) -> dict:
        """
        Attempt to automatically repair a buggy Python function.
        
        Args:
            buggy_code: The source code containing the bug
            test_code: An assertion that should pass if the bug is fixed (must set `result = True` if success)
            max_generations: Max mutation attempts
            
        Returns:
            dict with: 'success', 'fixed_code', 'attempts'
        """
        # First, ensure the bug actually exists (test should fail)
        combined_code = buggy_code + "\n" + test_code
        initial_result = self.executor.execute(combined_code)
        
        if initial_result['success'] and initial_result.get('result') is True:
            return {'success': True, 'fixed_code': buggy_code, 'attempts': 0, 'message': 'Code already passes test.'}
            
        try:
            original_tree = ast.parse(buggy_code)
        except SyntaxError:
            return {'success': False, 'error': 'Buggy code has a SyntaxError, cannot parse AST.'}
            
        # Genetic repair loop
        for attempt in range(1, max_generations + 1):
            # Clone the AST tree
            tree_copy = copy.deepcopy(original_tree)
            
            # Apply random mutations
            mutator = ASTMutator(mutation_rate=0.2)
            mutated_tree = mutator.visit(tree_copy)
            ast.fix_missing_locations(mutated_tree)
            
            if mutator.mutations_applied == 0:
                continue # No mutations applied, skip test
                
            # Convert back to source code
            try:
                candidate_code = ast.unparse(mutated_tree)
            except Exception:
                continue
                
            # Test the candidate
            candidate_combined = candidate_code + "\n" + test_code
            exec_result = self.executor.execute(candidate_combined)
            
            if exec_result['success'] and exec_result.get('result') is True:
                return {
                    'success': True,
                    'fixed_code': candidate_code,
                    'attempts': attempt,
                    'mutations': mutator.mutations_applied
                }
                
        return {'success': False, 'attempts': max_generations, 'error': 'Could not find a working patch.'}
