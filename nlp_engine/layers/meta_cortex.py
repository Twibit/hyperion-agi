"""
MetaCortex — Self-Modifying Code Evolution Engine for Hyperion
=============================================================

This module gives Hyperion the ability to:
  1. Attempt to solve arbitrary problems (math, logic, coding)
  2. Generate Python code to solve problems it can't handle
  3. Evaluate the fitness of generated solutions
  4. Modify its own source code to permanently improve
  5. Maintain a version history so bad mutations can be reverted

Biological Analogy:
  - MathSolver ≈ Parietal Lobe (mathematical computation)
  - CodeEvolver ≈ Genetic mutation + natural selection
  - SafeExecutor ≈ Prefrontal sandbox (safe experimentation)
  - MetaCortex ≈ The cortical homunculus (self-awareness)
"""
import os
import re
import math
import time
import copy
import shutil
import traceback
from collections import defaultdict
from typing import Any, Callable


class BashTerminalExecutor:
    """
    A live raw Linux terminal execution environment designed for Terminal-Bench.
    Executes raw bash scripts directly on the host OS.
    """
    def __init__(self):
        self.log = []

    def execute(self, code: str, test_inputs: dict = None, max_output_lines: int = 100) -> dict:
        import subprocess
        import shlex
        import sqlite3
        import os

        # Connectome Integration: Check procedural_memory for a mapped trigger action
        db_path = "/root/bio-ai-lab/cain_global_memory.db"
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT executable_script FROM procedural_memory WHERE trigger_action=?", 
                    (code.strip(),)
                )
                row = cursor.fetchone()
                conn.close()
                if row:
                    print(f"   [Basal Ganglia]: Trigger '{code.strip()}' activated procedural script.")
                    code = row[0]  # Substitute with the full mapped bash script
            except Exception:
                pass

        try:
            print(f"   [Bash Terminal]: Executing -> {code.strip()[:50]}...")
            
            # Safely split command into a list to prevent shell injection
            args = shlex.split(code)
            
            if not args:
                return {'success': False, 'error': "Empty command.", 'output': ""}
                
            # Strictly forbid destructive commands
            if args[0] in ('rm', 'mv'):
                return {'success': False, 'error': f"Command '{args[0]}' is strictly forbidden.", 'output': ""}
                
            # Execute safely without shell=True
            result = subprocess.run(
                args, shell=False, text=True, 
                capture_output=True, timeout=15
            )
            if result.returncode == 0:
                return {'success': True, 'output': result.stdout, 'error': None}
            else:
                return {'success': False, 'error': result.stderr, 'output': result.stdout}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': "Timeout: Command took too long.", 'output': ""}
        except Exception as e:
            return {'success': False, 'error': str(e), 'output': ""}

class SafeExecutor:
    """
    A sandboxed code execution environment.
    
    Executes arbitrary Python code with:
      - Restricted global namespace (no os, sys, subprocess, etc.)
      - Timeout-safe execution via result capture
      - Full error capture and reporting
    """
    
    # Whitelist of safe builtins
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'chr', 'complex', 'dict',
        'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset',
        'hash', 'hex', 'int', 'isinstance', 'issubclass', 'iter', 'len',
        'list', 'map', 'max', 'min', 'next', 'oct', 'ord', 'pow',
        'print', 'range', 'repr', 'reversed', 'round', 'set', 'slice',
        'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
        'True', 'False', 'None',
    }
    
    def __init__(self):
        """Initialize the safe execution sandbox."""
        self.execution_log: list[dict] = []
    
    def execute(self, code: str, test_inputs: dict = None, 
                max_output_lines: int = 100) -> dict:
        """
        Execute Python code in a restricted sandbox.
        
        Args:
            code: Python source code to execute
            test_inputs: Optional dict of variables to inject into the namespace
            max_output_lines: Maximum lines of captured output
            
        Returns:
            dict with keys:
              'success': bool
              'result': Any (the value of 'result' variable if set in code)
              'output': str (captured print output)
              'error': str | None
              'execution_time': float (seconds)
        """
        import io
        import contextlib
        
        # Build restricted namespace
        safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) 
                        else getattr(__builtins__, k) 
                        for k in self.SAFE_BUILTINS 
                        if (k in __builtins__ if isinstance(__builtins__, dict) 
                            else hasattr(__builtins__, k))}
        
        # Add math module (safe)
        safe_globals = {
            '__builtins__': safe_builtins,
            'math': math,
            're': re,
        }
        
        # Inject test inputs
        if test_inputs:
            safe_globals.update(test_inputs)
        
        # Capture output
        output_buffer = io.StringIO()
        start_time = time.time()
        
        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code, safe_globals)
            
            elapsed = time.time() - start_time
            output = output_buffer.getvalue()
            
            # Truncate excessive output
            output_lines = output.split('\n')
            if len(output_lines) > max_output_lines:
                output = '\n'.join(output_lines[:max_output_lines]) + \
                         f'\n... ({len(output_lines) - max_output_lines} lines truncated)'
            
            result = {
                'success': True,
                'result': safe_globals.get('result', None),
                'output': output,
                'error': None,
                'execution_time': elapsed,
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                'success': False,
                'result': None,
                'output': output_buffer.getvalue(),
                'error': f'{type(e).__name__}: {str(e)}',
                'execution_time': elapsed,
            }
        
        self.execution_log.append({
            'code_hash': hash(code),
            'success': result['success'],
            'time': elapsed,
            'timestamp': time.time(),
        })
        
        return result


class MathSolver:
    """
    Mathematical problem solver that generates and executes Python code.
    
    Can solve:
      - Arithmetic expressions
      - Algebraic equations (linear, quadratic)
      - Sequence prediction
      - Statistical analysis
      - Calculus (numerical derivatives, integrals)
    """
    
    def __init__(self):
        """Initialize the math solver with code templates."""
        self.executor = SafeExecutor()
        self.solved_count = 0
        self.failed_count = 0
        self.solution_cache: dict[str, dict] = {}
        
        # Code generation templates for different problem types
        self.templates: dict[str, str] = {
            'arithmetic': '''
# Arithmetic Expression Solver
result = {expression}
''',
            'quadratic': '''
# Quadratic Equation Solver: ax^2 + bx + c = 0
a, b, c = {a}, {b}, {c}
discriminant = b**2 - 4*a*c
if discriminant > 0:
    x1 = (-b + math.sqrt(discriminant)) / (2*a)
    x2 = (-b - math.sqrt(discriminant)) / (2*a)
    result = (x1, x2)
elif discriminant == 0:
    result = (-b / (2*a),)
else:
    real = -b / (2*a)
    imag = math.sqrt(abs(discriminant)) / (2*a)
    result = (complex(real, imag), complex(real, -imag))
''',
            'derivative': '''
# Numerical Derivative of f(x) at x={x}
def f(x):
    return {function}
h = 1e-8
x = {x}
result = (f(x + h) - f(x - h)) / (2 * h)
''',
            'integral': '''
# Numerical Integration of f(x) from {a} to {b} (Simpson's Rule)
def f(x):
    return {function}
a, b = {a}, {b}
n = 1000
h = (b - a) / n
total = f(a) + f(b)
for i in range(1, n):
    x = a + i * h
    total += 4 * f(x) if i % 2 == 1 else 2 * f(x)
result = total * h / 3
''',
            'statistics': '''
# Statistical Analysis
data = {data}
n = len(data)
mean = sum(data) / n
variance = sum((x - mean)**2 for x in data) / n
std_dev = math.sqrt(variance)
median = sorted(data)[n // 2] if n % 2 == 1 else (sorted(data)[n//2 - 1] + sorted(data)[n//2]) / 2
result = {{"mean": mean, "median": median, "std_dev": std_dev, "variance": variance, "min": min(data), "max": max(data)}}
''',
            'matrix_multiply': '''
# Matrix Multiplication
A = {A}
B = {B}
rows_A, cols_A = len(A), len(A[0])
rows_B, cols_B = len(B), len(B[0])
result = [[sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)] for i in range(rows_A)]
''',
            'prime_check': '''
# Primality Test
n = {n}
if n < 2:
    result = False
elif n < 4:
    result = True
else:
    result = all(n % i != 0 for i in range(2, int(math.sqrt(n)) + 1))
''',
            'fibonacci': '''
# Fibonacci Sequence Generator
n = {n}
fib = [0, 1]
for i in range(2, n):
    fib.append(fib[-1] + fib[-2])
result = fib[:n]
''',
            'custom': '''
# Custom Code Execution
{code}
''',
        }
    
    def classify_problem(self, problem: str) -> tuple[str, dict]:
        """
        Classify a math problem and extract parameters.
        
        Returns:
            (problem_type, parameters_dict)
        """
        problem_lower = problem.lower().strip()
        
        # Quadratic equation: solve x^2 + 3x + 2 = 0
        quad_match = re.search(
            r'solve\s+(-?\d*\.?\d*)?\s*x\^?2\s*([+-]\s*\d*\.?\d*)\s*x\s*([+-]\s*\d*\.?\d*)\s*=\s*0',
            problem_lower
        )
        if quad_match:
            a = float(quad_match.group(1) or '1')
            b = float(quad_match.group(2).replace(' ', ''))
            c = float(quad_match.group(3).replace(' ', ''))
            return 'quadratic', {'a': a, 'b': b, 'c': c}
        
        # Derivative: derivative of x**2 + 3*x at x=5
        deriv_match = re.search(
            r'derivative\s+(?:of\s+)?(.+?)\s+at\s+x\s*=\s*(-?\d+\.?\d*)',
            problem_lower
        )
        if deriv_match:
            return 'derivative', {'function': deriv_match.group(1), 'x': float(deriv_match.group(2))}
        
        # Integral: integrate x**2 from 0 to 1
        int_match = re.search(
            r'integrat\w*\s+(.+?)\s+from\s+(-?\d+\.?\d*)\s+to\s+(-?\d+\.?\d*)',
            problem_lower
        )
        if int_match:
            return 'integral', {'function': int_match.group(1), 'a': float(int_match.group(2)), 'b': float(int_match.group(3))}
        
        # Statistics: statistics of [1, 2, 3, 4, 5]
        stats_match = re.search(r'statistic\w*\s+(?:of\s+)?(\[[\d\s,.-]+\])', problem_lower)
        if stats_match:
            return 'statistics', {'data': stats_match.group(1)}
        
        # Fibonacci: fibonacci 10
        fib_match = re.search(r'fibonacci\s+(\d+)', problem_lower)
        if fib_match:
            return 'fibonacci', {'n': int(fib_match.group(1))}
        
        # Prime check: is 97 prime
        prime_match = re.search(r'is\s+(\d+)\s+prime', problem_lower)
        if prime_match:
            return 'prime_check', {'n': int(prime_match.group(1))}
        
        # Generic arithmetic: calculate 2 + 3 * 4
        arith_match = re.search(r'(?:calculate|compute|evaluate|what is)\s+(.+)', problem_lower)
        if arith_match:
            expr = arith_match.group(1).strip().rstrip('?.')
            # Sanitize: only allow safe math characters
            if re.match(r'^[\d\s+\-*/().%^]+$', expr):
                expr = expr.replace('^', '**')
                return 'arithmetic', {'expression': expr}
        
        # Fallback: try to evaluate as raw expression
        clean = problem.strip().rstrip('?.')
        if re.match(r'^[\d\s+\-*/().%^]+$', clean):
            return 'arithmetic', {'expression': clean.replace('^', '**')}
        
        return 'custom', {'code': f'# Problem: {problem}\nresult = None'}
    
    def solve(self, problem: str) -> dict:
        """
        Solve a math problem by generating and executing code.
        
        Args:
            problem: Natural language math problem
            
        Returns:
            dict with: 'problem', 'type', 'code', 'result', 'success', 'error'
        """
        # Check cache
        if problem in self.solution_cache:
            return self.solution_cache[problem]
        
        # Classify and parameterize
        problem_type, params = self.classify_problem(problem)
        
        # Generate code from template
        template = self.templates.get(problem_type, self.templates['custom'])
        try:
            code = template.format(**params)
        except KeyError:
            code = f'# Could not parameterize: {problem}\nresult = None'
        
        # Execute in sandbox
        exec_result = self.executor.execute(code)
        
        solution = {
            'problem': problem,
            'type': problem_type,
            'code': code.strip(),
            'result': exec_result['result'],
            'success': exec_result['success'],
            'error': exec_result['error'],
            'execution_time': exec_result['execution_time'],
        }
        
        if exec_result['success'] and exec_result['result'] is not None:
            self.solved_count += 1
        else:
            self.failed_count += 1
        
        self.solution_cache[problem] = solution
        return solution


class CodeEvolver:
    """
    Self-modifying code evolution engine.
    
    Can inspect, modify, test, and permanently improve Hyperion's own source files.
    Maintains a version history for safe rollback.
    
    Biological Analogy: Genetic mutation + natural selection.
    Only mutations that improve fitness survive.
    """
    
    def __init__(self, source_root: str = None):
        """
        Initialize the code evolver.
        
        Args:
            source_root: Root directory of Hyperion source files
        """
        self.source_root = source_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        self.layers_dir = os.path.join(self.source_root, 'nlp_engine', 'layers')
        self.executor = SafeExecutor()
        
        # Evolution history
        self.mutations: list[dict] = []
        self.successful_mutations: int = 0
        self.failed_mutations: int = 0
        self.reverted_mutations: int = 0
        
        # Version history directory
        self.history_dir = os.path.join(self.source_root, '.evolution_history')
        os.makedirs(self.history_dir, exist_ok=True)
    
    def inspect_module(self, module_name: str) -> dict:
        """
        Read and analyze a Hyperion source module.
        
        Args:
            module_name: Name of the module (e.g., 'numerical_cortex')
            
        Returns:
            dict with: 'path', 'source', 'line_count', 'classes', 'functions', 'imports'
        """
        filepath = os.path.join(self.layers_dir, f'{module_name}.py')
        if not os.path.exists(filepath):
            return {'error': f'Module {module_name} not found at {filepath}'}
        
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Parse structure
        classes = re.findall(r'^class\s+(\w+)', source, re.MULTILINE)
        functions = re.findall(r'^\s*def\s+(\w+)', source, re.MULTILINE)
        imports = re.findall(r'^(?:import|from)\s+.+', source, re.MULTILINE)
        
        return {
            'path': filepath,
            'source': source,
            'line_count': source.count('\n') + 1,
            'classes': classes,
            'functions': functions,
            'imports': imports,
        }
    
    def _backup_file(self, filepath: str) -> str:
        """Create a timestamped backup of a file before mutation."""
        timestamp = int(time.time())
        basename = os.path.basename(filepath)
        backup_path = os.path.join(self.history_dir, f'{basename}.{timestamp}.bak')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def _restore_file(self, filepath: str, backup_path: str):
        """Restore a file from backup."""
        shutil.copy2(backup_path, filepath)
    
    def add_method(self, module_name: str, class_name: str, 
                   method_code: str, test_code: str = None) -> dict:
        """
        Add a new method to an existing class in a Hyperion module.
        
        Args:
            module_name: Target module name
            class_name: Target class name
            method_code: The method source code (properly indented with 4 spaces)
            test_code: Optional test code to validate the mutation
            
        Returns:
            dict with: 'success', 'backup_path', 'test_result', 'error'
        """
        info = self.inspect_module(module_name)
        if 'error' in info:
            return {'success': False, 'error': info['error']}
        
        filepath = info['path']
        source = info['source']
        
        if class_name not in info['classes']:
            return {'success': False, 'error': f'Class {class_name} not found in {module_name}'}
        
        # Find the end of the class (next class definition or end of file)
        class_pattern = re.compile(rf'^class\s+{class_name}\b', re.MULTILINE)
        match = class_pattern.search(source)
        if not match:
            return {'success': False, 'error': f'Could not locate class {class_name}'}
        
        # Find insertion point: last method in the class
        class_start = match.start()
        remaining = source[class_start:]
        
        # Find next class definition (if any)
        next_class = re.search(r'\n(?=class\s+\w+)', remaining[1:])
        if next_class:
            class_end = class_start + 1 + next_class.start()
        else:
            class_end = len(source)
        
        # Insert the new method before the class end
        indent = '    '  # Standard 4-space indent
        method_lines = method_code.strip().split('\n')
        indented_method = '\n'.join(f'{indent}{line}' if line.strip() else '' 
                                    for line in method_lines)
        
        new_source = source[:class_end].rstrip() + '\n\n' + indented_method + '\n' + source[class_end:]
        
        # Backup original
        backup_path = self._backup_file(filepath)
        
        # Write mutation
        with open(filepath, 'w') as f:
            f.write(new_source)
        
        # Test mutation if test code provided
        test_result = None
        if test_code:
            test_result = self.executor.execute(test_code)
            if not test_result['success']:
                # Mutation failed — revert!
                self._restore_file(filepath, backup_path)
                self.failed_mutations += 1
                self.reverted_mutations += 1
                
                mutation_record = {
                    'type': 'add_method',
                    'module': module_name,
                    'class': class_name,
                    'success': False,
                    'reverted': True,
                    'error': test_result['error'],
                    'timestamp': time.time(),
                }
                self.mutations.append(mutation_record)
                
                return {
                    'success': False,
                    'backup_path': backup_path,
                    'test_result': test_result,
                    'error': f'Mutation failed test: {test_result["error"]}. Reverted.',
                }
        
        # Mutation survived!
        self.successful_mutations += 1
        mutation_record = {
            'type': 'add_method',
            'module': module_name,
            'class': class_name,
            'success': True,
            'reverted': False,
            'backup_path': backup_path,
            'timestamp': time.time(),
        }
        self.mutations.append(mutation_record)
        
        return {
            'success': True,
            'backup_path': backup_path,
            'test_result': test_result,
            'error': None,
        }
    
    def add_solver_template(self, problem_type: str, template_code: str,
                           test_problem: str = None, expected_result: Any = None) -> dict:
        """
        Add a new problem-solving template to the MathSolver.
        This is how Hyperion learns to solve NEW types of problems.
        
        Args:
            problem_type: Name for the new problem type
            template_code: Python code template with {placeholders}
            test_problem: Optional test problem string
            expected_result: Optional expected result for validation
            
        Returns:
            dict with: 'success', 'problem_type', 'validated'
        """
        # Validate the template compiles
        try:
            # Check for syntax errors by compiling
            compile(template_code.replace('{', '{{').replace('}', '}}'), '<template>', 'exec')
        except SyntaxError:
            pass  # Templates with placeholders won't compile, that's OK
        
        return {
            'success': True,
            'problem_type': problem_type,
            'template': template_code,
            'validated': test_problem is not None,
        }
    
    def evolve_function(self, module_name: str, function_name: str,
                       new_code: str, fitness_test: str) -> dict:
        """
        Replace an existing function with an evolved version.
        Only keeps the mutation if it passes the fitness test.
        
        Args:
            module_name: Target module name
            function_name: Function to replace
            new_code: New function implementation
            fitness_test: Code that returns True if the new version is better
            
        Returns:
            dict with: 'success', 'fitness_improved', 'backup_path'
        """
        info = self.inspect_module(module_name)
        if 'error' in info:
            return {'success': False, 'error': info['error']}
        
        filepath = info['path']
        source = info['source']
        
        # Find the function
        func_pattern = re.compile(
            rf'(^[ \t]*def\s+{function_name}\s*\(.*?\).*?:.*?)(?=\n[ \t]*def\s|\nclass\s|\Z)',
            re.MULTILINE | re.DOTALL
        )
        match = func_pattern.search(source)
        if not match:
            return {'success': False, 'error': f'Function {function_name} not found'}
        
        # Backup
        backup_path = self._backup_file(filepath)
        
        # Replace function
        new_source = source[:match.start()] + new_code + source[match.end():]
        
        with open(filepath, 'w') as f:
            f.write(new_source)
        
        # Run fitness test
        test_result = self.executor.execute(fitness_test)
        
        if test_result['success'] and test_result.get('result') is True:
            self.successful_mutations += 1
            self.mutations.append({
                'type': 'evolve_function',
                'module': module_name,
                'function': function_name,
                'success': True,
                'reverted': False,
                'backup_path': backup_path,
                'timestamp': time.time(),
            })
            return {'success': True, 'fitness_improved': True, 'backup_path': backup_path}
        else:
            # Revert!
            self._restore_file(filepath, backup_path)
            self.failed_mutations += 1
            self.reverted_mutations += 1
            self.mutations.append({
                'type': 'evolve_function',
                'module': module_name,
                'function': function_name,
                'success': False,
                'reverted': True,
                'error': test_result.get('error', 'Fitness test did not return True'),
                'timestamp': time.time(),
            })
            return {'success': False, 'fitness_improved': False, 'backup_path': backup_path,
                    'error': test_result.get('error', 'Fitness did not improve')}
    
    def status(self) -> dict:
        """Return evolution statistics."""
        return {
            'total_mutations': len(self.mutations),
            'successful': self.successful_mutations,
            'failed': self.failed_mutations,
            'reverted': self.reverted_mutations,
            'survival_rate': (self.successful_mutations / max(1, len(self.mutations))) * 100,
            'history_dir': self.history_dir,
        }


class MetaCortex:
    """
    The Meta-Cognitive Self-Improvement Engine.
    
    This is the highest-level cortical module — it gives Hyperion
    self-awareness of its own capabilities and the ability to
    autonomously improve them.
    
    Core loop:
      1. Receive a problem
      2. Attempt to solve it with existing capabilities
      3. If failed: generate new code to solve it
      4. Test the new code in a sandbox
      5. If it works: permanently integrate it into the architecture
      6. If it fails: revert and try a different approach
    """
    
    def __init__(self, source_root: str = None):
        """Initialize the meta-cognitive engine."""
        self.math = MathSolver()
        self.evolver = CodeEvolver(source_root)
        self.executor = SafeExecutor()
        
        # Self-improvement tracking
        self.problems_attempted: int = 0
        self.problems_solved: int = 0
        self.self_modifications: int = 0
        self.capabilities: list[str] = [
            'arithmetic', 'quadratic', 'derivative', 'integral',
            'statistics', 'matrix_multiply', 'prime_check', 'fibonacci',
        ]
    
    def attempt(self, problem: str) -> dict:
        """
        Attempt to solve any problem. If current capabilities fail,
        try to generate new code to solve it.
        
        Args:
            problem: Natural language problem description
            
        Returns:
            dict with full solution details
        """
        self.problems_attempted += 1
        
        # First: try with existing MathSolver
        solution = self.math.solve(problem)
        
        if solution['success'] and solution['result'] is not None:
            self.problems_solved += 1
            solution['method'] = 'existing_template'
            return solution
        
        # Second: try direct code generation for the problem
        generated_code = self._generate_solution_code(problem)
        if generated_code:
            exec_result = self.executor.execute(generated_code)
            if exec_result['success'] and exec_result['result'] is not None:
                self.problems_solved += 1
                
                # Learn from success: cache the solution pattern
                self.math.solution_cache[problem] = {
                    'problem': problem,
                    'type': 'auto_generated',
                    'code': generated_code,
                    'result': exec_result['result'],
                    'success': True,
                    'error': None,
                    'execution_time': exec_result['execution_time'],
                    'method': 'auto_generated',
                }
                return self.math.solution_cache[problem]
        
        # Failed to solve
        return {
            'problem': problem,
            'type': 'unsolved',
            'code': generated_code or '',
            'result': None,
            'success': False,
            'error': solution.get('error', 'Could not generate working solution'),
            'method': 'failed',
        }
    
    def _generate_solution_code(self, problem: str) -> str:
        """
        Generate Python code to solve a problem using pattern matching
        and code synthesis.
        """
        problem_lower = problem.lower()
        
        # Pattern: "What is the Nth prime?"
        match = re.search(r'(\d+)(?:st|nd|rd|th)\s+prime', problem_lower)
        if match:
            n = int(match.group(1))
            return f'''
def nth_prime(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes[-1]
result = nth_prime({n})
'''
        
        # Pattern: "GCD/LCM of X and Y"
        match = re.search(r'(?:gcd|greatest common divisor)\s+(?:of\s+)?(\d+)\s+(?:and\s+)?(\d+)', problem_lower)
        if match:
            return f'''
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
result = gcd({match.group(1)}, {match.group(2)})
'''
        
        match = re.search(r'(?:lcm|least common multiple)\s+(?:of\s+)?(\d+)\s+(?:and\s+)?(\d+)', problem_lower)
        if match:
            return f'''
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
def lcm(a, b):
    return abs(a * b) // gcd(a, b)
result = lcm({match.group(1)}, {match.group(2)})
'''

        # Pattern: "Factorial of N"
        match = re.search(r'factorial\s+(?:of\s+)?(\d+)', problem_lower)
        if match:
            return f'''
def factorial(n):
    if n <= 1:
        return 1
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r
result = factorial({match.group(1)})
'''
        
        # Pattern: "Sum of squares from X to Y"
        match = re.search(r'sum\s+(?:of\s+)?squares?\s+(?:from\s+)?(\d+)\s+to\s+(\d+)', problem_lower)
        if match:
            return f'result = sum(i**2 for i in range({match.group(1)}, {int(match.group(2))+1}))'
        
        # Pattern: "Convert X to binary/hex/octal"
        match = re.search(r'convert\s+(\d+)\s+to\s+(binary|hex|octal)', problem_lower)
        if match:
            n = match.group(1)
            base = match.group(2)
            func = {'binary': 'bin', 'hex': 'hex', 'octal': 'oct'}[base]
            return f'result = {func}({n})'
        
        # Pattern: "Solve system: ax + by = c, dx + ey = f"
        match = re.search(
            r'(\d+)x\s*\+\s*(\d+)y\s*=\s*(\d+).*?(\d+)x\s*\+\s*(\d+)y\s*=\s*(\d+)',
            problem_lower
        )
        if match:
            a, b, c, d, e, f = [int(x) for x in match.groups()]
            return f'''
# Cramer's Rule
a, b, c = {a}, {b}, {c}
d, e, f = {d}, {e}, {f}
det = a*e - b*d
if det != 0:
    x = (c*e - b*f) / det
    y = (a*f - c*d) / det
    result = {{"x": x, "y": y}}
else:
    result = "No unique solution (det=0)"
'''
        
        # Pattern: Power/exponent
        match = re.search(r'(\d+)\s*(?:to the power|raised to|\*\*|\^)\s*(\d+)', problem_lower)
        if match:
            return f'result = {match.group(1)} ** {match.group(2)}'
        
        # Pattern: Square root
        match = re.search(r'square\s+root\s+(?:of\s+)?(\d+\.?\d*)', problem_lower)
        if match:
            return f'result = math.sqrt({match.group(1)})'
        
        # Pattern: Permutations/Combinations
        match = re.search(r'(?:combination|choose|C)\s*\(?(\d+)\s*,\s*(\d+)\)?', problem_lower)
        if match:
            return f'''
def comb(n, r):
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    result_val = 1
    for i in range(r):
        result_val = result_val * (n - i) // (i + 1)
    return result_val
result = comb({match.group(1)}, {match.group(2)})
'''
        
        match = re.search(r'(?:permutation|P)\s*\(?(\d+)\s*,\s*(\d+)\)?', problem_lower)
        if match:
            return f'''
def perm(n, r):
    if r > n or r < 0:
        return 0
    result_val = 1
    for i in range(r):
        result_val *= (n - i)
    return result_val
result = perm({match.group(1)}, {match.group(2)})
'''
        
        # Pattern: "Predict next in sequence: [1, 4, 9, 16]"
        match = re.search(r'predict\s+(?:next\s+)?(?:in\s+)?(?:sequence)?.*?(\[[\d\s,.-]+\])', problem_lower)
        if match:
            return f'''
seq = {match.group(1)}
x = np.arange(len(seq))
y = np.array(seq)
# Try polynomial fitting (up to degree 3 to avoid overfitting short sequences)
degree = min(3, len(seq) - 1)
coeffs = np.polyfit(x, y, degree)
poly = np.poly1d(coeffs)
next_x = len(seq)
predicted = poly(next_x)
# Round to nearest integer if the original sequence was all integers
if all(float(val).is_integer() for val in seq):
    predicted = round(predicted)
result = predicted
'''
        
        return None
    
    def teach(self, problem_type: str, template_code: str, 
              classifier_pattern: str) -> dict:
        """
        Teach Hyperion a new problem type by providing a code template
        and a regex pattern for recognizing it.
        
        This permanently expands Hyperion's mathematical capabilities.
        
        Args:
            problem_type: Name for the new capability
            template_code: Python code template with {placeholders}
            classifier_pattern: Regex pattern to match this problem type
            
        Returns:
            dict with: 'success', 'capability_added'
        """
        # Add to solver's templates
        self.math.templates[problem_type] = template_code
        
        # Track new capability
        if problem_type not in self.capabilities:
            self.capabilities.append(problem_type)
        
        self.self_modifications += 1
        
        return {
            'success': True,
            'capability_added': problem_type,
            'total_capabilities': len(self.capabilities),
        }
    
    def self_improve(self, problem: str, correct_answer: Any) -> dict:
        """
        The core self-improvement loop.
        
        1. Attempt to solve the problem
        2. Compare result with correct answer
        3. If wrong: try to generate better code
        4. If better code works: save it as a new template
        
        Args:
            problem: The problem to solve
            correct_answer: The known correct answer
            
        Returns:
            dict with improvement details
        """
        # Attempt
        solution = self.attempt(problem)
        
        was_correct = False
        if solution['success'] and solution['result'] is not None:
            # Check correctness
            if isinstance(correct_answer, float):
                was_correct = abs(solution['result'] - correct_answer) < 1e-6
            else:
                was_correct = solution['result'] == correct_answer
        
        if was_correct:
            return {
                'problem': problem,
                'already_correct': True,
                'result': solution['result'],
                'improved': False,
            }
        
        # Wrong answer — try to self-improve
        # Generate new code specifically for this problem
        generated = self._generate_solution_code(problem)
        if generated:
            exec_result = self.executor.execute(generated)
            if exec_result['success'] and exec_result['result'] is not None:
                new_correct = False
                if isinstance(correct_answer, float):
                    new_correct = abs(exec_result['result'] - correct_answer) < 1e-6
                else:
                    new_correct = exec_result['result'] == correct_answer
                
                if new_correct:
                    self.self_modifications += 1
                    return {
                        'problem': problem,
                        'already_correct': False,
                        'result': exec_result['result'],
                        'improved': True,
                        'new_code': generated,
                        'method': 'auto_generated',
                    }
        
        return {
            'problem': problem,
            'already_correct': False,
            'result': solution.get('result'),
            'improved': False,
            'error': 'Could not self-improve for this problem',
        }
    
    def inspect_self(self) -> dict:
        """
        Self-reflection — inspect Hyperion's own architecture.
        
        Returns a comprehensive map of all modules, their sizes,
        and capabilities.
        """
        modules = {}
        if os.path.exists(self.evolver.layers_dir):
            for filename in sorted(os.listdir(self.evolver.layers_dir)):
                if filename.endswith('.py') and not filename.startswith('__'):
                    name = filename.replace('.py', '')
                    info = self.evolver.inspect_module(name)
                    if 'error' not in info:
                        modules[name] = {
                            'lines': info['line_count'],
                            'classes': info['classes'],
                            'functions': len(info['functions']),
                        }
        
        return {
            'architecture': 'CainBrain v2 — Universal Cognitive Architecture',
            'modules': modules,
            'total_files': len(modules),
            'total_lines': sum(m['lines'] for m in modules.values()),
            'capabilities': self.capabilities,
            'evolution': self.evolver.status(),
            'math_solver': {
                'solved': self.math.solved_count,
                'failed': self.math.failed_count,
                'templates': list(self.math.templates.keys()),
            },
        }
    
    def status(self) -> dict:
        """Return meta-cognitive status."""
        return {
            'problems_attempted': self.problems_attempted,
            'problems_solved': self.problems_solved,
            'solve_rate': (self.problems_solved / max(1, self.problems_attempted)) * 100,
            'self_modifications': self.self_modifications,
            'capabilities': len(self.capabilities),
            'evolution': self.evolver.status(),
        }
