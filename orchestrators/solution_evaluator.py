import logging
import re
import ast
import json

logger = logging.getLogger(__name__)

class SolutionEvaluator:
    """
    Responsible for evaluating code solutions against test cases
    """
    
    def evaluate_solution(self, task, solution):
        """
        Evaluation implementation for tasks with input/expected_output format
        
        Args:
            task (dict): Task document with 'input' and 'expected_output'
            solution (str): Generated solution
            
        Returns:
            dict: Evaluation result
        """
        code = self._extract_code(solution)
        
        # 1. Syntax validation
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {
                "success": False, 
                "explanation": f"Syntax error: {str(e)}", 
                "error_type": "syntax_error"
            }
        
        # 2. Security check
        if not self._is_safe_code(code):
            return {
                "success": False,
                "explanation": "Code contains unsafe operations",
                "error_type": "security_error"
            }
        
        # 3. Execute and test
        try:
            # Create isolated namespace
            namespace = {}
            exec(code, {"__builtins__": {}}, namespace)
            
            # Find the function
            function_name = self._extract_function_name(code)
            if not function_name or function_name not in namespace:
                return {
                    "success": False,
                    "explanation": "No valid function found in solution",
                    "error_type": "no_function_error"
                }
            
            func = namespace[function_name]
            
            # Prepare test input
            test_input = self._parse_input(task['input'])
            expected_output = self._parse_expected_output(task['expected_output'])
            
            # Execute function
            if isinstance(test_input, tuple):
                actual_output = func(*test_input)
            else:
                actual_output = func(test_input)
            
            # Compare results
            if self._compare_outputs(actual_output, expected_output):
                return {
                    "success": True,
                    "explanation": "Solution correct",
                    "actual_output": str(actual_output),
                    "expected_output": str(expected_output)
                }
            else:
                return {
                    "success": False,
                    "explanation": f"Wrong output. Expected: {expected_output}, Got: {actual_output}",
                    "error_type": "wrong_output",
                    "actual_output": str(actual_output),
                    "expected_output": str(expected_output)
                }
                
        except Exception as e:
            return {
                "success": False,
                "explanation": f"Runtime error: {str(e)}",
                "error_type": "runtime_error"
            }
    
    def _extract_function_name(self, code):
        """Extract function name from code"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            pass
        return None
    
    def _parse_input(self, input_str):
        """Parse input string to Python object"""
        try:
            return eval(input_str)
        except:
            return input_str
    
    def _parse_expected_output(self, output_str):
        """Parse expected output string to Python object"""
        try:
            return eval(output_str)
        except:
            return output_str
    
    def _compare_outputs(self, actual, expected):
        """Compare actual and expected outputs"""
        if type(actual) != type(expected):
            # Try string comparison
            return str(actual).strip() == str(expected).strip()
        
        if isinstance(expected, list):
            return sorted(actual) == sorted(expected)
        
        if isinstance(expected, (int, float)):
            return abs(actual - expected) < 1e-6
        
        return actual == expected
    
    def _extract_code(self, solution):
        """Extract code from solution text"""
        code_pattern = r"```python\s*(.*?)\s*```"
        code_match = re.search(code_pattern, solution, re.DOTALL)
        
        if code_match:
            return code_match.group(1)
        else:
            return solution