import logging
import re

logger = logging.getLogger(__name__)

class SolutionEvaluator:
    """
    Responsible for evaluating code solutions against test cases
    """
    
    def evaluate_solution(self, task, solution):
        """
        Evaluation implementation that compiles and executes the solution with test cases
        
        Args:
            task (dict): Task document
            solution (str): Generated solution
            
        Returns:
            dict: Evaluation result
        """
        code = self._extract_code(solution)
            
        try:
            compile(code, "<string>", "exec")
            
            test_list = task.get('test_list', [])
            if test_list:
                local_namespace = {}
                global_namespace = {"__builtins__": __builtins__}
                
                try:
                    exec(code, global_namespace, local_namespace)
                except Exception as e:
                    return {
                        "success": False, 
                        "explanation": f"Solution execution failed: {str(e)}", 
                        "error_type": "execution_error"
                    }
                
                test_setup_code = task.get('test_setup_code', '')
                if test_setup_code:
                    try:
                        exec(test_setup_code, global_namespace, local_namespace)
                    except Exception as e:
                        return {
                            "success": False, 
                            "explanation": f"Test setup code execution failed: {str(e)}", 
                            "error_type": "test_setup_error"
                        }
                
                test_results = []
                for test_idx, test in enumerate(test_list):
                    try:
                        exec(test, global_namespace, local_namespace)
                        test_results.append({
                            "test_idx": test_idx,
                            "test": test,
                            "success": True
                        })
                    except AssertionError as e:
                        test_results.append({
                            "test_idx": test_idx,
                            "test": test,
                            "success": False,
                            "error": f"Assertion failed: {str(e) if str(e) else 'assertion error'}"
                        })
                        return {
                            "success": False, 
                            "explanation": f"Test case {test_idx} failed: {test}", 
                            "error_type": "test_failure",
                            "test_results": test_results
                        }
                    except Exception as e:
                        test_results.append({
                            "test_idx": test_idx,
                            "test": test,
                            "success": False,
                            "error": str(e)
                        })
                        return {
                            "success": False, 
                            "explanation": f"Test case {test_idx} execution error: {str(e)}", 
                            "error_type": "test_execution_error",
                            "test_results": test_results
                        }
                
                return {
                    "success": True, 
                    "explanation": f"All {len(test_results)} test cases passed successfully",
                    "test_results": test_results
                }
            else:
                return {"success": True, "explanation": "Solution compiles successfully, but no test cases were provided"}
                
        except Exception as e:
            return {"success": False, "explanation": f"Solution fails to compile: {str(e)}", "error_type": "compile_error"}
    
    def _extract_code(self, solution):
        """
        Extract code from a solution text
        
        Args:
            solution (str): Solution text, potentially containing code blocks
            
        Returns:
            str: Extracted code
        """
        code_pattern = r"```python\s*(.*?)\s*```"
        code_match = re.search(code_pattern, solution, re.DOTALL)
        
        if code_match:
            return code_match.group(1)
        else:
            return solution