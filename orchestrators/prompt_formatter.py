import logging

logger = logging.getLogger(__name__)

class PromptFormatter:
    """
    Responsible for formatting prompts for different models
    """
    
    def format_task_prompt(self, task, model_name):
        """
        Format prompt for a task based on the model being used
        
        Args:
            task (dict): Task document
            model_name (str): Name of the model
            
        Returns:
            str: Formatted prompt
        """
        text = task.get('text', '')
        test_list = task.get('test_list', [])
        test_setup_code = task.get('test_setup_code', '')
        
        prompt = f"Write a Python function to solve the following problem:\n\n{text}\n\n"
        
        if test_list:
            prompt += "Here are some examples of expected behavior:\n\n"
            for test in test_list:
                prompt += f"Test: {test}\n"
            prompt += "\n"
        
        if test_setup_code:
            prompt += f"Test setup code:\n{test_setup_code}\n\n"
        
        prompt += "Please provide a correct implementation of the function. Make sure to include docstrings and comments to explain your approach.\n\n"
        prompt += "IMPORTANT: Format your code solution inside a Python code block using triple backticks like this:\n"
        prompt += "```python\n"
        prompt += "# Your solution here\n"
        prompt += "```\n"
        prompt += "This precise formatting is required for proper evaluation of your solution."
        
        return prompt
    
    def format_enhanced_prompt(self, task, model_name, previous_solution, error_message, level, branch_index):
        """
        Format an enhanced prompt with previous solution and error information
        
        Args:
            task (dict): Task document
            model_name (str): Name of the model
            previous_solution (str): Previous failed solution
            error_message (str): Error message from previous attempt
            level (int): Branching level
            branch_index (int): Index of this branch within the level
            
        Returns:
            str: Enhanced prompt for the model
        """
        base_prompt = self.format_task_prompt(task, model_name)
        
        enhanced_prompt = f"{base_prompt}\n\n"
        enhanced_prompt += "A previous attempt at solving this problem failed. Here is the attempted solution:\n\n"
        
        if previous_solution:
            enhanced_prompt += f"```python\n{previous_solution}\n```\n\n"
        
        enhanced_prompt += f"The error encountered was: {error_message}\n\n"
        
        strategies = [
            "Try a completely different approach to solve this problem.",
            "Correct the logical errors in the previous solution.",
            "Simplify the solution and focus on handling edge cases properly.",
            "Rewrite the solution with better error handling.",
            "Use a more efficient algorithm to solve this problem.",
            "Start from scratch with a clearer implementation.",
            "Fix syntax errors and improve the code structure.",
            "Implement the solution step by step with careful testing."
        ]
        
        strategy_index = branch_index % len(strategies)
        enhanced_prompt += f"Instructions for this attempt: {strategies[strategy_index]}\n\n"
        
        enhanced_prompt += "IMPORTANT: Format your solution as Python code inside triple backticks like this:\n"
        enhanced_prompt += "```python\n"
        enhanced_prompt += "# Your solution here\n"
        enhanced_prompt += "```\n"
        enhanced_prompt += "This precise formatting is required for proper evaluation of your solution."
        
        return enhanced_prompt