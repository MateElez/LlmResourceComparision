import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultProcessor:
    """
    Responsible for processing results and creating summary reports
    """
    
    def __init__(self, mongo_client):
        """
        Initialize result processor
        
        Args:
            mongo_client: MongoDB client for storing results
        """
        self.mongo_client = mongo_client
    
    def create_task_result(self, task):
        """
        Create an initial result object for a task
        
        Args:
            task (dict): Task document
            
        Returns:
            dict: Initial result object
        """
        task_id = task.get('task_id', 'unknown')
        
        return {
            "task_id": task_id,
            "task_text": task.get('text', ''),
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "summary": {}
        }
    
    def create_failed_result(self, task, error_message):
        """
        Create a result object for a failed task with all metrics set to 0
        
        Args:
            task (dict): Task document
            error_message (str): Error message
            
        Returns:
            dict: Failed result object
        """
        task_id = task.get('task_id', 'unknown')
        
        zero_resources = {
            "avg_cpu_percent": 0,
            "avg_memory_mb": 0,
            "max_memory_mb": 0,
            "duration_seconds": 0
        }
        
        failed_evaluation = {
            "success": False,
            "explanation": f"Task processing failed: {error_message}"
        }
        
        return {
            "task_id": task_id,
            "task_text": task.get('text', ''),
            "timestamp": datetime.now().isoformat(),
            "models": {
                "large_model": {
                    "name": "codellama",
                    "resources": zero_resources,
                    "solution": None,
                    "evaluation": failed_evaluation,
                    "error": error_message
                },
                "small_model": {
                    "small_model": {
                        "name": "tinyllama",
                        "resources": zero_resources,
                        "solution": None,
                        "evaluation": failed_evaluation,
                        "error": error_message
                    }
                }
            },
            "summary": {
                "large_model_success": False,
                "small_model_success": False,
                "overall_success": False
            }
        }
    
    def process_large_model_result(self, result, large_model_result):
        """
        Process and integrate large model result into the overall result
        
        Args:
            result (dict): Overall result object to update
            large_model_result (dict): Large model result to integrate
            
        Returns:
            dict: Updated result
        """
        if large_model_result:
            result["models"]["large_model"] = large_model_result
            
            if large_model_result.get("evaluation", {}).get("success", False):
                result["summary"]["large_model_success"] = True
            else:
                result["summary"]["large_model_success"] = False
                
                if "resources" not in large_model_result or large_model_result["resources"] is None:
                    large_model_result["resources"] = {
                        "avg_cpu_percent": 0,
                        "avg_memory_mb": 0,
                        "max_memory_mb": 0,
                        "duration_seconds": 0
                    }
        
        return result
    
    def process_small_model_result(self, result, small_model_result):
        """
        Process and integrate small model results into the overall result
        
        Args:
            result (dict): Overall result object to update
            small_model_result (dict): Small model results to integrate
            
        Returns:
            dict: Updated result
        """
        if small_model_result:
            result["models"]["small_model"] = small_model_result
            
            succeeded_models = []
            
            # Provjerimo je li small_model_result rječnik
            if isinstance(small_model_result, dict):
                # Iteriramo kroz sve ključeve osim posebnih
                for model, data in small_model_result.items():
                    if model != "average_resources" and model != "error":
                        # Provjerimo je li data rječnik prije pristupa .get metodi
                        if isinstance(data, dict):
                            # Sigurno pristupamo .get metodi na rječniku
                            evaluation = data.get("evaluation", {})
                            if isinstance(evaluation, dict) and evaluation.get("success", False):
                                succeeded_models.append(model)
                            
                            # Provjera i inicijalizacija resources ako je potrebno
                            if "resources" not in data or data["resources"] is None:
                                data["resources"] = {
                                    "avg_cpu_percent": 0,
                                    "avg_memory_mb": 0,
                                    "max_memory_mb": 0,
                                    "duration_seconds": 0
                                }
            
            result["summary"]["small_model_success"] = len(succeeded_models) > 0
            result["summary"]["successful_small_models"] = succeeded_models
        
        return result
    
    def finalize_result(self, result):
        """
        Finalize the result by adding summary information
        
        Args:
            result (dict): Result object to finalize
            
        Returns:
            dict: Finalized result
        """
        result["summary"]["overall_success"] = (
            result["summary"].get("large_model_success", False) or 
            result["summary"].get("small_model_success", False)
        )
        
        return result
    
    def save_result(self, result):
        """
        Save result to MongoDB
        
        Args:
            result (dict): Result to save
            
        Returns:
            str: ID of the saved result
        """
        result_id = self.mongo_client.save_evaluation_result(result)
        logger.info(f"Saved evaluation result with ID: {result_id}")
        return result_id
    
    def print_summary(self, result):
        """
        Print a human-readable summary of the results
        
        Args:
            result (dict): Result to summarize
        """
        logger.info("\n--- EVALUATION RESULTS ---")
        
        if result["summary"].get("large_model_success", False):
            logger.info("✅ Large model (CodeLlama) succeeded!")
            large_model = result["models"]["large_model"]
            solution = large_model.get("solution", "")
            
            truncated_solution = solution[:300] + "..." if len(solution) > 300 else solution
            logger.info(f"Solution (truncated):\n{truncated_solution}")
            
            if large_model.get("resources"):
                res = large_model["resources"]
                logger.info(f"Resources: CPU: {res.get('avg_cpu_percent', 0):.2f}%, Memory: {res.get('avg_memory_mb', 0):.2f}MB")
        else:
            logger.info("❌ Large model (CodeLlama) failed")
        
        if result["summary"].get("small_model_success", False):
            logger.info("✅ Small model (TinyLlama) succeeded!")
            
            small_models = result["models"]["small_model"]
            successful_branches = result["summary"].get("successful_small_models", [])
            
            if successful_branches:
                successful_branch = successful_branches[0]
                if successful_branch in small_models:
                    branch_data = small_models[successful_branch]
                    solution = branch_data.get("solution", "")
                    
                    truncated_solution = solution[:300] + "..." if len(solution) > 300 else solution
                    logger.info(f"Solution in branch {successful_branch} (truncated):\n{truncated_solution}")
                    
                    if branch_data.get("resources"):
                        res = branch_data["resources"]
                        logger.info(f"Resources: CPU: {res.get('avg_cpu_percent', 0):.2f}%, Memory: {res.get('avg_memory_mb', 0):.2f}MB")
        else:
            logger.info("❌ Small model (TinyLlama) failed")
        
        logger.info(f"Overall success: {'✅ Yes' if result['summary'].get('overall_success', False) else '❌ No'}")
        logger.info("------------------------\n")