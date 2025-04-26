"""
Upravlja izvršavanjem LLM modela, praćenjem resursa i obradom rezultata.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelExecutionManager:
    """
    Upravlja izvršavanjem LLM modela, praćenjem resursa i procesiranjem rezultata
    """
    
    def __init__(self, model_manager, resource_manager, sandbox_manager, prompt_formatter, solution_evaluator):
        """
        Inicijalizira manager za izvršavanje modela
        
        Args:
            model_manager: Manager za LLM modele
            resource_manager: Manager za praćenje resursa
            sandbox_manager: Manager za Daytona sandbox
            prompt_formatter: Formatter za promptove
            solution_evaluator: Evaluator za generirana rješenja
        """
        self.model_manager = model_manager
        self.resource_manager = resource_manager
        self.sandbox_manager = sandbox_manager
        self.prompt_formatter = prompt_formatter
        self.solution_evaluator = solution_evaluator
        
    async def execute_large_model(self, task):
        """
        Izvršava veliki model (CodeLlama) za zadani zadatak
        
        Args:
            task (dict): Zadatak iz baze podataka
            
        Returns:
            dict: Rezultat izvršavanja velikog modela
        """
        logger.info("Izvršavam veliki model (CodeLlama) za zadatak")
        
        result = {
            "name": "codellama",
            "start_time": datetime.now().isoformat(),
            "resources": None,
            "solution": None,
            "evaluation": None,
            "error": None
        }
        
        try:
            model = self.model_manager.get_large_model()
            model_name = model.model_name
            
            if not self.model_manager.check_model_availability(model):
                logger.warning(f"Ollama model {model_name} nije dostupan")
                result["error"] = f"Ollama model {model_name} nije dostupan"
                return result
            
            task_id = task.get('task_id', 'unknown')
            session_id = await self.resource_manager.track_model_resources(model, task_id)
            
            task_prompt = self.prompt_formatter.format_task_prompt(task, model_name)
            
            logger.info(f"Izvršavanje {model_name} modela u Daytona sandboxu za generiranje rješenja")
            sandbox_solution = await self.sandbox_manager.execute_model_in_sandbox(
                task_prompt, 
                model_name
            )
            
            if sandbox_solution is not None:
                logger.info(f"Uspješno generirano rješenje u Daytona sandboxu")
                solution = sandbox_solution
            else:
                logger.warning(f"Sandbox generiranje nije uspjelo, pokušavam lokalno izvršavanje modela")
                solution = await model.generate(task_prompt)
                
            result["solution"] = solution
            
            result["resources"] = await self.resource_manager.get_resource_stats(
                session_id, model, task_id
            )
            
            if solution:
                logger.info(f"Evaluacija rješenja u Daytona sandboxu")
                is_valid = await self.sandbox_manager.validate_in_sandbox(
                    solution, 
                    task.get("test_list", []),
                    task.get("test_setup_code", "")
                )
                
                if is_valid:
                    logger.info(f"Rješenje velikog modela {model_name} uspješno validirano u Daytona sandboxu")
                    result["evaluation"] = {
                        "success": True, 
                        "explanation": "All test cases passed successfully in Daytona sandbox"
                    }
                else:
                    logger.warning(f"Rješenje velikog modela {model_name} nije prošlo validaciju u Daytona sandboxu")
                    result["evaluation"] = {
                        "success": False,
                        "explanation": "Solution failed validation in Daytona sandbox"
                    }
                    
                if result["evaluation"] is None:
                    logger.warning("Fallback na lokalnu evaluaciju")
                    result["evaluation"] = self.solution_evaluator.evaluate_solution(task, solution)
            else:
                result["evaluation"] = {"success": False, "explanation": "No solution generated"}
            
        except Exception as e:
            logger.error(f"Greška pri izvršavanju velikog modela: {str(e)}")
            result["error"] = str(e)
        finally:
            result["end_time"] = datetime.now().isoformat()
        
        return result
        
    async def execute_single_model(self, model, task, task_prompt, branch_name="small_model"):
        """
        Izvršava pojedinačni model i prati njegove resurse
        
        Args:
            model: Instanca modela za izvršavanje
            task (dict): Zadatak iz baze podataka
            task_prompt (str): Prompt za zadatak
            branch_name (str): Naziv za ovu granu modela
            
        Returns:
            dict: Rezultat izvršavanja modela
        """
        result = {
            "name": branch_name,
            "start_time": datetime.now().isoformat(),
            "resources": None,
            "solution": None,
            "evaluation": None,
            "error": None
        }
        
        try:
            task_id = task.get('task_id', 'unknown')
            session_id = await self.resource_manager.track_model_resources(model, task_id, branch_name)
            
            model_name = model.model_name if hasattr(model, 'model_name') else "default"
            logger.info(f"Izvršavanje {model_name} modela u Daytona sandboxu za generiranje rješenja")
            sandbox_solution = await self.sandbox_manager.execute_model_in_sandbox(
                task_prompt, 
                model_name
            )
            
            if sandbox_solution is not None:
                logger.info(f"Uspješno generirano rješenje u Daytona sandboxu za {branch_name}")
                solution = sandbox_solution
            else:
                logger.warning(f"Sandbox generiranje nije uspjelo za {branch_name}, pokušavam lokalno izvršavanje modela")
                solution = await model.generate(task_prompt)
                
            result["solution"] = solution
            
            result["resources"] = await self.resource_manager.get_resource_stats(
                session_id, model, task_id, branch_name
            )
            
            if solution:
                logger.info(f"Evaluacija rješenja za {branch_name} u Daytona sandboxu")
                is_valid = await self.sandbox_manager.validate_in_sandbox(
                    solution, 
                    task.get("test_list", []),
                    task.get("test_setup_code", "")
                )
                
                if is_valid:
                    logger.info(f"Rješenje za {branch_name} uspješno validirano u Daytona sandboxu")
                    result["evaluation"] = {
                        "success": True, 
                        "explanation": "All test cases passed successfully in Daytona sandbox"
                    }
                else:
                    logger.warning(f"Rješenje za {branch_name} nije prošlo validaciju u Daytona sandboxu")
                    result["evaluation"] = {
                        "success": False,
                        "explanation": "Solution failed validation in Daytona sandbox"
                    }
                    
                if result["evaluation"] is None:
                    logger.warning(f"Fallback na lokalnu evaluaciju za {branch_name}")
                    result["evaluation"] = self.solution_evaluator.evaluate_solution(task, solution)
            else:
                result["evaluation"] = {"success": False, "explanation": "No solution generated"}
            
        except Exception as e:
            logger.error(f"Greška pri izvršavanju modela {branch_name}: {str(e)}")
            result["error"] = str(e)
        finally:
            result["end_time"] = datetime.now().isoformat()
        
        return result