"""
Implementira strategiju za eksponencijalno grananje modela kada početni model ne uspije.
"""

import logging
import asyncio
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class BranchingStrategy:
    """
    Implementira strategiju za eksponencijalno grananje kada model ne uspije
    """
    
    def __init__(self, sandbox_manager, max_branching_level=3):
        """
        Inicijalizacija strategije grananja
        
        Args:
            sandbox_manager: Manager za Daytona sandbox
            max_branching_level (int): Maksimalna razina grananja (2^level modela)
        """
        self.max_branching_level = max_branching_level
        self.sandbox_manager = sandbox_manager
        
    async def execute_branching(self, model, task, prompt_formatter, model_executor, first_result):
        """
        Izvršava strategiju eksponencijalnog grananja
        
        Args:
            model: Model koji se koristi za sve grane
            task (dict): Zadatak za rješavanje
            prompt_formatter: Instanca PromptFormatter za kreiranje poboljšanih promptova
            model_executor: Funkcija za izvršavanje modela s promptom
            first_result (dict): Rezultat prvog izvršavanja modela
            
        Returns:
            tuple: (results_dict, resource_stats_list) - Kombinirani rezultati i statistike resursa
        """
        logger.info("Izvršavam strategiju eksponencijalnog grananja")
        
        results = {
            "small_model": first_result  
        }
        
        if first_result.get("evaluation", {}).get("success", False):
            logger.info("Prvi model je uspio, nema potrebe za grananjem")
            return results, []
        
        logger.info("Prvi model nije uspio, započinjem eksponencijalno grananje")
        
        all_resource_stats = []
        if first_result.get("resources"):
            all_resource_stats.append(first_result["resources"])
        
        first_solution = first_result.get("solution")
        first_error = first_result.get("evaluation", {}).get("explanation", "")
        model_name = model.model_name if hasattr(model, 'model_name') else "default"
        
        for level in range(1, self.max_branching_level + 1):
            num_models = 2 ** level
            logger.info(f"Razina grananja {level}: Pokrećem {num_models} modela paralelno")
            
            branch_tasks = []
            
            for i in range(num_models):
                branch_name = f"branch_{level}_{i+1}"
                
                enhanced_prompt = prompt_formatter.format_enhanced_prompt(
                    task, 
                    model_name, 
                    first_solution,
                    first_error,
                    level,
                    i
                )
                
                branch_task = asyncio.create_task(
                    model_executor(model, task, enhanced_prompt, branch_name)
                )
                branch_tasks.append(branch_task)
            
            branching_results = await asyncio.gather(*branch_tasks)
            
            for i, branch_result in enumerate(branching_results):
                branch_name = f"branch_{level}_{i+1}"
                results[branch_name] = branch_result
                
                if branch_result.get("resources"):
                    all_resource_stats.append(branch_result["resources"])
                
                if branch_result.get("evaluation", {}).get("success", False):
                    logger.info(f"Grana {branch_name} je proizvela potencijalno uspješno rješenje, validiram u Daytona sandboxu")
                    
                    is_valid = await self.sandbox_manager.validate_in_sandbox(
                        branch_result.get("solution", ""), 
                        task.get("test_list", []),
                        task.get("test_setup_code", "")
                    )
                    
                    if is_valid:
                        logger.info(f"Rješenje iz grane {branch_name} uspješno validirano u Daytona sandboxu")
                        results["successful_branch"] = branch_name
                        break
                    else:
                        logger.warning(f"Rješenje iz grane {branch_name} nije prošlo validaciju u Daytona sandboxu")
                        branch_result["evaluation"]["success"] = False
                        branch_result["evaluation"]["explanation"] += " (Neuspješna validacija u Daytona sandboxu)"
            
            if "successful_branch" in results:
                break
        
        # Vraćamo prikupljene rezultate
        return results, all_resource_stats
    
    async def execute_model_in_sandbox(self, prompt, model_name="default"):
        """
        Izvršava LLM model u Daytona sandboxu
        
        Args:
            prompt (str): Prompt za model
            model_name (str): Naziv modela
            
        Returns:
            str: Generirano rješenje
        """
        return await self.sandbox_manager.execute_model_in_sandbox(prompt, model_name)
        
    async def _validate_in_daytona_sandbox(self, solution, test_list, test_setup_code):
        """
        Validira rješenje u Daytona sandboxu
        
        Args:
            solution (str): Generirani kod rješenja
            test_list (list): Lista test slučajeva
            test_setup_code (str): Setup kod za testove
            
        Returns:
            bool: True ako je rješenje validno, False inače
        """
        return await self.sandbox_manager.validate_in_sandbox(solution, test_list, test_setup_code)