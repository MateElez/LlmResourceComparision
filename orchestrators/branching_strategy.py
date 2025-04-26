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
        
        # Ovdje ćemo kreirati sve zadatke za sve razine grananja unaprijed
        all_level_tasks = []
        
        # Prvo kreiramo sve zadatke za sve razine grananja bez čekanja
        for level in range(1, self.max_branching_level + 1):
            num_models = 2 ** level
            level_tasks = []
            
            logger.info(f"Priprema razine grananja {level}: Kreiram {num_models} zadataka za paralelno izvršavanje")
            
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
                
                # Stvaramo zadatak ali ga ne pokrećemo još
                branch_coro = model_executor(model, task, enhanced_prompt, branch_name)
                level_tasks.append((branch_name, branch_coro))
            
            all_level_tasks.append(level_tasks)
        
        # Sada izvršavamo zadatke po razinama, ali osiguravamo da je svaki zadatak unutar razine stvarno paralelan
        for level, level_tasks in enumerate(all_level_tasks, 1):
            num_models = len(level_tasks)
            logger.info(f"Razina grananja {level}: Pokrećem {num_models} modela STVARNO paralelno")
            
            # Stvaramo i odmah pokrećemo sve zadatke na ovoj razini
            running_tasks = []
            task_names = []
            
            for branch_name, branch_coro in level_tasks:
                # Stvarno pokretanje zadatka
                task = asyncio.create_task(branch_coro)
                running_tasks.append(task)
                task_names.append(branch_name)
            
            # Čekamo da se svi zadaci izvrše paralelno
            branching_results = await asyncio.gather(*running_tasks)
            
            # Obrada rezultata
            for i, branch_result in enumerate(branching_results):
                branch_name = task_names[i]
                results[branch_name] = branch_result
                
                if branch_result.get("resources"):
                    all_resource_stats.append(branch_result["resources"])
                
                if branch_result.get("evaluation", {}).get("success", False):
                    logger.info(f"Grana {branch_name} je proizvela uspješno rješenje")
                    results["successful_branch"] = branch_name
            
            # Ako je pronađeno uspješno rješenje, prekini grananje
            if "successful_branch" in results:
                logger.info(f"Pronađeno uspješno rješenje u grani {results['successful_branch']}, završavam grananje")
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