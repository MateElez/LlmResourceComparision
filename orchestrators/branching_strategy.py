"""
Implementira strategiju za eksponencijalno grananje modela kada početni model ne uspije.
"""

import logging
import asyncio

logger = logging.getLogger(__name__)

class BranchingStrategy:
    """
    Implementira strategiju za eksponencijalno grananje kada model ne uspije
    """
    
    def __init__(self, max_branching_level=3):
        """
        Inicijalizacija strategije grananja
        
        Args:
            max_branching_level (int): Maksimalna razina grananja (2^level modela)
        """        
        self.max_branching_level = max_branching_level
        
    async def execute_branching(self, model, task, prompt_formatter, model_executor, first_result):
        """
        Izvršava strategiju eksponencijalnog grananja s hijerarhijskim kontekstom
        
        Args:
            model: Model koji se koristi za sve grane
            task (dict): Zadatak za rješavanje
            prompt_formatter: Instanca PromptFormatter za kreiranje poboljšanih promptova
            model_executor: Funkcija za izvršavanje modela s promptom
            first_result (dict): Rezultat prvog izvršavanja modela
            
        Returns:
            tuple: (results_dict, resource_stats_list) - Kombinirani rezultati i statistike resursa
        """
        logger.info("Izvršavam strategiju eksponencijalnog grananja s hijerarhijskim kontekstom")
        
        results = {
            "small_model": first_result  
        }
        
        if first_result.get("evaluation", {}).get("success", False):
            logger.info("Prvi model je uspio, nema potrebe za grananjem")
            return results, []
        
        logger.info("Prvi model nije uspio, započinjem hijerarhijsko grananje")
        
        all_resource_stats = []
        if first_result.get("resources"):
            all_resource_stats.append(first_result["resources"])
        
        # Kreiranje početne historie pokušaja
        attempt_history = [{
            "solution": first_result.get("solution", ""),
            "error": first_result.get("evaluation", {}).get("explanation", ""),
            "level": 0,
            "branch": "initial"
        }]
        
        # Struktura za čuvanje rezultata po razinama - potrebno za hijerarhiju
        level_results = {0: [first_result]}
        
        # Izvršavanje grananja po razinama
        for level in range(1, self.max_branching_level + 1):
            num_models = 2 ** level
            logger.info(f"Razina grananja {level}: Priprema {num_models} modela")
            
            level_tasks = []
            current_level_results = []
            
            # Za svaki model na ovoj razini, biramo jednog od roditelja s prethodne razine
            for i in range(num_models):
                branch_name = f"branch_{level}_{i+1}"
                
                # Biramo roditelja iz prethodne razine (ciklički)
                parent_index = i % len(level_results[level - 1])
                parent_result = level_results[level - 1][parent_index]
                
                # Kreiranje povijesti pokušaja za ovaj model
                # Uključuje sve prethodne pokušaje u lančanoj hijerarhiji
                branch_history = attempt_history.copy()
                
                # Ako roditelj nije početni rezultat, dodaj ga u povijest
                if level > 1:
                    parent_solution = parent_result.get("solution", "")
                    parent_error = parent_result.get("evaluation", {}).get("explanation", "")
                    if parent_solution or parent_error:
                        branch_history.append({
                            "solution": parent_solution,
                            "error": parent_error,
                            "level": level - 1,
                            "branch": f"parent_of_{branch_name}"
                        })
                
                # Kreiranje poboljšanog prompta s kompletnom poviješću
                enhanced_prompt = prompt_formatter.format_hierarchical_prompt(
                    task,
                    model.model_name if hasattr(model, 'model_name') else "default",
                    branch_history
                )
                
                # Kreiranje task-a za izvršavanje
                branch_coro = model_executor(model, task, enhanced_prompt, branch_name)
                level_tasks.append((branch_name, branch_coro, branch_history))
            
            # Paralelno izvršavanje svih zadataka na ovoj razini
            logger.info(f"Razina grananja {level}: Pokrećem {num_models} modela paralelno")
            
            running_tasks = []
            task_names = []
            task_histories = []
            
            for branch_name, branch_coro, branch_history in level_tasks:
                task = asyncio.create_task(branch_coro)
                running_tasks.append(task)
                task_names.append(branch_name)
                task_histories.append(branch_history)
            
            # Čekamo da se svi zadaci izvrše
            branching_results = await asyncio.gather(*running_tasks)
            
            # Obrada rezultata
            for i, branch_result in enumerate(branching_results):
                branch_name = task_names[i]
                results[branch_name] = branch_result
                current_level_results.append(branch_result)
                
                if branch_result.get("resources"):
                    all_resource_stats.append(branch_result["resources"])
                
                logger.info(f"Grana {branch_name} završena. Uspjeh: {branch_result.get('evaluation', {}).get('success', False)}")
                
                if branch_result.get("evaluation", {}).get("success", False):
                    logger.info(f"Grana {branch_name} je proizvela uspješno rješenje")
                    results["successful_branch"] = branch_name
                    # Vraćamo odmah kada pronađemo uspješno rješenje
                    return results, all_resource_stats
            
            # Spremamo rezultate ove razine za sljedeću razinu
            level_results[level] = current_level_results
            
            # Ako nijedan model na ovoj razini nije uspio, nastavi na sljedeću razinu
            logger.info(f"Razina {level} završena. Nijedan model nije uspio, nastavljam na sljedeću razinu...")
        
        logger.info("Sve razine grananja završene. Nijedan model nije pronašao uspješno rješenje.")
        return results, all_resource_stats