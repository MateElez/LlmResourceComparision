"""
Upravlja izvršavanjem LLM modela, praćenjem resursa i obradom rezultata.
"""

import logging
from datetime import datetime
import time
import asyncio

logger = logging.getLogger(__name__)

class ModelExecutionManager:
    """
    Upravlja izvršavanjem LLM modela, praćenjem resursa i procesiranjem rezultata
    """
    
    def __init__(self, model_manager, resource_manager, prompt_formatter, solution_evaluator):
        """
        Inicijalizira manager za izvršavanje modela
        
        Args:
            model_manager: Manager za LLM modele
            resource_manager: Manager za praćenje resursa
            prompt_formatter: Formatter za promptove
            solution_evaluator: Evaluator za generirana rješenja
        """
        self.model_manager = model_manager
        self.resource_manager = resource_manager
        self.prompt_formatter = prompt_formatter
        self.solution_evaluator = solution_evaluator
        
        logger.info("ModelExecutionManager: Konfiguriran za lokalno izvršavanje modela")
        
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
            
            # 1. Započinjemo praćenje resursa
            task_id = task.get('task_id', 'unknown')
            logger.info(f"Započinjem praćenje resursa za veliki model {model_name}")
            session_id = await self.resource_manager.track_model_resources(model, task_id)
            
            # 2. Pripremamo prompt i izvršavamo model
            task_prompt = self.prompt_formatter.format_task_prompt(task, model_name)
            
            execution_start_time = time.time()
            
            # Izvršavamo model lokalno
            logger.info(f"Izvršavanje {model_name} modela LOKALNO")
            solution = await model.generate(task_prompt)
            logger.info(f"Lokalno izvršavanje velikog modela završeno")
            
            execution_duration = time.time() - execution_start_time
            logger.info(f"Izvršavanje velikog modela {model_name} trajalo: {execution_duration:.2f} sekundi")
                
            result["solution"] = solution
            
            # 3. Prikupljamo podatke o korištenim resursima
            if session_id is not None:
                logger.info(f"Prikupljam statistiku o korištenim resursima za veliki model")
                # Osiguravamo da praćenje traje dovoljno dugo kako bismo dobili dobre podatke
                await asyncio.sleep(2.0)
                
                # Sada prikupljamo statistiku resursa (s novim, dužim timeoutom)
                resource_stats = await self.resource_manager.get_resource_stats(
                    session_id, model, task_id
                )
                
                # Ako statistika ima 0 uzoraka, pokušajmo procijeniti vrijednosti na temelju trajanja izvršavanja
                if resource_stats.get("samples", 0) == 0:
                    logger.warning(f"Nisu prikupljeni uzorci za veliki model, procjenjujem resurse na temelju trajanja")
                    # Procjena CPU i memorije na temelju trajanja izvršavanja
                    # Veći model troši više resursa
                    if execution_duration > 60:
                        estimated_cpu = 40.0
                        estimated_memory = 4000.0
                    elif execution_duration > 30:
                        estimated_cpu = 35.0
                        estimated_memory = 3500.0
                    elif execution_duration > 10:
                        estimated_cpu = 30.0
                        estimated_memory = 3000.0
                    else:
                        estimated_cpu = 25.0
                        estimated_memory = 2500.0
                        
                    resource_stats = {
                        "avg_cpu_percent": estimated_cpu,
                        "avg_memory_mb": estimated_memory,
                        "max_memory_mb": estimated_memory * 1.2,
                        "duration_seconds": execution_duration,
                        "samples": 0,
                        "estimated": True,
                        "note": f"Resources estimated based on execution duration: {execution_duration:.2f}s"
                    }
                
                result["resources"] = resource_stats
                logger.info(f"Resursi za veliki model: CPU: {resource_stats.get('avg_cpu_percent', 0):.2f}%, Memorija: {resource_stats.get('avg_memory_mb', 0):.2f}MB")
            else:
                logger.warning(f"Nije bilo moguće pratiti resurse za veliki model, koristim procijenjene vrijednosti")
                result["resources"] = {
                    "avg_cpu_percent": 30.0,
                    "avg_memory_mb": 3000.0,
                    "max_memory_mb": 3600.0,
                    "duration_seconds": execution_duration,
                    "samples": 0,
                    "estimated": True,
                    "note": "Resources estimated (no tracking session available)"
                }
            
            # 4. Evaluacija rješenja
            if solution:
                logger.info(f"Evaluacija rješenja za {model_name} lokalno")
                result["evaluation"] = self.solution_evaluator.evaluate_solution(task, solution)
                
                if result["evaluation"]["success"]:
                    logger.info(f"Rješenje velikog modela {model_name} uspješno validirano lokalno")
                else:
                    logger.warning(f"Rješenje velikog modela {model_name} nije prošlo lokalnu validaciju: {result['evaluation']['explanation']}")
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
            
            # 1. Započnemo praćenje resursa
            logger.info(f"Započinjem praćenje resursa za model {branch_name}")
            session_id = await self.resource_manager.track_model_resources(model, task_id, branch_name)
            
            # 2. Izvršavamo model i mjerimo vrijeme
            model_name = model.model_name if hasattr(model, 'model_name') else "default"
            
            execution_start_time = time.time()
            
            # Izvršavamo model lokalno
            logger.info(f"Izvršavanje {model_name} modela LOKALNO za {branch_name}")
            solution = await model.generate(task_prompt)
            logger.info(f"Lokalno izvršavanje modela za {branch_name} završeno")
            
            execution_duration = time.time() - execution_start_time
            logger.info(f"Izvršavanje modela za {branch_name} trajalo: {execution_duration:.2f} sekundi")
                
            result["solution"] = solution
            
            # 3. Tek nakon što model završi s izvršavanjem prikupljamo statistike resursa
            if session_id is not None:
                logger.info(f"Prikupljam statistiku o korištenim resursima za {branch_name}")
                # Osiguravamo da praćenje traje dovoljno dugo kako bismo dobili dobre podatke
                wait_start = time.time()
                
                # Čekamo kraće vrijeme kako bismo omogućili da uzorci budu prikupljeni
                await asyncio.sleep(2.0)
                
                # Sada prikupljamo statistiku resursa - ovo može trajati do 60 sekundi zbog novog timeoutu
                resource_stats = await self.resource_manager.get_resource_stats(
                    session_id, model, task_id, branch_name
                )
                
                # Ako statistika ima 0 uzoraka, pokušajmo procijeniti vrijednosti na temelju trajanja izvršavanja
                if resource_stats.get("samples", 0) == 0:
                    logger.warning(f"Nisu prikupljeni uzorci za {branch_name}, procjenjujem resurse na temelju trajanja")
                    # Procjena CPU i memorije na temelju trajanja izvršavanja
                    # Ovo su aproksimativne vrijednosti bazirane na empirijskim podacima
                    if execution_duration > 60:
                        estimated_cpu = 25.0
                        estimated_memory = 1200.0
                    elif execution_duration > 30:
                        estimated_cpu = 20.0
                        estimated_memory = 800.0
                    elif execution_duration > 10:
                        estimated_cpu = 15.0
                        estimated_memory = 600.0
                    else:
                        estimated_cpu = 10.0
                        estimated_memory = 400.0
                        
                    resource_stats = {
                        "avg_cpu_percent": estimated_cpu,
                        "avg_memory_mb": estimated_memory,
                        "max_memory_mb": estimated_memory * 1.5,
                        "duration_seconds": execution_duration,
                        "samples": 0,
                        "estimated": True,
                        "note": f"Resources estimated based on execution duration: {execution_duration:.2f}s"
                    }
                    
                result["resources"] = resource_stats
                logger.info(f"Resursi za {branch_name}: CPU: {resource_stats.get('avg_cpu_percent', 0):.2f}%, Memorija: {resource_stats.get('avg_memory_mb', 0):.2f}MB")
            else:
                logger.warning(f"Nije bilo moguće pratiti resurse za {branch_name}, koristim procijenjene vrijednosti")
                result["resources"] = {
                    "avg_cpu_percent": 15.0,
                    "avg_memory_mb": 600.0,
                    "max_memory_mb": 900.0,
                    "duration_seconds": execution_duration,
                    "samples": 0,
                    "estimated": True,
                    "note": "Resources estimated (no tracking session available)"
                }
            
            # 4. Evaluiramo rješenje
            if solution:
                logger.info(f"Evaluacija rješenja za {branch_name} lokalno")
                result["evaluation"] = self.solution_evaluator.evaluate_solution(task, solution)
                
                if result["evaluation"]["success"]:
                    logger.info(f"Rješenje za {branch_name} uspješno validirano lokalno")
                else:
                    logger.warning(f"Rješenje za {branch_name} nije prošlo lokalnu validaciju: {result['evaluation']['explanation']}")
            else:
                result["evaluation"] = {"success": False, "explanation": "No solution generated"}
            
        except Exception as e:
            logger.error(f"Greška pri izvršavanju modela {branch_name}: {str(e)}")
            result["error"] = str(e)
        finally:
            result["end_time"] = datetime.now().isoformat()
        
        return result