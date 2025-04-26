"""
Glavni koordinatorski modul koji upravlja tijekom rada evaluacije MBPP zadataka s različitim modelima.
"""

import logging
import asyncio
import traceback

from utils.ollama_resource_tracker import OllamaResourceTracker
from orchestrators.model_manager import ModelManager
from orchestrators.prompt_formatter import PromptFormatter
from orchestrators.solution_evaluator import SolutionEvaluator
from orchestrators.resource_manager import ResourceManager
from orchestrators.branching_strategy import BranchingStrategy
from orchestrators.result_processor import ResultProcessor
from orchestrators.daytona_sandbox_manager import DaytonaSandboxManager
from orchestrators.model_execution_manager import ModelExecutionManager

logger = logging.getLogger(__name__)

class WorkflowManager:
    """
    Glavni koordinatorski razred koji upravlja tijekom rada za evaluaciju MBPP zadataka
    """
    
    def __init__(self, mongo_client, config):
        """
        Inicijalizacija workflow managera i komponenti
        
        Args:
            mongo_client: Instanca MongoDB klijenta
            config (dict): Konfiguracijski rječnik
        """
        self.mongo_client = mongo_client
        self.config = config
        
        # Inicijalizacija osnovnih komponenti
        self.ollama_tracker = OllamaResourceTracker()
        self.model_manager = ModelManager(config)
        self.prompt_formatter = PromptFormatter()
        self.solution_evaluator = SolutionEvaluator()
        self.resource_manager = ResourceManager(config, self.ollama_tracker)
        self.result_processor = ResultProcessor(mongo_client)
        
        # Inicijalizacija novih komponenti
        self.sandbox_manager = DaytonaSandboxManager()
        self.branching_strategy = BranchingStrategy(self.sandbox_manager, max_branching_level=3)
        self.model_execution_manager = ModelExecutionManager(
            self.model_manager,
            self.resource_manager,
            self.sandbox_manager,
            self.prompt_formatter,
            self.solution_evaluator
        )
        
        logger.info("Workflow manager inicijaliziran")
    
    def import_mbpp_data(self, file_path):
        """
        Uvoz MBPP podataka u MongoDB
        
        Args:
            file_path (str): Put do MBPP dataset datoteke
            
        Returns:
            int: Broj uvezenih zadataka
        """
        return self.mongo_client.import_mbpp_data(file_path)
    
    async def run_pipeline(self):
        """
        Pokreni evaluacijski pipeline za sve zadatke
        """
        logger.info("Pokrećem evaluacijski pipeline")
        
        await self.model_manager.download_models()
        
        tasks = self.mongo_client.get_all_tasks()
        logger.info(f"Pronađeno {len(tasks)} zadataka za evaluaciju")
        
        for i, task in enumerate(tasks):
            logger.info(f"Obrađujem zadatak {i+1}/{len(tasks)}: {task.get('task_id', 'unknown')}")
            
            try:
                result = await self.process_task(task)
                
                if result:
                    self.result_processor.save_result(result)
            except Exception as e:
                logger.error(f"Greška pri obradi zadatka: {str(e)}")
                logger.error(traceback.format_exc())
                
                failed_result = self.result_processor.create_failed_result(task, str(e))
                self.result_processor.save_result(failed_result)
        
        logger.info("Evaluacijski pipeline završen")
    
    async def process_task(self, task):
        """
        Obradi jedan MBPP zadatak kroz cijeli evaluacijski tijek rada
        
        Args:
            task (dict): Dokument zadatka iz MongoDB-a
            
        Returns:
            dict: Rezultati evaluacije
        """
        task_id = task.get('task_id', 'unknown')
        logger.info(f"Obrađujem zadatak: {task_id}")
        
        result = self.result_processor.create_task_result(task)
        
        # Izvršavanje velikog modela
        large_model_result = await self.model_execution_manager.execute_large_model(task)
        result = self.result_processor.process_large_model_result(result, large_model_result)
        
        # Izvršavanje malog modela s fallback strategijom
        small_model_result = await self.run_small_model_with_fallbacks(task)
        result = self.result_processor.process_small_model_result(result, small_model_result)
        
        result = self.result_processor.finalize_result(result)
        
        return result
    
    async def run_small_model_with_fallbacks(self, task):
        """
        Izvršava mali model (TinyLlama via Ollama) na zadatku, s eksponencijalnim grananjem ako ne uspije
        
        Args:
            task (dict): Dokument zadatka
            
        Returns:
            dict: Rezultati za sve male modele koji su pokušani
        """
        logger.info("Izvršavam mali model (TinyLlama via Ollama) s eksponencijalnim grananjem")
        
        model = self.model_manager.get_small_model()
        
        if not self.model_manager.check_model_availability(model):
            logger.warning(f"Ollama model {model.model_name} nije dostupan")
            return {"error": f"Ollama model {model.model_name} nije dostupan"}
        
        task_prompt = self.prompt_formatter.format_task_prompt(task, model.model_name)
        
        logger.info(f"Započinjem prvi model: {model.model_name}")
        first_result = await self.model_execution_manager.execute_single_model(model, task, task_prompt)
        
        results, all_resource_stats = await self.branching_strategy.execute_branching(
            model, 
            task, 
            self.prompt_formatter, 
            self.model_execution_manager.execute_single_model, 
            first_result
        )
        
        if all_resource_stats:
            avg_resources = self.resource_manager.calculate_average_resources(all_resource_stats)
            results["average_resources"] = avg_resources
            logger.info(f"Prosječna potrošnja resursa: CPU: {avg_resources['avg_cpu_percent']:.2f}%, "
                       f"Memorija: {avg_resources['avg_memory_mb']:.2f}MB")
        
        return results
    
    def run_single_task(self, task):
        """
        Pokreni evaluacijski pipeline za jedan zadatak
        
        Args:
            task (dict): Dokument zadatka za obradu
        """
        task_id = task.get('task_id', 'unknown')
        logger.info(f"Izvršavam pojedinačni zadatak: {task_id}")
        logger.info(f"Opis zadatka: {task.get('text', '')}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.model_manager.download_models())
            
            result = loop.run_until_complete(self.process_task(task))
            
            if result:
                self.result_processor.save_result(result)
                self.result_processor.print_summary(result)
                return result
        except Exception as e:
            logger.error(f"Greška pri obradi zadatka: {str(e)}")
            logger.error(traceback.format_exc())
            
            failed_result = self.result_processor.create_failed_result(task, str(e))
            self.result_processor.save_result(failed_result)
        finally:
            loop.close()