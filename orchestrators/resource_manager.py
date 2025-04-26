import logging
import asyncio
from models.ollama_model import OllamaModel

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Responsible for tracking and managing resource usage of models
    """
    
    def __init__(self, config, ollama_tracker):
        """
        Initialize resource manager
        
        Args:
            config (dict): Configuration dictionary
            ollama_tracker: OllamaResourceTracker instance
        """
        self.config = config
        self.ollama_tracker = ollama_tracker
    
    async def track_model_resources(self, model, task_id, branch_name="default"):
        """
        Track resource usage for a model
        
        Args:
            model: Model instance
            task_id: Task identifier
            branch_name (str): Branch name for tracking
            
        Returns:
            session_id: Resource tracking session ID
        """
        if not isinstance(model, OllamaModel):
            return None
        
        track_resources = self.config.get('resource_monitoring', True)
        if not track_resources:
            return None
            
        logger.info(f"Starting resource tracking for branch {branch_name}")
        session_id = await self.ollama_tracker.start_tracking(model.model_name)
        return session_id
    
    async def get_resource_stats(self, session_id, model, task_id, branch_name="default"):
        """
        Get resource usage statistics and save to file
        
        Args:
            session_id: Resource tracking session ID
            model: Model instance
            task_id: Task identifier
            branch_name (str): Branch name for tracking
            
        Returns:
            dict: Resource statistics
        """
        if session_id is None or session_id == -1:
            return self._get_default_resource_values()
        
        track_resources = self.config.get('resource_monitoring', True)
        if not track_resources:
            return self._get_default_resource_values()
        
        try:
            max_samples = 15  
            resource_tracking_task = asyncio.create_task(
                self.ollama_tracker.track_resources(
                    session_id, 
                    interval=1.0, 
                    max_samples=max_samples,
                    track_gpu=True
                )
            )
            
            resource_stats = await asyncio.wait_for(resource_tracking_task, timeout=30.0)
            
            if isinstance(resource_stats, dict) and not resource_stats.get('error'):
                result = {
                    "avg_cpu_percent": resource_stats.get("avg_cpu_percent", 0),
                    "avg_memory_mb": resource_stats.get("avg_memory_mb", 0),
                    "max_memory_mb": resource_stats.get("max_memory_mb", 0),
                    "duration_seconds": resource_stats.get("duration_seconds", 0)
                }
                
                if isinstance(model, OllamaModel):
                    formatted_branch_name = f"{task_id}_{branch_name}" if branch_name else task_id
                    stats_file = self.ollama_tracker.save_stats_to_file(
                        resource_stats,
                        model.model_name,
                        formatted_branch_name
                    )
                    result["stats_file"] = stats_file
                
                return result
        except asyncio.TimeoutError:
            logger.warning("Resource tracking task timed out, setting default resource values")
            return self._get_default_resource_values()
        except Exception as e:
            logger.error(f"Error getting resource tracking results: {str(e)}")
            return self._get_default_resource_values()
    
    def _get_default_resource_values(self):
        """
        Get default resource values when tracking fails
        
        Returns:
            dict: Default resource values
        """
        return {
            "avg_cpu_percent": 10.0,  
            "avg_memory_mb": 500.0,   
            "max_memory_mb": 1000.0,  
            "duration_seconds": 30.0  
        }
    
    def calculate_average_resources(self, resource_stats_list):
        """
        Calculate average resource usage across multiple runs
        
        Args:
            resource_stats_list (list): List of resource statistics dictionaries
            
        Returns:
            dict: Average resource statistics
        """
        import statistics
        
        if not resource_stats_list:
            return self._get_default_resource_values()
        
        cpu_values = [stats.get("avg_cpu_percent", 0) for stats in resource_stats_list]
        memory_values = [stats.get("avg_memory_mb", 0) for stats in resource_stats_list]
        max_memory_values = [stats.get("max_memory_mb", 0) for stats in resource_stats_list]
        duration_values = [stats.get("duration_seconds", 0) for stats in resource_stats_list]
        
        avg_cpu = statistics.mean(cpu_values) if cpu_values else 0
        avg_memory = statistics.mean(memory_values) if memory_values else 0
        avg_max_memory = statistics.mean(max_memory_values) if max_memory_values else 0
        avg_duration = statistics.mean(duration_values) if duration_values else 0
        
        return {
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory,
            "max_memory_mb": avg_max_memory,
            "duration_seconds": avg_duration,
            "total_models_used": len(resource_stats_list)
        }