import logging
from models.large_model import download_code_llama
from models.small_model import download_tinyllama
from models.ollama_model import OllamaModel, CodeLlamaModel, TinyLlamaModel

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Responsible for model initialization, download, and availability checking
    """
    
    def __init__(self, config):
        """
        Initialize model manager
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
    
    async def download_models(self):
        """
        Download all required models at startup
        """
        logger.info("Downloading required models")
        
        try:
            download_code_llama("./models_data")
            logger.info("CodeLlama model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download CodeLlama model: {str(e)}")
        
        try:
            download_tinyllama("./models_data")
            logger.info("TinyLlama model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download TinyLlama model: {str(e)}")
    
    def get_large_model(self):
        """
        Get configured large model instance
        
        Returns:
            OllamaModel: Initialized large model
        """
        large_model_config = self.config["models"]["large_model"]
        model_name = large_model_config.get("name", "codellama")
        
        if model_name == "codellama":
            return CodeLlamaModel()
        else:
            return OllamaModel(model_name)
    
    def get_small_model(self):
        """
        Get configured small model instance
        
        Returns:
            OllamaModel: Initialized small model
        """
        small_model_config = self.config["models"]["small_model"]
        model_name = small_model_config.get("name", "tinyllama")
        
        if model_name == "tinyllama":
            return TinyLlamaModel()
        else:
            return OllamaModel(model_name)
    
    def check_model_availability(self, model):
        """
        Check if a model is available
        
        Args:
            model (OllamaModel): Model to check
            
        Returns:
            bool: True if model is available, False otherwise
        """
        return model.check_status()