import logging
import aiohttp
import requests

logger = logging.getLogger(__name__)

class OllamaModel:
    """Base class for Ollama models"""
    
    def __init__(self, model_name):
        """
        Initialize the Ollama model
        
        Args:
            model_name (str): Name of the model in Ollama
        """
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        logger.info(f"Initialized Ollama model: {model_name}")
    
    def check_status(self):
        """
        Check if the model is available in Ollama
        
        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                logger.error(f"Ollama server not available. Status code: {response.status_code}")
                return False
            
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            if self.model_name in model_names:
                logger.info(f"Model '{self.model_name}' found in Ollama")
                return True
            
            if f"{self.model_name}:latest" in model_names:
                logger.info(f"Model '{self.model_name}:latest' found in Ollama")
                self.model_name = f"{self.model_name}:latest"
                return True
                
            logger.warning(f"Model '{self.model_name}' not found in Ollama")
            return False
        except Exception as e:
            logger.error(f"Error checking model status: {str(e)}")
            return False
    
    async def generate(self, prompt, system_prompt=None, max_tokens=2000):
        """
        Generate text using the Ollama model
        
        Args:
            prompt (str): Prompt to generate from
            system_prompt (str, optional): System prompt to use
            max_tokens (int, optional): Maximum number of tokens to generate
            
        Returns:
            str: Generated text
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "")
                    else:
                        error_text = await response.text()
                        logger.error(f"Error generating from model. Status code: {response.status}. Error: {error_text}")
                        return f"Error: {error_text}"
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return f"Error: {str(e)}"


class CodeLlamaModel(OllamaModel):
    """CodeLlama model implementation for Ollama"""
    
    def __init__(self):
        """Initialize CodeLlama model"""
        super().__init__("codellama")
        
        self.system_prompt = "You are a helpful coding assistant. Provide Python code to solve programming problems."
    
    async def generate(self, prompt, max_tokens=2000):
        """Generate code using CodeLlama model"""
        return await super().generate(prompt, self.system_prompt, max_tokens)


class TinyLlamaModel(OllamaModel):
    """TinyLlama model implementation for Ollama"""
    
    def __init__(self):
        """Initialize TinyLlama model"""
        super().__init__("tinyllama")
        
        self.system_prompt = "You are a helpful coding assistant. Provide Python code to solve programming problems."
    
    async def generate(self, prompt, max_tokens=2000):
        """Generate code using TinyLlama model"""
        return await super().generate(prompt, self.system_prompt, max_tokens)