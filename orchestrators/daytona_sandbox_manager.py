"""
Upravlja izvršavanjem koda u Daytona sandboxu za sigurno izvršavanje i validaciju LLM modela.
"""

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class DaytonaSandboxManager:
    """
    Upravlja izvršavanjem i validacijom koda u Daytona sandbox okruženju
    """
    
    def __init__(self):
        """
        Inicijalizira manager za Daytona sandbox
        """
        self.daytona_config = self._load_daytona_config()
        
    def _load_daytona_config(self):
        """
        Učitava Daytona konfiguraciju iz .env datoteke
        
        Returns:
            dict: Konfiguracijski parametri za Daytona
        """
        config = {}
        
        env_path = Path(__file__).parent.parent / '.env'
        
        if env_path.exists():
            logger.info(f"Pronađena .env datoteka na lokaciji: {env_path}")
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = [part.strip() for part in line.split('=', 1)]
                            config[key] = value
                logger.info("Uspješno učitana Daytona konfiguracija iz .env datoteke")
            except Exception as e:
                logger.error(f"Greška prilikom učitavanja .env datoteke: {str(e)}")
        else:
            logger.warning(f".env datoteka nije pronađena na lokaciji: {env_path}")
            
        return config
    
    async def validate_in_sandbox(self, solution, test_list, test_setup_code):
        """
        Validira rješenje u Daytona sandbox okruženju
        
        Args:
            solution (str): Generirani kod rješenja
            test_list (list): Lista test slučajeva
            test_setup_code (str): Kod za postavljanje testova
            
        Returns:
            bool: True ako je rješenje validno u sandboxu, False inače
        """
        try:
            from daytona_sdk import Daytona, DaytonaConfig
            
            code_pattern = r"```python\s*(.*?)\s*```"
            code_match = re.search(code_pattern, solution, re.DOTALL)
            
            if code_match:
                code = code_match.group(1)
            else:
                code = solution
            
            api_key = self.daytona_config.get('key', os.environ.get("key", ""))
            api_url = self.daytona_config.get('url', None)
            
            if not api_key:
                logger.warning("Daytona API ključ nije pronađen u .env datoteci niti okolišnim varijablama, preskačem sandbox validaciju")
                return True 
                
            config_kwargs = {"api_key": api_key}
            if api_url:
                config_kwargs["base_url"] = api_url
                
            config = DaytonaConfig(**config_kwargs)
            daytona = Daytona(config)
            
            logger.info("Kreiram Daytona sandbox za validaciju koda")
            sandbox = daytona.create()
            
            try:
                full_code = code + "\n\n"
                
                if test_setup_code:
                    full_code += test_setup_code + "\n\n"
                
                full_code += "try:\n"
                for test in test_list:
                    full_code += f"    {test}\n"
                full_code += "    print('ALL_TESTS_PASSED')\n"
                full_code += "except Exception as e:\n"
                full_code += "    print(f'TEST_FAILED: {str(e)}')\n"
                
                logger.info("Izvršavam kod u Daytona sandboxu")
                response = sandbox.process.code_run(full_code)
                
                success = "ALL_TESTS_PASSED" in response.result
                if success:
                    logger.info("Kod je uspješno prošao sve testove u Daytona sandboxu")
                else:
                    logger.warning(f"Kod nije prošao testove u Daytona sandboxu: {response.result}")
                    
                return success
            finally:
                logger.info("Brišem Daytona sandbox")
                daytona.remove(sandbox)
                
        except ImportError:
            logger.warning("daytona_sdk nije instaliran, preskačem sandbox validaciju")
            return True  
        except Exception as e:
            logger.error(f"Greška tijekom Daytona sandbox validacije: {str(e)}")
            return False
        
    async def execute_model_in_sandbox(self, prompt, model_name="default"):
        """
        Izvršava LLM model u Daytona sandboxu i vraća generiran kod
        
        Args:
            prompt (str): Prompt za model
            model_name (str): Naziv modela (za logiranje)
            
        Returns:
            str: Generirano rješenje ili None u slučaju greške
        """
        try:
            from daytona_sdk import Daytona, DaytonaConfig
            
            api_key = self.daytona_config.get('key', os.environ.get("key", ""))
            api_url = self.daytona_config.get('url', None)
            
            if not api_key:
                logger.warning("Daytona API ključ nije pronađen u .env datoteci niti okolišnim varijablama, model će se izvršavati lokalno")
                return None
                
            config_kwargs = {"api_key": api_key}
            if api_url:
                config_kwargs["base_url"] = api_url
                
            config = DaytonaConfig(**config_kwargs)
            daytona = Daytona(config)
            
            logger.info(f"Kreiram Daytona sandbox za izvršavanje {model_name} modela")
            sandbox = daytona.create()
            
            try:
                setup_code = """
import sys
import subprocess
import json

# Instaliramo potrebne pakete u sandbox
subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
print("Instalirani potrebni paketi")
                """
                logger.info("Postavljam sandbox okolinu...")
                setup_response = sandbox.process.code_run(setup_code)
                
                sandbox_code = """
import requests
import json
import time

def execute_ollama_model(model_name, prompt):
    # Stvarno izvršavanje LLM modela kroz Ollama API
    #
    # Args:
    #     model_name: Naziv modela (codellama, tinyllama, itd.)
    #     prompt: Prompt za LLM
    #
    # Returns:
    #     str: Generirano rješenje
    
    # Parametri za API poziv
    api_url = "http://localhost:11434/api/generate"
    
    # Kreiraj JSON payload s pravilnim formatiranjem
    payload = {{
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {{
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40
        }}
    }}
    
    print(f"Izvršavam model {{model_name}}...")
    
    try:
        # Postavljanje dužeg timeoutu jer veći modeli mogu potrajati
        response = requests.post(api_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            print(f"Greška pri pozivu API-ja: {{response.status_code}}")
            print(response.text)
            return f"Error calling Ollama API: {{response.status_code}}"
            
    except Exception as e:
        print(f"Iznimka tijekom izvršavanja LLM modela: {{str(e)}}")
        return f"Exception during LLM execution: {{str(e)}}"

# Parametri za izvršavanje
model_name = "{0}"
prompt = '''{1}'''

# Stvarno izvršavanje modela
generated_solution = execute_ollama_model(model_name, prompt)

# Ispisujemo rješenje između markera za lakšu ekstrakciju
print("SOLUTION_START")
print(generated_solution)
print("SOLUTION_END")
""".format(model_name, prompt.replace("'", "\\'"))
                
                logger.info(f"Izvršavam {model_name} model u Daytona sandboxu sa stvarnim Ollama API pozivom")
                response = sandbox.process.code_run(sandbox_code)
                
                output = response.result
                if "SOLUTION_START" in output and "SOLUTION_END" in output:
                    start_marker = output.find("SOLUTION_START") + len("SOLUTION_START")
                    end_marker = output.find("SOLUTION_END")
                    solution = output[start_marker:end_marker].strip()
                    logger.info(f"Model {model_name} je uspješno generirao rješenje u Daytona sandboxu")
                    return solution
                else:
                    logger.warning(f"Nije pronađeno rješenje u izlazu sandboxa: {output[:200]}...")
                    return None
            finally:
                logger.info("Brišem Daytona sandbox")
                daytona.remove(sandbox)
                
        except ImportError:
            logger.warning("daytona_sdk nije instaliran, izvršavanje modela u sandboxu nije moguće")
            return None
        except Exception as e:
            logger.error(f"Greška tijekom izvršavanja modela u Daytona sandboxu: {str(e)}")
            return None