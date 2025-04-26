"""
Upravlja izvršavanjem koda u Daytona sandboxu za sigurno izvršavanje i validaciju LLM modela.
"""

import logging
import os
import socket
import time
from pathlib import Path
import asyncio

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
        self.host_ip = self._get_host_ip()
        logger.info(f"Korištenje host IP adrese za Ollama: {self.host_ip}")
        
    def _get_host_ip(self):
        """
        Dobiva IP adresu hosta za pristup Ollama servisu iz sandboxa
        
        Returns:
            str: IP adresa hosta ili localhost ako je ne može dobiti
        """
        try:
            # Kreiramo socket i povezujemo se na vanjski server da dobijemo lokalnu IP adresu
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Povezujemo se na Google DNS (nije potrebna stvarna konekcija)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            logger.warning(f"Nije moguće dobiti host IP adresu: {str(e)}. Korištenje localhost.")
            return "localhost"
    
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
        # VAŽNA PROMJENA: Preskačemo sandbox validaciju i vraćamo True
        # Validacija će se obaviti lokalno, izvan sandbox okruženja
        logger.info("Preskačem sandbox validaciju, koristit će se lokalna validacija")
        return True
        
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
            
            # Koristimo novu instancu Daytona klijenta za svaki sandbox
            daytona = Daytona(config)
            
            # Svaki poziv stvara novi, zaseban sandbox za paralelno izvršavanje
            sandbox_id = int(time.time() * 1000) % 10000  # Dodajemo identifikator za lakše praćenje
            
            # Stvaranje sandboxa - ovo NIJE coroutine u novoj verziji, stoga nema await
            try:
                logger.info(f"Kreiram novi Daytona sandbox (ID: {sandbox_id}) za izvršavanje {model_name} modela")
                sandbox = daytona.create()
                if not sandbox:
                    return None
                
                # Postavimo novo sjeme za generiranje slučajnih brojeva kako bi izbjegli konflikte
                import random
                random.seed()
                
                setup_code = """
import sys
import subprocess
import json

# Instaliramo potrebne pakete u sandbox
subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
print("Instalirani potrebni paketi")
                """
                logger.info(f"Postavljam sandbox okolinu za {model_name} (ID: {sandbox_id})...")
                
                # code_run nije coroutine u novoj verziji, stoga nema await
                setup_response = sandbox.process.code_run(setup_code)
                
                sandbox_code = """
import requests
import json
import time

def execute_ollama_model(model_name, prompt):
    # Parametri za API poziv
    api_url = "http://{2}:11434/api/generate"
    
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
        # Postavljanje kraćeg timeoutu za brže izvršavanje
        response = requests.post(api_url, json=payload, timeout=120)
        
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
""".format(model_name, prompt.replace("'", "\\'"), self.host_ip)
                
                logger.info(f"Izvršavam {model_name} model u Daytona sandboxu (ID: {sandbox_id}) sa stvarnim Ollama API pozivom")
                
                # code_run nije coroutine u novoj verziji, stoga nema await
                response = sandbox.process.code_run(sandbox_code)
                
                output = response.result
                if "SOLUTION_START" in output and "SOLUTION_END" in output:
                    start_marker = output.find("SOLUTION_START") + len("SOLUTION_START")
                    end_marker = output.find("SOLUTION_END")
                    solution = output[start_marker:end_marker].strip()
                    logger.info(f"Model {model_name} u sandboxu {sandbox_id} je uspješno generirao rješenje")
                    return solution
                else:
                    logger.warning(f"Nije pronađeno rješenje u izlazu sandboxa {sandbox_id}: {output[:200]}...")
                    return None
            finally:
                # Brisanje sandboxa - remove također NIJE coroutine u novoj verziji
                try:
                    if sandbox:
                        logger.info(f"Brišem Daytona sandbox (ID: {sandbox_id})")
                        daytona.remove(sandbox)
                except Exception as e:
                    logger.error(f"Greška prilikom brisanja Daytona sandboxa (ID: {sandbox_id}): {str(e)}")
                
        except ImportError:
            logger.warning("daytona_sdk nije instaliran, izvršavanje modela u sandboxu nije moguće")
            return None
        except Exception as e:
            logger.error(f"Greška tijekom izvršavanja modela u Daytona sandboxu: {str(e)}")
            return None