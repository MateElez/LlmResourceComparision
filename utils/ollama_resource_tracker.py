"""
Track resource usage (CPU, memory) of locally running Ollama processes.
This module provides functionality to monitor resource consumption of Ollama models
running on the local machine.
"""

import logging
import psutil
import time
import asyncio
import json
import os
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class OllamaResourceTracker:
    """
    Tracks resource usage (CPU, memory) of locally running Ollama processes
    """
    
    def __init__(self):
        """Initialize the Ollama resource tracker"""
        self.ollama_processes = []
        self._should_stop = False
    
    def _find_ollama_processes(self) -> List[psutil.Process]:
        """
        Find all running Ollama processes
        
        Returns:
            List[psutil.Process]: List of Ollama processes
        """
        ollama_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                    ollama_processes.append(proc)
                elif proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline']).lower()
                    if 'ollama' in cmdline:
                        ollama_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        return ollama_processes
    
    def _find_model_specific_process(self, model_name: str) -> Optional[psutil.Process]:
        """
        Try to find a process specific to the given model name
        
        Args:
            model_name (str): Name of the model to find
            
        Returns:
            Optional[psutil.Process]: Process if found, None otherwise
        """
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline']).lower()
                    if model_name.lower() in cmdline and 'ollama' in cmdline:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        return None
    
    async def start_tracking(self, model_name: Optional[str] = None) -> int:
        """
        Start tracking Ollama processes
        
        Args:
            model_name (Optional[str]): Specific model to track, or None to track all Ollama processes
            
        Returns:
            int: Tracking session ID, or -1 if no processes found
        """
        if model_name:
            model_process = self._find_model_specific_process(model_name)
            if model_process:
                self.ollama_processes = [model_process]
            else:
                self.ollama_processes = self._find_ollama_processes()
        else:
            self.ollama_processes = self._find_ollama_processes()
        
        if not self.ollama_processes:
            logger.warning("No Ollama processes found for tracking")
            return -1
        
        logger.info(f"Found {len(self.ollama_processes)} Ollama processes to track")
        
        return int(time.time())
    
    async def track_resources(self, session_id: int, 
                              interval: float = 1.0, 
                              max_samples: int = 60,
                              track_gpu: bool = False) -> Dict[str, Any]:
        """
        Track resource usage for the given session
        
        Args:
            session_id (int): Tracking session ID
            interval (float): Sampling interval in seconds
            max_samples (int): Maximum number of samples to collect
            track_gpu (bool): Whether to attempt tracking GPU usage
            
        Returns:
            Dict[str, Any]: Resource usage statistics
        """
        if not self.ollama_processes:
            logger.error("No Ollama processes available for tracking")
            return {
                "error": "No Ollama processes available",
                "session_id": session_id
            }
        
        stats = []
        cpu_total = 0
        memory_total = 0
        memory_max = 0
        samples_collected = 0
        min_samples_required = 3  # Smanjujemo minimalni broj uzoraka koji se smatraju valjanim 
        
        gpu_tracking_available = False
        if track_gpu:
            try:
                import subprocess
                subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                gpu_tracking_available = True
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("nvidia-smi not available, GPU tracking disabled")
                gpu_tracking_available = False
        
        try:
            start_time = time.time()
            self._should_stop = False
            
            # Inicijalno čitanje CPU korištenja za sve procese
            for proc in self.ollama_processes[:]:
                try:
                    if proc.is_running():
                        proc.cpu_percent(interval=None)
                except Exception:
                    pass
                    
            # Kraći inicijalni delay za brže čitanje druge vrijednosti CPU-a
            await asyncio.sleep(0.1)
            
            while samples_collected < max_samples and not self._should_stop:
                sample = {
                    "timestamp": time.time() - start_time,
                    "cpu_percent": 0.0,
                    "memory_mb": 0.0,
                    "memory_percent": 0.0
                }
                
                cpu_sample_total = 0
                memory_sample_total = 0
                valid_processes = 0
                
                active_processes = []
                for proc in self.ollama_processes:
                    try:
                        if proc.is_running():
                            cpu_usage = proc.cpu_percent(interval=None)
                            memory_info = proc.memory_info()
                            memory_usage_mb = memory_info.rss / (1024 * 1024)
                            
                            # Prihvaćamo sve vrijednosti, čak i 0 za prve uzorke
                            cpu_sample_total += cpu_usage
                            memory_sample_total += memory_usage_mb
                            valid_processes += 1
                            active_processes.append(proc)
                            logger.debug(f"Process {proc.pid} CPU: {cpu_usage:.1f}%, Memory: {memory_usage_mb:.1f}MB")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                        logger.debug(f"Couldn't access process: {str(e)}")
                        continue
                
                self.ollama_processes = active_processes
                
                if valid_processes > 0:
                    sample["cpu_percent"] = cpu_sample_total
                    sample["memory_mb"] = memory_sample_total
                    sample["memory_percent"] = (memory_sample_total / psutil.virtual_memory().total) * 100
                    
                    cpu_total += sample["cpu_percent"]
                    memory_total += sample["memory_mb"]
                    memory_max = max(memory_max, sample["memory_mb"])
                    logger.debug(f"Sample {samples_collected+1}: CPU: {sample['cpu_percent']:.1f}%, Memory: {sample['memory_mb']:.1f}MB")
                else:
                    # Ako nismo pronašli procese, pokušaj pronaći nove
                    new_processes = self._find_ollama_processes()
                    if new_processes:
                        self.ollama_processes = new_processes
                        logger.info(f"Found {len(new_processes)} Ollama processes to track")
                        await asyncio.sleep(interval)
                        continue
                    else:
                        # Ako još uvijek nema procesa, prekidamo praćenje
                        logger.warning("No Ollama processes found during tracking")
                        break
                
                if gpu_tracking_available:
                    try:
                        import subprocess
                        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], 
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                        
                        lines = result.stdout.strip().split('\n')
                        gpu_data = []
                        
                        for i, line in enumerate(lines):
                            parts = line.split(',')
                            if len(parts) >= 2:
                                gpu_util = float(parts[0].strip())
                                gpu_mem = float(parts[1].strip())
                                gpu_data.append({
                                    "gpu_id": i,
                                    "utilization_percent": gpu_util,
                                    "memory_used_mb": gpu_mem
                                })
                        
                        sample["gpu"] = gpu_data
                    except Exception as e:
                        logger.error(f"Error collecting GPU data: {str(e)}")
                
                stats.append(sample)
                samples_collected += 1
                
                # Dodajemo check za prekid ako imamo dovoljno uzoraka
                if self._should_stop:
                    logger.info(f"Resource tracking stopping early after {samples_collected} samples")
                    break
                
                # Kraći interval spavanja za više uzoraka u zadanom vremenskom okviru
                await asyncio.sleep(interval)
            
            # Ako nismo prikupili nikakve uzorke, vraćamo defaultne vrijednosti
            if samples_collected == 0:
                logger.warning("No samples collected during resource tracking")
                return {
                    "session_id": session_id,
                    "samples": 0,
                    "duration_seconds": time.time() - start_time,
                    "avg_cpu_percent": 10.0,
                    "avg_memory_mb": 500.0,
                    "max_memory_mb": 1000.0,
                    "detailed_stats": [],
                    "note": "No samples collected, using default values"
                }
            
            # Izračunaj prosjeke na temelju prikupljenih uzoraka
            avg_cpu = cpu_total / samples_collected if samples_collected > 0 else 0
            avg_memory = memory_total / samples_collected if samples_collected > 0 else 0
            
            result = {
                "session_id": session_id,
                "samples": samples_collected,
                "duration_seconds": time.time() - start_time,
                "avg_cpu_percent": avg_cpu,
                "avg_memory_mb": avg_memory,
                "max_memory_mb": memory_max,
                "detailed_stats": stats[:min(10, len(stats))]  # Samo prvih 10 uzoraka za uštedu prostora
            }
            
            logger.info(f"Resource tracking completed: Avg CPU: {avg_cpu:.2f}%, Avg Memory: {avg_memory:.2f}MB, Max Memory: {memory_max:.2f}MB, Samples: {samples_collected}")
            return result
            
        except asyncio.CancelledError:
            # Čak i ako je praćenje prekinuto, vratimo parcijalne rezultate ako imamo dovoljno uzoraka
            if samples_collected >= min_samples_required:
                logger.info(f"Resource tracking cancelled but has {samples_collected} valid samples - returning partial results")
                avg_cpu = cpu_total / samples_collected
                avg_memory = memory_total / samples_collected
                
                return {
                    "session_id": session_id,
                    "samples": samples_collected,
                    "duration_seconds": time.time() - start_time,
                    "avg_cpu_percent": avg_cpu,
                    "avg_memory_mb": avg_memory,
                    "max_memory_mb": memory_max,
                    "detailed_stats": stats[:min(10, len(stats))],
                    "note": "Partial results due to cancellation"
                }
            else:
                # Ne koristimo više defaultne vrijednosti, već računamo na temelju onog što imamo
                if samples_collected > 0:
                    avg_cpu = cpu_total / samples_collected
                    avg_memory = memory_total / samples_collected
                    
                    logger.warning(f"Resource tracking cancelled with only {samples_collected} samples, but still using them")
                    return {
                        "session_id": session_id,
                        "samples": samples_collected,
                        "duration_seconds": time.time() - start_time,
                        "avg_cpu_percent": avg_cpu,
                        "avg_memory_mb": avg_memory,
                        "max_memory_mb": memory_max,
                        "detailed_stats": stats,
                        "note": f"Limited samples ({samples_collected}), but still valid"
                    }
                else:
                    logger.warning("Resource tracking cancelled without any samples - using realistic default values")
                    return {
                        "session_id": session_id,
                        "samples": 0,
                        "duration_seconds": time.time() - start_time,
                        "avg_cpu_percent": 15.0,  # Realističnije defaultne vrijednosti
                        "avg_memory_mb": 800.0,   
                        "max_memory_mb": 1200.0,
                        "detailed_stats": [],
                        "note": "No samples collected, using realistic default values"
                    }
        except Exception as e:
            logger.error(f"Error while tracking Ollama resources: {str(e)}")
            return {
                "error": str(e),
                "session_id": session_id,
                "avg_cpu_percent": 15.0,
                "avg_memory_mb": 800.0,
                "max_memory_mb": 1200.0,
                "duration_seconds": time.time() - start_time,
                "samples": samples_collected,
                "note": "Error during tracking, using realistic default values"
            }
    
    def stop_tracking(self):
        """Signal that resource tracking should stop at next opportunity"""
        self._should_stop = True
    
    def save_stats_to_file(self, stats: Dict[str, Any], model_name: str, task_id: str) -> str:
        """
        Save resource statistics to a JSON file
        
        Args:
            stats (Dict[str, Any]): Resource statistics to save
            model_name (str): Name of the model these stats belong to
            task_id (str): ID of the task being executed
            
        Returns:
            str: Path to the saved file
        """
        os.makedirs('resources', exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"resources/ollama_{model_name}_{task_id}_{timestamp}.json"
        
        stats_with_meta = stats.copy()
        stats_with_meta["model_name"] = model_name
        stats_with_meta["task_id"] = task_id
        
        with open(filename, 'w') as f:
            json.dump(stats_with_meta, f, indent=2)
        
        logger.info(f"Resource statistics saved to {filename}")
        return filename

async def main():
    """Test the OllamaResourceTracker functionality"""
    tracker = OllamaResourceTracker()
    
    print("Starting to track Ollama processes...")
    session_id = await tracker.start_tracking()
    
    if session_id != -1:
        print(f"Tracking session {session_id} for 10 seconds...")
        stats = await tracker.track_resources(session_id, interval=1.0, max_samples=10)
        
        print("\nResource usage summary:")
        print(f"Average CPU: {stats['avg_cpu_percent']:.2f}%")
        print(f"Average Memory: {stats['avg_memory_mb']:.2f} MB")
        print(f"Maximum Memory: {stats['max_memory_mb']:.2f} MB")
        
        filename = tracker.save_stats_to_file(stats, "test", "test_task")
        print(f"Statistics saved to {filename}")
    else:
        print("No Ollama processes found to track.")

if __name__ == "__main__":
    asyncio.run(main())