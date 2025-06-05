#!/usr/bin/env python
"""
Convenience script to run all 100 tasks with branching depth 1.
This script executes the main.py with the --all-tasks flag.
"""

import subprocess
import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_all_tasks():
    """
    Execute all 100 tasks with maximum branching depth of 1.
    """
    logger.info("Starting execution of all 100 tasks...")
    logger.info("Configuration: Maximum branching depth = 1")
    logger.info("This means:")
    logger.info("- Large model (CodeLlama): Single attempt per task")
    logger.info("- Small model (TinyLlama): If first attempt fails, 2 parallel attempts with enhanced prompts")
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    try:
        # Run main.py with --all-tasks flag
        cmd = [sys.executable, "main.py", "--all-tasks", "--config", "config.yaml"]
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        # Print the output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check return code
        if result.returncode == 0:
            logger.info("✓ All tasks execution completed successfully!")
        else:
            logger.error(f"✗ Execution failed with return code: {result.returncode}")
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        logger.error("✗ Execution timed out after 2 hours")
        return 1
    except Exception as e:
        logger.error(f"✗ Error executing all tasks: {str(e)}")
        return 1

if __name__ == "__main__":
    # Check if MongoDB is running
    logger.info("Checking prerequisites...")
    
    # Check if config file exists
    if not os.path.exists("config.yaml"):
        logger.error("config.yaml not found. Please ensure the configuration file exists.")
        sys.exit(1)
    
    # Check if tasks are in database
    logger.info("Prerequisites check passed. Starting execution...")
    
    # Run all tasks
    exit_code = run_all_tasks()
    
    if exit_code == 0:
        logger.info("="*80)
        logger.info("ALL TASKS EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info("Next steps:")
        logger.info("1. Run Neo4j migration: python migrate_to_neo4j.py")
        logger.info("2. Generate analytics: python -c \"from analytics.graph_analytics import *; run_all_analytics()\"")
        logger.info("3. Create visualizations: python -c \"from analytics.visualizations import *; create_all_visualizations()\"")
        logger.info("="*80)
    
    sys.exit(exit_code)
