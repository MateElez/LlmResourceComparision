#!/usr/bin/env python
import logging
import argparse
import os
import yaml
from database.mongo_client import MongoDBClient
from orchestrators.workflow_manager import WorkflowManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='MBPP Task Evaluation Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--import-data', action='store_true',
                        help='Import MBPP data to MongoDB')
    parser.add_argument('--single-task', action='store_true',
                        help='Run only a single task for testing')
    parser.add_argument('--task-id', type=str, 
                        help='Specific task ID to run (if not specified, first task is used)')
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from a YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}, using default configuration")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return get_default_config()

def get_default_config():
    """
    Return default configuration when config file is not found
    
    Returns:
        dict: Default configuration dictionary
    """
    return {
        "mongodb": {
            "uri": "mongodb://localhost:27017",
            "db_name": "ResourceComparison"
        },
        "data_path": {
            "mbpp": "./data/mbpp.jsonl"
        },
        "resource_monitoring": True,
        "resources_output_dir": "./resources",
        "models": {
            "large_model": {
                "name": "codellama",
                "type": "ollama",
                "api_endpoint": "http://localhost:11434/api/generate",
                "use_fallback": True
            },
            "small_model": {
                "name": "tinyllama",
                "type": "ollama",
                "api_endpoint": "http://localhost:11434/api/generate",
                "use_fallback": True
            },
            "fallback_model_1": {
                "name": "fallback-model-1",
                "api_endpoint": "http://localhost:8002/generate"
            },
            "fallback_model_2": {
                "name": "fallback-model-2",
                "api_endpoint": "http://localhost:8003/generate"
            }
        },
        "docker": {
            "use_containers": False,
            "resource_monitoring": False,
            "timeout": 600,
            "memory_limit": "4g",
            "cpu_limit": 1
        }
    }

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize MongoDB client
    mongo_client = MongoDBClient(
        config['mongodb']['uri'],
        config['mongodb']['db_name']
    )
    
    # Initialize workflow manager
    workflow_manager = WorkflowManager(mongo_client, config)
    
    # Import MBPP data if requested
    if args.import_data:
        logger.info("Importing MBPP dataset to MongoDB")
        workflow_manager.import_mbpp_data(config['data_path']['mbpp'])
    
    # Run the workflow pipeline
    if args.single_task:
        # Run with a single task for testing
        logger.info("Running with a single task for testing")
        
        if args.task_id:
            # If task ID is provided, get that specific task
            task = mongo_client.get_task_by_id(args.task_id)
            if not task:
                logger.error(f"Task with ID {args.task_id} not found")
                return
            logger.info(f"Running specific task: {args.task_id}")
        else:
            # Get the first task from MongoDB instead of reading from file
            tasks = mongo_client.get_all_tasks()
            if not tasks or len(tasks) == 0:
                logger.error("No tasks found in MongoDB. Try importing data first with --import-data flag")
                return
            task = tasks[0]  # Just take the first task
            logger.info(f"Running first task from MongoDB: {task.get('task_id')}")
        
        # Run the single task
        workflow_manager.run_single_task(task)
    else:
        # Run the full pipeline
        workflow_manager.run_pipeline()
    
    # Clean up resources
    mongo_client.close()

if __name__ == "__main__":
    main()