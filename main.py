#!/usr/bin/env python
import logging
import argparse
import os
import yaml
import json
import asyncio
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
    parser.add_argument('--multi-task', action='store_true',
                        help='Run multiple tasks with limited branching')
    parser.add_argument('--all-tasks', action='store_true',
                        help='Run all 100 tasks with branching depth 1')
    parser.add_argument('--task-count', type=int, default=5,
                        help='Number of tasks to run with --multi-task (default: 5)')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Maximum branching depth for small models (default: 2)')
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
        }
    }

def generate_tasks():
    categories = [
        "Stringovi",
        "Liste i nizovi",
        "Matematika",
        "Rad s riječnicima i skupovima",
        "Algoritmi i logika (sortiranje, pretraga itd.)"
    ]

    tasks = []
    task_id = 1

    for category in categories:
        for i in range(20):
            task = {
                "task_id": task_id,
                "category": category,
                "description": f"Task {i+1} in category {category}",
                "difficulty": "medium",  # Example difficulty level
                "input": "Example input for task",
                "expected_output": "Expected output for task"
            }
            tasks.append(task)
            task_id += 1

    return tasks

def store_tasks_in_mongo(tasks):
    # Initialize MongoDB client
    config = load_config("config.yaml")
    mongo_client = MongoDBClient(
        config['mongodb']['uri'],
        config['mongodb']['db_name']
    )

    # Store tasks in the 'tasks' collection
    mongo_client.db.tasks.insert_many(tasks)
    print(f"Stored {len(tasks)} tasks in MongoDB 'tasks' collection.")

def insert_tasks_from_json():
    # Load tasks from tasks.json
    with open("tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Initialize MongoDB client
    config = load_config("config.yaml")
    mongo_client = MongoDBClient(
        config['mongodb']['uri'],
        config['mongodb']['db_name']
    )

    # Insert tasks into the 'tasks' collection
    mongo_client.db.tasks.insert_many(tasks)
    print(f"Inserted {len(tasks)} tasks into MongoDB 'tasks' collection.")

def execute_all_tasks_with_branching():
    """
    Execute all 100 tasks with maximum branching depth of 1.
    For small model: if first attempt fails, branch to 2 parallel attempts with different prompts.
    """
    # Load configuration
    config = load_config("config.yaml")
    
    # Set branching depth to 1 for this execution
    config["max_branching_depth"] = 1
    
    # Initialize MongoDB client
    mongo_client = MongoDBClient(
        config['mongodb']['uri'],
        config['mongodb']['db_name']
    )

    # Initialize workflow manager
    workflow_manager = WorkflowManager(mongo_client, config)
    
    # Get all tasks from database
    tasks = list(mongo_client.db.tasks.find().sort("task_id", 1))
    total_tasks = len(tasks)
    
    logger.info(f"Starting execution of {total_tasks} tasks with max branching depth 1")
    
    # Track statistics
    processed_count = 0
    successful_count = 0
    failed_count = 0
    
    # Process each task
    for i, task in enumerate(tasks):
        try:
            logger.info(f"Processing task {i+1}/{total_tasks}: Task ID {task['task_id']} - {task.get('description', 'No description')[:100]}...")
            
            # Process the task using the WorkflowManager
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(workflow_manager.process_task(task))
                
                if result:
                    # Ensure result is a proper dictionary before saving to MongoDB
                    if isinstance(result, dict):
                        # Add execution metadata
                        result['execution_batch'] = 'all_tasks_branching_depth_1'
                        result['execution_timestamp'] = result.get('execution_timestamp')
                        result['max_branching_depth'] = 1
                        
                        mongo_client.db.results.insert_one(result)
                        successful_count += 1
                        logger.info(f"✓ Task {task['task_id']} completed successfully")
                    else:
                        logger.error(f"✗ Task {task['task_id']} - Invalid result format: {type(result)}")
                        failed_count += 1
                else:
                    logger.error(f"✗ Task {task['task_id']} - No result returned")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"✗ Task {task['task_id']} - Error during processing: {str(e)}")
                failed_count += 1
                import traceback
                logger.error(traceback.format_exc())
                
            finally:
                loop.close()
                processed_count += 1
                
            # Progress update every 10 tasks
            if processed_count % 10 == 0:
                logger.info(f"Progress: {processed_count}/{total_tasks} tasks processed "
                           f"({successful_count} successful, {failed_count} failed)")
                
        except Exception as e:
            logger.error(f"✗ Critical error processing task {task.get('task_id', 'unknown')}: {str(e)}")
            failed_count += 1
            processed_count += 1
    
    # Final statistics
    logger.info("="*80)
    logger.info("EXECUTION COMPLETE")
    logger.info(f"Total tasks processed: {processed_count}")
    logger.info(f"Successful executions: {successful_count}")
    logger.info(f"Failed executions: {failed_count}")
    logger.info(f"Success rate: {(successful_count/processed_count)*100:.2f}%")
    logger.info("="*80)
    
    # Save execution summary
    summary = {
        'execution_type': 'all_tasks_branching_depth_1',
        'total_tasks': total_tasks,
        'processed_count': processed_count,
        'successful_count': successful_count,
        'failed_count': failed_count,
        'success_rate': (successful_count/processed_count)*100 if processed_count > 0 else 0,
        'max_branching_depth': 1,
        'timestamp': workflow_manager.get_timestamp() if hasattr(workflow_manager, 'get_timestamp') else None
    }
    
    mongo_client.db.execution_summaries.insert_one(summary)
    logger.info("Execution summary saved to database")
    
    return summary

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

    # Set max branching depth in the configuration
    config["max_branching_depth"] = args.max_depth

    # Check if we should run all tasks
    if args.all_tasks:
        logger.info("Starting execution of all 100 tasks with branching depth 1")
        summary = execute_all_tasks_with_branching()
        logger.info(f"All tasks execution completed. Success rate: {summary['success_rate']:.2f}%")
        return

    # Run the workflow pipeline for only 1 task (original behavior)
    task = mongo_client.db.tasks.find_one()
    if task:
        logger.info(f"Processing task {task['task_id']}: {task['description']}")

        # Process the task using the WorkflowManager
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(workflow_manager.process_task(task))
            if result:
                # Ensure result is a proper dictionary before saving to MongoDB
                if isinstance(result, dict):
                    mongo_client.db.results.insert_one(result)
                    logger.info(f"Task {task['task_id']} processed and result saved.")
                else:
                    logger.error(f"Invalid result format: {type(result)}. Expected dictionary.")
        except Exception as e:
            logger.error(f"Error processing task {task['task_id']}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())  # Add detailed traceback for debugging
        finally:
            loop.close()

    logger.info("Processed 1 task and saved result.")

    # Execute all 100 tasks with maximum branching depth of 1
    execute_all_tasks_with_branching()

if __name__ == "__main__":
    # Insert tasks from tasks.json into MongoDB
    insert_tasks_from_json()

    # Run the main workflow
    main()