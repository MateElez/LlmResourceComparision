import json
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

class MongoDBClient:
    """
    Client for MongoDB operations related to MBPP tasks and model evaluations
    """
    
    def __init__(self, uri, db_name):
        """
        Initialize MongoDB client
        
        Args:
            uri (str): MongoDB connection URI
            db_name (str): Name of the database to use
        """
        self.client = None
        self.db = None
        
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.server_info() 
            self.db = self.client["ResourceComparison"]  
            logger.info(f"Connected to MongoDB database: ResourceComparison")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def import_mbpp_data(self, file_path):
        """
        Import MBPP dataset from a jsonl file to MongoDB
        
        Args:
            file_path (str): Path to MBPP jsonl file
            
        Returns:
            int: Number of imported tasks
        """
        tasks_collection = self.db.tasks
        
        existing_count = tasks_collection.count_documents({})
        if existing_count > 0:
            logger.info(f"Found {existing_count} existing tasks in the database")
            return existing_count
        
        try:
            with open(file_path, 'r') as f:
                tasks = []
                for line in f:
                    task = json.loads(line)
                    if 'task_id' not in task and 'id' in task:
                        task['task_id'] = task['id']
                    tasks.append(task)
                
                if tasks:
                    result = tasks_collection.insert_many(tasks)
                    inserted_count = len(result.inserted_ids)
                    logger.info(f"Imported {inserted_count} tasks to MongoDB")
                    return inserted_count
                else:
                    logger.warning("No tasks found in the file")
                    return 0
        except Exception as e:
            logger.error(f"Error importing MBPP data: {str(e)}")
            raise
    
    def get_all_tasks(self):
        """
        Retrieve all tasks from the database
        
        Returns:
            list: List of task documents
        """
        return list(self.db.tasks.find({}))
    
    def get_task_by_id(self, task_id):
        """
        Retrieve a specific task by its ID
        
        Args:
            task_id: Task identifier
            
        Returns:
            dict: Task document or None if not found
        """
        task = self.db.tasks.find_one({"task_id": task_id})
        if not task:
            task = self.db.tasks.find_one({"_id": task_id})
        return task
    
    def save_evaluation_result(self, result):
        """
        Save model evaluation result to the database
        
        Args:
            result (dict): Evaluation result document
            
        Returns:
            str: ID of the inserted document
        """
        return str(self.db.results.insert_one(result).inserted_id)
    
    def save_resource_usage(self, usage_data):
        """
        Save resource usage data for a model run
        
        Args:
            usage_data (dict): Resource usage data
            
        Returns:
            str: ID of the inserted document
        """
        return str(self.db.resource_usage.insert_one(usage_data).inserted_id)
    
    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")