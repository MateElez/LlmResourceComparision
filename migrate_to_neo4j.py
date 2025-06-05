"""
Migration script to transfer data from MongoDB to Neo4j for graph analytics
"""
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.mongo_client import MongoDBClient
from database.neo4j_client import Neo4jClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main migration function"""
    
    # Configuration
    MONGO_URI = "mongodb://localhost:27017/"
    MONGO_DB = "ResourceComparison"
    
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this to your Neo4j password
    
    TASKS_JSON_PATH = "tasks.json"
    
    try:
        logger.info("Starting migration from MongoDB to Neo4j...")
        
        # Initialize clients
        logger.info("Connecting to MongoDB...")
        mongo_client = MongoDBClient(MONGO_URI, MONGO_DB)
        
        logger.info("Connecting to Neo4j...")
        neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Step 1: Create constraints and indexes
        logger.info("Creating Neo4j constraints and indexes...")
        neo4j_client.create_constraints_and_indexes()
        
        # Step 2: Create tasks and categories from JSON
        logger.info("Creating task and category nodes...")
        neo4j_client.create_tasks_from_json(TASKS_JSON_PATH)
        
        # Step 3: Create model nodes
        logger.info("Creating model nodes...")
        neo4j_client.create_models()
        
        # Step 4: Migrate execution results
        logger.info("Migrating execution results...")
        neo4j_client.migrate_execution_results(mongo_client)
        
        # Step 5: Create task similarity relationships
        logger.info("Creating task similarity relationships...")
        neo4j_client.create_task_similarity_relationships()
        
        # Step 6: Generate sample analytics
        logger.info("Generating sample analytics...")
        analytics = neo4j_client.get_performance_analytics()
        
        # Display some results
        print("\n" + "="*60)
        print("MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nModel Performance by Category:")
        for record in analytics['model_performance_by_category'][:5]:
            print(f"  {record['model']} on {record['category']}: "
                  f"{record['success_rate']:.2%} success, "
                  f"{record['avg_memory']:.1f}MB avg memory")
        
        print(f"\nResource Efficiency Analysis:")
        for record in analytics['resource_efficiency']:
            print(f"  {record['model']}: ROI={record['roi']:.4f}, "
                  f"Success Rate={record['success_rate']:.2%}")
        
        print(f"\nFailure Patterns:")
        failure_patterns = neo4j_client.get_failure_patterns()
        for pattern in failure_patterns[:3]:
            print(f"  {pattern['model']} in {pattern['category']}: "
                  f"{pattern['failures']} failures")
        
        print(f"\nNeo4j database is ready for graph analytics!")
        print(f"Access Neo4j Browser at: http://localhost:7474")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise
    finally:
        # Clean up connections
        if 'mongo_client' in locals():
            mongo_client.close()
        if 'neo4j_client' in locals():
            neo4j_client.close()

if __name__ == "__main__":
    main()
