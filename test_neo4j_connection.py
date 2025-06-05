"""
Test Neo4j connection and setup for LLM Resource Comparison project
"""
import sys
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neo4j_connection(uri="bolt://localhost:7687", username="neo4j", password="password"):
    """
    Test connection to Neo4j database
    
    Args:
        uri: Neo4j connection URI
        username: Database username
        password: Database password
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        logger.info(f"Testing connection to {uri}...")
        
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection with a simple query
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                logger.info("✅ Neo4j connection successful!")
                
                # Get database info
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                for record in result:
                    logger.info(f"   Database: {record['name']} {record['versions'][0]} ({record['edition']})")
                
                driver.close()
                return True
            else:
                logger.error("❌ Unexpected response from Neo4j")
                driver.close()
                return False
                
    except ServiceUnavailable as e:
        logger.error(f"❌ Neo4j service unavailable at {uri}")
        logger.error(f"   Make sure Neo4j is running on {uri}")
        return False
        
    except AuthError as e:
        logger.error(f"❌ Authentication failed for user '{username}'")
        logger.error(f"   Check your username and password")
        return False
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        return False

def test_multiple_configurations():
    """
    Test multiple common Neo4j configurations
    """
    configurations = [
        {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "password"},
        {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "neo4j"},
        {"uri": "bolt://localhost:7687", "username": "neo4j", "password": ""},
        {"uri": "neo4j://localhost:7687", "username": "neo4j", "password": "password"},
    ]
    
    logger.info("Testing common Neo4j configurations...")
    
    for i, config in enumerate(configurations, 1):
        logger.info(f"\n--- Configuration {i} ---")
        if test_neo4j_connection(**config):
            logger.info(f"✅ Working configuration found!")
            logger.info(f"   URI: {config['uri']}")
            logger.info(f"   Username: {config['username']}")
            logger.info(f"   Password: {'*' * len(config['password']) if config['password'] else '(empty)'}")
            return config
    
    return None

def main():
    """
    Main function to test Neo4j setup
    """
    print("="*60)
    print("NEO4J CONNECTION TEST")
    print("="*60)
    
    # Test if any configuration works
    working_config = test_multiple_configurations()
    
    if working_config:
        print("\n" + "="*60)
        print("✅ NEO4J IS READY!")
        print("="*60)
        print("You can now run the migration script:")
        print("python migrate_to_neo4j.py")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("❌ NEO4J NOT ACCESSIBLE")
        print("="*60)
        print("Please set up Neo4j using one of these methods:")
        print("1. Neo4j Desktop: https://neo4j.com/download/")
        print("2. Neo4j Aura: https://neo4j.com/cloud/aura/")
        print("3. Local installation: https://neo4j.com/download-center/")
        print("\nSee NEO4J_SETUP.md for detailed instructions.")
        print("="*60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
