"""
Automated Neo4j setup and migration for LLM Resource Comparison project
"""
import sys
import webbrowser
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_neo4j_availability():
    """Check if Neo4j is already running"""
    try:
        from test_neo4j_connection import test_neo4j_connection
        return test_neo4j_connection()
    except ImportError:
        # If test module not available, try direct connection
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            return True
        except:
            return False

def open_neo4j_download():
    """Open Neo4j Desktop download page"""
    url = "https://neo4j.com/download/"
    logger.info(f"Opening Neo4j Desktop download page: {url}")
    webbrowser.open(url)

def wait_for_neo4j_setup():
    """Wait for user to set up Neo4j and test connection"""
    print("\n" + "="*60)
    print("NEO4J SETUP REQUIRED")
    print("="*60)
    print("1. Download and install Neo4j Desktop from the opened webpage")
    print("2. Create a new database with these settings:")
    print("   - Database Name: llm-resource-comparison")
    print("   - Password: password")
    print("   - Version: Latest stable")
    print("3. Start the database")
    print("4. Come back here and press Enter to continue")
    print("="*60)
    
    # Open download page
    open_neo4j_download()
    
    # Wait for user confirmation
    input("\nPress Enter when Neo4j is running...")
    
    # Test connection
    logger.info("Testing Neo4j connection...")
    max_attempts = 3
    for attempt in range(max_attempts):
        if check_neo4j_availability():
            logger.info("✅ Neo4j connection successful!")
            return True
        
        if attempt < max_attempts - 1:
            logger.warning(f"❌ Connection failed (attempt {attempt + 1}/{max_attempts})")
            retry = input("Neo4j not detected. Press Enter to retry, or 'q' to quit: ")
            if retry.lower() == 'q':
                return False
        else:
            logger.error("❌ Could not connect to Neo4j after all attempts")
            return False
    
    return False

def run_migration():
    """Run the data migration from MongoDB to Neo4j"""
    try:
        logger.info("Starting migration from MongoDB to Neo4j...")
        
        # Import and run migration
        import subprocess
        result = subprocess.run([sys.executable, "migrate_to_neo4j.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Migration completed successfully!")
            print(result.stdout)
            return True
        else:
            logger.error("❌ Migration failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Migration error: {str(e)}")
        return False

def run_analytics():
    """Run graph analytics and generate visualizations"""
    try:
        logger.info("Running graph analytics...")
        
        # Create and run analytics script
        analytics_script = '''
import sys
import os
sys.path.append(os.getcwd())

from analytics import GraphAnalytics, GraphVisualizer
from database.neo4j_client import Neo4jClient

def main():
    # Connect to Neo4j
    neo4j_client = Neo4jClient()
    
    # Initialize analytics
    analytics = GraphAnalytics(neo4j_client)
    visualizer = GraphVisualizer()
    
    print("Running graph analytics...")
    
    # Run all analyses
    results = {}
    
    try:
        print("1. Analyzing model efficiency clusters...")
        results["efficiency_data"] = analytics.analyze_model_efficiency_clusters()
        print("   ✅ Efficiency analysis complete")
    except Exception as e:
        print(f"   ❌ Efficiency analysis failed: {e}")
    
    try:
        print("2. Analyzing branching strategy effectiveness...")
        results["branching_data"] = analytics.analyze_branching_strategy_effectiveness()
        print("   ✅ Branching analysis complete")
    except Exception as e:
        print(f"   ❌ Branching analysis failed: {e}")
    
    try:
        print("3. Analyzing task similarity networks...")
        results["network_data"] = analytics.analyze_task_similarity_networks()
        print("   ✅ Network analysis complete")
    except Exception as e:
        print(f"   ❌ Network analysis failed: {e}")
    
    try:
        print("4. Comparing model performance paths...")
        results["comparison_data"] = analytics.compare_model_performance_paths()
        print("   ✅ Performance comparison complete")
    except Exception as e:
        print(f"   ❌ Performance comparison failed: {e}")
    
    # Generate visualizations
    if results:
        print("\\nGenerating visualizations...")
        try:
            visualizer.save_all_visualizations(results, "visualizations")
            print("✅ All visualizations saved to visualizations/ directory")
            
            # Generate summary report
            summary = visualizer.generate_summary_report(results)
            print("\\n" + summary)
            
            # Save summary to file
            with open("analytics_summary.txt", "w") as f:
                f.write(summary)
            print("\\n📊 Summary report saved to analytics_summary.txt")
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
    else:
        print("❌ No analytics results to visualize")
    
    neo4j_client.close()

if __name__ == "__main__":
    main()
'''
        
        # Write and execute analytics script
        with open("run_analytics.py", "w") as f:
            f.write(analytics_script)
        
        import subprocess
        result = subprocess.run([sys.executable, "run_analytics.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Analytics completed successfully!")
            print(result.stdout)
            return True
        else:
            logger.error("❌ Analytics failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Analytics error: {str(e)}")
        return False

def main():
    """Main setup and migration workflow"""
    print("="*60)
    print("LLM RESOURCE COMPARISON - NEO4J SETUP & MIGRATION")
    print("="*60)
    
    # Step 1: Check if Neo4j is already running
    logger.info("Step 1: Checking Neo4j availability...")
    if check_neo4j_availability():
        logger.info("✅ Neo4j is already running!")
    else:
        logger.info("Neo4j not detected. Setting up...")
        if not wait_for_neo4j_setup():
            logger.error("❌ Neo4j setup failed. Exiting.")
            return False
    
    # Step 2: Run migration
    logger.info("Step 2: Running data migration...")
    if not run_migration():
        logger.error("❌ Migration failed. Exiting.")
        return False
    
    # Step 3: Run analytics
    logger.info("Step 3: Running graph analytics...")
    if not run_analytics():
        logger.error("❌ Analytics failed.")
        return False
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("Your LLM Resource Comparison project is now using Neo4j!")
    print("Check the following outputs:")
    print("- visualizations/ directory for graphs and charts")
    print("- analytics_summary.txt for key insights")
    print("- Neo4j Browser at http://localhost:7474 for interactive queries")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
