# complete_migration.py
from neo4j import GraphDatabase
from pymongo import MongoClient
from datetime import datetime
import logging

def complete_migration():
    # MongoDB connection
    mongo_client = MongoClient('mongodb://localhost:27017')
    db = mongo_client['ResourceComparison']
    
    # Neo4j connection
    driver = GraphDatabase.driver("bolt://localhost:7687", 
                                 auth=("neo4j", "password"))
    
    with driver.session() as session:
        print("🗑️ Clearing existing Neo4j data...")
        session.run("MATCH (n) DETACH DELETE n")
        
        print("📊 Creating base structure...")
        
        # Create Models
        session.run("""
            CREATE (m1:Model {
                name: 'CodeLlama', 
                type: 'large', 
                size: '7B',
                ollama_name: 'codellama'
            })
            CREATE (m2:Model {
                name: 'TinyLlama', 
                type: 'small', 
                size: '1.1B',
                ollama_name: 'small_model'
            })
        """)
        
        # Get all execution results
        results = list(db.results.find())
        print(f"Found {len(results)} execution results")
        
        # Track created tasks and categories
        created_tasks = set()
        categories = set()
        
        for i, result in enumerate(results, 1):
            print(f"Processing result {i}/{len(results)}: Task {result.get('task_id', 'unknown')}")
            
            task_id = result.get('task_id')
            task_text = result.get('task_text', '')
            
            # Extract category from task_text or use default
            if task_id:
                if task_id <= 20:
                    category = "Stringovi"
                elif task_id <= 40:
                    category = "Matematika"
                elif task_id <= 60:
                    category = "Liste i nizovi"
                elif task_id <= 80:
                    category = "Rad s riječnicima i skupovima"
                else:
                    category = "Algoritmi i logika"
            else:
                category = "Unknown"
            
            categories.add(category)
            
            # Create Task if not exists
            if task_id not in created_tasks:
                session.run("""
                    CREATE (t:Task {
                        task_id: $task_id,
                        description: $description,
                        category: $category,
                        timestamp: datetime($timestamp)
                    })
                """, 
                task_id=task_id,
                description=task_text,
                category=category,
                timestamp=result.get('timestamp', datetime.now().isoformat())
                )
                created_tasks.add(task_id)
            
            # Process Large Model Result - POPRAVKA!
            if 'models' in result and 'large_model' in result['models']:
                large_model = result['models']['large_model']
                print(f"  📊 Processing Large Model for Task {task_id}")
                
                session.run("""
                    MATCH (m:Model {name: 'CodeLlama'}), (t:Task {task_id: $task_id})
                    CREATE (e:Execution {
                        model_name: $model_name,
                        success: $success,
                        execution_time: $execution_time,
                        start_time: datetime($start_time),
                        end_time: datetime($end_time),
                        solution: $solution,
                        error_type: $error_type,
                        error_explanation: $error_explanation,
                        attempts: 1,
                        branching_level: 0
                    })
                    CREATE (m)-[:EXECUTED]->(e)
                    CREATE (e)-[:ON_TASK]->(t)
                """,
                task_id=task_id,
                model_name=large_model.get('name', 'codellama'),
                success=large_model.get('evaluation', {}).get('success', False),
                execution_time=large_model.get('resources', {}).get('duration_seconds', 0),
                start_time=large_model.get('start_time', datetime.now().isoformat()),
                end_time=large_model.get('end_time', datetime.now().isoformat()),
                solution=large_model.get('solution', ''),
                error_type=large_model.get('evaluation', {}).get('error_type', ''),
                error_explanation=large_model.get('evaluation', {}).get('explanation', '')
                )
                
                # Create Resource Usage for Large Model
                if 'resources' in large_model:
                    resources = large_model['resources']
                    print(f"  💾 Creating ResourceUsage for Large Model Task {task_id}")
                    session.run("""
                        MATCH (e:Execution)
                        WHERE e.model_name = $model_name 
                          AND e.execution_time = $execution_time
                          AND exists((e)-[:ON_TASK]->(:Task {task_id: $task_id}))
                        CREATE (r:ResourceUsage {
                            peak_memory_mb: $peak_memory_mb,
                            avg_memory_mb: $avg_memory_mb,
                            avg_cpu_percent: $avg_cpu_percent,
                            duration_seconds: $duration_seconds,
                            samples: $samples,
                            stats_file: $stats_file
                        })
                        CREATE (e)-[:USED_RESOURCES]->(r)
                    """,
                    model_name=large_model.get('name', 'codellama'),
                    execution_time=resources.get('duration_seconds', 0),
                    task_id=task_id,
                    peak_memory_mb=resources.get('max_memory_mb', 0),
                    avg_memory_mb=resources.get('avg_memory_mb', 0),
                    avg_cpu_percent=resources.get('avg_cpu_percent', 0),
                    duration_seconds=resources.get('duration_seconds', 0),
                    samples=resources.get('samples', 0),
                    stats_file=resources.get('stats_file', '')
                    )
            
            # Process Small Model Result - POPRAVKA!
            if 'models' in result and 'small_model' in result['models']:
                small_model = result['models']['small_model']
                print(f"  📱 Processing Small Model for Task {task_id}")
                
                # Determine attempts and branching
                attempts = 1
                branching_level = 0
                
                # Check for branching results
                branch_data = {}
                if 'branch_1_1' in result and result['branch_1_1']:
                    attempts += 1
                    branching_level = 1
                    branch_data['branch_1_1'] = result['branch_1_1']
                
                if 'branch_1_2' in result and result['branch_1_2']:
                    attempts += 1
                    branch_data['branch_1_2'] = result['branch_1_2']
                
                session.run("""
                    MATCH (m:Model {name: 'TinyLlama'}), (t:Task {task_id: $task_id})
                    CREATE (e:Execution {
                        model_name: $model_name,
                        success: $success,
                        execution_time: $execution_time,
                        start_time: datetime($start_time),
                        end_time: datetime($end_time),
                        solution: $solution,
                        error_type: $error_type,
                        error_explanation: $error_explanation,
                        attempts: $attempts,
                        branching_level: $branching_level,
                        branch_count: $branch_count
                    })
                    CREATE (m)-[:EXECUTED]->(e)
                    CREATE (e)-[:ON_TASK]->(t)
                """,
                task_id=task_id,
                model_name=small_model.get('name', 'small_model'),
                success=small_model.get('evaluation', {}).get('success', False),
                execution_time=small_model.get('resources', {}).get('duration_seconds', 0),
                start_time=small_model.get('start_time', datetime.now().isoformat()),
                end_time=small_model.get('end_time', datetime.now().isoformat()),
                solution=small_model.get('solution', ''),
                error_type=small_model.get('evaluation', {}).get('error_type', ''),
                error_explanation=small_model.get('evaluation', {}).get('explanation', ''),
                attempts=attempts,
                branching_level=branching_level,
                branch_count=len(branch_data)
                )
                
                # Create Resource Usage for Small Model
                if 'resources' in small_model:
                    resources = small_model['resources']
                    print(f"  💾 Creating ResourceUsage for Small Model Task {task_id}")
                    session.run("""
                        MATCH (e:Execution)
                        WHERE e.model_name = $model_name 
                          AND e.execution_time = $execution_time
                          AND exists((e)-[:ON_TASK]->(:Task {task_id: $task_id}))
                        CREATE (r:ResourceUsage {
                            peak_memory_mb: $peak_memory_mb,
                            avg_memory_mb: $avg_memory_mb,
                            avg_cpu_percent: $avg_cpu_percent,
                            duration_seconds: $duration_seconds,
                            samples: $samples,
                            stats_file: $stats_file
                        })
                        CREATE (e)-[:USED_RESOURCES]->(r)
                    """,
                    model_name=small_model.get('name', 'small_model'),
                    execution_time=resources.get('duration_seconds', 0),
                    task_id=task_id,
                    peak_memory_mb=resources.get('max_memory_mb', 0),
                    avg_memory_mb=resources.get('avg_memory_mb', 0),
                    avg_cpu_percent=resources.get('avg_cpu_percent', 0),
                    duration_seconds=resources.get('duration_seconds', 0),
                    samples=resources.get('samples', 0),
                    stats_file=resources.get('stats_file', '')
                    )
        
        print("🏷️ Creating Categories...")
        # Create Categories
        for category in categories:
            task_count = len([r for r in results if get_category_for_task(r.get('task_id', 0)) == category])
            session.run("""
                CREATE (c:Category {
                    name: $name,
                    task_count: $task_count
                })
            """, name=category, task_count=task_count)
        
        # Create Task-Category relationships
        session.run("""
            MATCH (t:Task), (c:Category)
            WHERE t.category = c.name
            CREATE (t)-[:BELONGS_TO]->(c)
        """)
        
        print("✅ Migration completed!")
        
        # Show final statistics
        stats = session.run("""
            MATCH (n) 
            RETURN labels(n)[0] as node_type, count(n) as count
            ORDER BY count DESC
        """)
        
        print("\n📊 Final Neo4j Statistics:")
        for record in stats:
            print(f"  {record['node_type']}: {record['count']}")
            
        # Show relationship statistics
        rel_stats = session.run("""
            MATCH ()-[r]->() 
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY count DESC
        """)
        
        print("\n🔗 Relationship Statistics:")
        for record in rel_stats:
            print(f"  {record['relationship_type']}: {record['count']}")
    
    driver.close()
    mongo_client.close()

def get_category_for_task(task_id):
    """Map task_id to category"""
    if task_id <= 20:
        return "Stringovi"
    elif task_id <= 40:
        return "Matematika"
    elif task_id <= 60:
        return "Liste i nizovi"
    elif task_id <= 80:
        return "Rad s riječnicima i skupovima"
    else:
        return "Algoritmi i logika"

if __name__ == "__main__":
    complete_migration()