import json
import logging
from datetime import datetime
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class Neo4jClient:
    """
    Client for Neo4j operations for LLM resource comparison analysis
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j client
        
        Args:
            uri (str): Neo4j connection URI (e.g., 'bolt://localhost:7687')
            user (str): Username for authentication
            password (str): Password for authentication
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j database at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def create_constraints_and_indexes(self):
        """Create necessary constraints and indexes for the graph"""
        with self.driver.session() as session:
            constraints_and_indexes = [
                "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT model_name IF NOT EXISTS FOR (m:Model) REQUIRE m.name IS UNIQUE",
                "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
                "CREATE INDEX execution_timestamp IF NOT EXISTS FOR (e:Execution) ON (e.timestamp)",
                "CREATE INDEX task_difficulty IF NOT EXISTS FOR (t:Task) ON (t.difficulty)",
                "CREATE INDEX execution_success IF NOT EXISTS FOR (e:Execution) ON (e.success)",
                "CREATE INDEX resource_memory IF NOT EXISTS FOR (r:ResourceUsage) ON (r.peak_memory_mb)"
            ]
            
            for query in constraints_and_indexes:
                try:
                    session.run(query)
                    logger.info(f"Created constraint/index: {query.split()[1]}")
                except Exception as e:
                    logger.warning(f"Constraint/index may already exist: {e}")
    
    def create_tasks_from_json(self, tasks_file_path: str):
        """
        Create task nodes from tasks.json file
        
        Args:
            tasks_file_path (str): Path to tasks.json file
        """
        with open(tasks_file_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        
        with self.driver.session() as session:
            # First create categories
            categories = set(task['category'] for task in tasks)
            for category in categories:
                session.run("""
                    MERGE (c:Category {name: $category})
                    SET c.domain = $domain
                """, {
                    'category': category,
                    'domain': self._categorize_domain(category)
                })
            
            # Then create tasks and link to categories
            for task in tasks:
                session.run("""
                    MERGE (t:Task {id: $id})
                    SET t.description = $description,
                        t.difficulty = $difficulty,
                        t.input = $input,
                        t.expected_output = $expected_output,
                        t.complexity_score = $complexity_score
                    WITH t
                    MATCH (c:Category {name: $category})
                    MERGE (t)-[:BELONGS_TO]->(c)
                """, {
                    'id': task['task_id'],
                    'description': task['description'],
                    'category': task['category'],
                    'difficulty': task['difficulty'],
                    'input': str(task['input']),
                    'expected_output': str(task['expected_output']),
                    'complexity_score': self._calculate_complexity_score(task)
                })
            
            logger.info(f"Created {len(tasks)} task nodes and {len(categories)} category nodes")
    
    def create_models(self):
        """Create model nodes"""
        models = [
            {
                "name": "CodeLlama",
                "type": "large",
                "size": "7B",
                "parameters": 7000000000,
                "architecture": "transformer",
                "specialization": "code_generation"
            },
            {
                "name": "TinyLlama", 
                "type": "small",
                "size": "1.1B", 
                "parameters": 1100000000,
                "architecture": "transformer",
                "specialization": "general_purpose"
            }
        ]
        
        with self.driver.session() as session:
            for model in models:
                session.run("""
                    MERGE (m:Model {name: $name})
                    SET m.type = $type,
                        m.size = $size,
                        m.parameters = $parameters,
                        m.architecture = $architecture,
                        m.specialization = $specialization,
                        m.efficiency_class = $efficiency_class
                """, {
                    **model,
                    'efficiency_class': 'high_resource' if model['type'] == 'large' else 'low_resource'
                })
            
            # Create competition relationship
            session.run("""
                MATCH (large:Model {type: 'large'})
                MATCH (small:Model {type: 'small'})
                MERGE (large)-[:COMPETES_WITH {comparison_type: 'resource_efficiency'}]->(small)
                MERGE (small)-[:COMPETES_WITH {comparison_type: 'performance_accuracy'}]->(large)
            """)
            
            logger.info(f"Created {len(models)} model nodes with competition relationships")
    
    def migrate_execution_results(self, mongo_client):
        """
        Migrate execution results from MongoDB to Neo4j
        
        Args:
            mongo_client: MongoDB client instance
        """
        # Get all results from MongoDB
        results = list(mongo_client.db.results.find({}))
        
        with self.driver.session() as session:
            for result in results:
                task_id = result.get('task_id')
                timestamp = result.get('timestamp', datetime.now().isoformat())
                
                # Process large model result
                large_result = result.get('large_model_result', {})
                self._create_execution_with_resources(
                    session, task_id, "CodeLlama", large_result, timestamp
                )
                
                # Process small model result
                small_result = result.get('small_model_result', {})
                self._create_execution_with_resources(
                    session, task_id, "TinyLlama", small_result, timestamp
                )
                
                # Create comparison relationship
                self._create_comparison_relationship(
                    session, task_id, large_result, small_result, 
                    result.get('comparison_metrics', {})
                )
        
        logger.info(f"Migrated {len(results)} execution results")
    
    def _create_execution_with_resources(self, session, task_id: int, model_name: str, 
                                       model_result: Dict, timestamp: str):
        """Create execution and resource usage nodes with relationships"""
        if not model_result:
            return
            
        execution_id = f"{model_name}_{task_id}_{timestamp}"
        
        # Create execution node
        session.run("""
            MATCH (t:Task {id: $task_id})
            MATCH (m:Model {name: $model_name})
            CREATE (e:Execution {
                id: $execution_id,
                timestamp: datetime($timestamp),
                success: $success,
                execution_time: $execution_time,
                attempts: $attempts,
                solution: $solution,
                error_message: $error_message,
                strategy_used: $strategy_used
            })
            CREATE (m)-[:EXECUTED]->(e)
            CREATE (e)-[:ON_TASK]->(t)
        """, {
            'task_id': task_id,
            'model_name': model_name,
            'execution_id': execution_id,
            'timestamp': timestamp,
            'success': model_result.get('success', False),
            'execution_time': model_result.get('execution_time', 0.0),
            'attempts': model_result.get('attempts', 1),
            'solution': model_result.get('solution', ''),
            'error_message': model_result.get('error_message', ''),
            'strategy_used': 'branching' if model_result.get('attempts', 1) > 1 else 'single_attempt'
        })
        
        # Create resource usage node
        resource_usage = model_result.get('resource_usage', {})
        if resource_usage:
            session.run("""
                MATCH (e:Execution {id: $execution_id})
                CREATE (r:ResourceUsage {
                    peak_cpu_percent: $peak_cpu,
                    peak_memory_mb: $peak_memory,
                    avg_cpu_percent: $avg_cpu,
                    avg_memory_mb: $avg_memory,
                    resource_efficiency_score: $efficiency_score,
                    memory_category: $memory_category
                })
                CREATE (e)-[:USED_RESOURCES]->(r)
            """, {
                'execution_id': execution_id,
                'peak_cpu': resource_usage.get('peak_cpu_percent', 0.0),
                'peak_memory': resource_usage.get('peak_memory_mb', 0.0),
                'avg_cpu': resource_usage.get('avg_cpu_percent', 0.0),
                'avg_memory': resource_usage.get('avg_memory_mb', 0.0),
                'efficiency_score': self._calculate_efficiency_score(resource_usage),
                'memory_category': self._categorize_memory_usage(resource_usage.get('peak_memory_mb', 0))
            })
        
        # Create branching strategy details for small model
        if model_name == "TinyLlama" and model_result.get('branching_details'):
            branching_details = model_result['branching_details']
            session.run("""
                MATCH (e:Execution {id: $execution_id})
                CREATE (s:BranchingStrategy {
                    level_1_attempts: $level_1,
                    level_2_attempts: $level_2,
                    level_3_attempts: $level_3,
                    total_attempts: $total_attempts,
                    strategy_effectiveness: $effectiveness
                })
                CREATE (e)-[:APPLIED_STRATEGY]->(s)
            """, {
                'execution_id': execution_id,
                'level_1': branching_details.get('level_1_attempts', 0),
                'level_2': branching_details.get('level_2_attempts', 0),
                'level_3': branching_details.get('level_3_attempts', 0),
                'total_attempts': branching_details.get('total_attempts', 0),
                'effectiveness': 'high' if model_result.get('success') else 'low'
            })
    
    def _create_comparison_relationship(self, session, task_id: int, large_result: Dict, 
                                      small_result: Dict, comparison_metrics: Dict):
        """Create comparison relationships between executions"""
        if not large_result or not small_result:
            return
            
        session.run("""
            MATCH (large_exec:Execution)-[:ON_TASK]->(t:Task {id: $task_id})
            MATCH (large_exec)<-[:EXECUTED]-(large_model:Model {name: 'CodeLlama'})
            MATCH (small_exec:Execution)-[:ON_TASK]->(t)
            MATCH (small_exec)<-[:EXECUTED]-(small_model:Model {name: 'TinyLlama'})
            CREATE (large_exec)-[:COMPARED_WITH {
                resource_efficiency_ratio: $efficiency_ratio,
                time_ratio: $time_ratio,
                success_comparison: $success_comparison,
                winner: $winner,
                comparison_type: 'head_to_head'
            }]->(small_exec)
        """, {
            'task_id': task_id,
            'efficiency_ratio': comparison_metrics.get('resource_efficiency_ratio', 0.0),
            'time_ratio': comparison_metrics.get('time_ratio', 0.0),
            'success_comparison': comparison_metrics.get('success_comparison', 'unknown'),
            'winner': self._determine_winner(large_result, small_result)
        })
    
    def create_task_similarity_relationships(self):
        """Create similarity relationships between tasks based on various criteria"""
        with self.driver.session() as session:
            # Category-based similarity
            session.run("""
                MATCH (t1:Task)-[:BELONGS_TO]->(c:Category)<-[:BELONGS_TO]-(t2:Task)
                WHERE t1.id < t2.id
                CREATE (t1)-[:SIMILAR_CATEGORY {
                    similarity_type: 'category',
                    strength: 0.8
                }]->(t2)
            """)
            
            # Difficulty-based similarity
            session.run("""
                MATCH (t1:Task), (t2:Task)
                WHERE t1.id < t2.id AND t1.difficulty = t2.difficulty
                CREATE (t1)-[:SIMILAR_DIFFICULTY {
                    similarity_type: 'difficulty',
                    strength: 0.7
                }]->(t2)
            """)
            
            # Resource pattern similarity
            session.run("""
                MATCH (t1:Task)<-[:ON_TASK]-(e1:Execution)-[:USED_RESOURCES]->(r1:ResourceUsage)
                MATCH (t2:Task)<-[:ON_TASK]-(e2:Execution)-[:USED_RESOURCES]->(r2:ResourceUsage)
                MATCH (e1)<-[:EXECUTED]-(m1:Model)-[:EXECUTED]->(e2)
                WHERE t1.id < t2.id 
                  AND abs(r1.peak_memory_mb - r2.peak_memory_mb) < 100
                  AND abs(e1.execution_time - e2.execution_time) < 2.0
                CREATE (t1)-[:SIMILAR_RESOURCE_PATTERN {
                    similarity_type: 'resource_usage',
                    memory_diff: abs(r1.peak_memory_mb - r2.peak_memory_mb),
                    time_diff: abs(e1.execution_time - e2.execution_time),
                    model: m1.name,
                    strength: 0.9
                }]->(t2)
            """)
            
            logger.info("Created task similarity relationships")
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        with self.driver.session() as session:
            analytics = {}
            
            # Model comparison by category
            result = session.run("""
                MATCH (m:Model)-[:EXECUTED]->(e:Execution)-[:ON_TASK]->(t:Task)-[:BELONGS_TO]->(c:Category)
                MATCH (e)-[:USED_RESOURCES]->(r:ResourceUsage)
                WITH m.name as model, c.name as category, 
                     avg(r.peak_memory_mb) as avg_memory,
                     avg(e.execution_time) as avg_time,
                     sum(case when e.success then 1 else 0 end) * 1.0 / count(e) as success_rate,
                     count(e) as total_executions
                RETURN model, category, avg_memory, avg_time, success_rate, total_executions
                ORDER BY category, model
            """)
            analytics['model_performance_by_category'] = [record.data() for record in result]
            
            # Resource efficiency analysis
            result = session.run("""
                MATCH (m:Model)-[:EXECUTED]->(e:Execution)-[:USED_RESOURCES]->(r:ResourceUsage)
                WITH m.name as model,
                     sum(case when e.success then 1 else 0 end) as total_successes,
                     sum(r.peak_memory_mb * e.execution_time) as total_resource_cost,
                     count(e) as total_attempts,
                     avg(r.resource_efficiency_score) as avg_efficiency
                RETURN model, total_successes, total_resource_cost, 
                       total_successes * 1.0 / total_resource_cost as roi,
                       total_successes * 1.0 / total_attempts as success_rate,
                       avg_efficiency
                ORDER BY roi DESC
            """)
            analytics['resource_efficiency'] = [record.data() for record in result]
            
            # Branching strategy effectiveness
            result = session.run("""
                MATCH (e:Execution)-[:APPLIED_STRATEGY]->(s:BranchingStrategy)
                MATCH (e)-[:ON_TASK]->(t:Task)-[:BELONGS_TO]->(c:Category)
                WITH c.name as category,
                     avg(s.total_attempts) as avg_attempts,
                     sum(case when e.success then 1 else 0 end) * 1.0 / count(e) as success_rate,
                     count(e) as sample_size
                WHERE sample_size >= 3
                RETURN category, avg_attempts, success_rate, sample_size
                ORDER BY success_rate DESC
            """)
            analytics['branching_effectiveness'] = [record.data() for record in result]
            
            # Task difficulty vs resource scaling
            result = session.run("""
                MATCH (t:Task)<-[:ON_TASK]-(e:Execution)-[:USED_RESOURCES]->(r:ResourceUsage)
                MATCH (e)<-[:EXECUTED]-(m:Model)
                WITH t.difficulty as difficulty, m.name as model,
                     percentileCont(r.peak_memory_mb, 0.95) as p95_memory,
                     percentileCont(e.execution_time, 0.95) as p95_time,
                     avg(r.peak_memory_mb) as avg_memory
                RETURN difficulty, model, p95_memory, p95_time, avg_memory
                ORDER BY difficulty, model
            """)
            analytics['difficulty_scaling'] = [record.data() for record in result]
            
            return analytics
    
    def get_failure_patterns(self) -> List[Dict[str, Any]]:
        """Analyze failure patterns across models and categories"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Task)<-[:ON_TASK]-(e:Execution {success: false})
                MATCH (e)<-[:EXECUTED]-(m:Model)
                MATCH (t)-[:BELONGS_TO]->(c:Category)
                WITH c.name as category, m.name as model, count(e) as failures,
                     collect(distinct t.description)[0..5] as sample_failed_tasks,
                     collect(distinct e.error_message)[0..3] as common_errors
                WHERE failures > 2
                RETURN category, model, failures, sample_failed_tasks, common_errors
                ORDER BY failures DESC
            """)
            return [record.data() for record in result]
    
    def _categorize_domain(self, category: str) -> str:
        """Categorize domain based on category name"""
        if "string" in category.lower() or "stringovi" in category.lower():
            return "text_processing"
        elif "list" in category.lower() or "nizovi" in category.lower():
            return "data_structures"
        elif "math" in category.lower() or "matematika" in category.lower():
            return "mathematics"
        elif "algoritm" in category.lower():
            return "algorithms"
        elif "riječnic" in category.lower() or "skupov" in category.lower():
            return "collections"
        else:
            return "general"
    
    def _calculate_complexity_score(self, task: Dict) -> float:
        """Calculate complexity score based on task characteristics"""
        score = 1.0
        
        # Difficulty weight
        if task['difficulty'] == 'easy':
            score *= 1.0
        elif task['difficulty'] == 'medium':
            score *= 1.5
        elif task['difficulty'] == 'hard':
            score *= 2.0
        
        # Description length (proxy for complexity)
        desc_length = len(task['description'])
        if desc_length > 100:
            score *= 1.2
        elif desc_length > 150:
            score *= 1.4
        
        return round(score, 2)
    
    def _calculate_efficiency_score(self, resource_usage: Dict) -> float:
        """Calculate efficiency score based on resource usage"""
        if not resource_usage:
            return 0.0
        
        memory = resource_usage.get('peak_memory_mb', 0)
        cpu = resource_usage.get('peak_cpu_percent', 0)
        
        # Lower resource usage = higher efficiency
        # Normalize to 0-1 scale (inverse relationship)
        memory_score = max(0, 1 - (memory / 1000))  # Assume 1GB as high usage
        cpu_score = max(0, 1 - (cpu / 100))
        
        return round((memory_score + cpu_score) / 2, 3)
    
    def _categorize_memory_usage(self, memory_mb: float) -> str:
        """Categorize memory usage into tiers"""
        if memory_mb < 100:
            return "low"
        elif memory_mb < 500:
            return "medium"
        elif memory_mb < 1000:
            return "high"
        else:
            return "very_high"
    
    def _determine_winner(self, large_result: Dict, small_result: Dict) -> str:
        """Determine winner based on success and resource efficiency"""
        large_success = large_result.get('success', False)
        small_success = small_result.get('success', False)
        
        if large_success and small_success:
            # Both succeeded, check efficiency
            large_resources = large_result.get('resource_usage', {})
            small_resources = small_result.get('resource_usage', {})
            
            large_cost = (large_resources.get('peak_memory_mb', 0) * 
                         large_result.get('execution_time', 1))
            small_cost = (small_resources.get('peak_memory_mb', 0) * 
                         small_result.get('execution_time', 1))
            
            return "small_model" if small_cost < large_cost else "large_model"
        elif large_success:
            return "large_model"
        elif small_success:
            return "small_model"
        else:
            return "tie_both_failed"
