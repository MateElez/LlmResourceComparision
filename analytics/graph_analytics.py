"""
Graph analytics module for LLM resource comparison using Neo4j
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
from database.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

class GraphAnalytics:
    """
    Advanced graph analytics for LLM resource comparison
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize analytics with Neo4j client
        
        Args:
            neo4j_client: Connected Neo4j client instance
        """
        self.neo4j = neo4j_client
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_model_efficiency_clusters(self) -> Dict[str, Any]:
        """
        Analyze clusters of tasks with similar resource patterns
        """
        with self.neo4j.driver.session() as session:
            result = session.run("""
                MATCH (t:Task)<-[:ON_TASK]-(e:Execution)-[:USED_RESOURCES]->(r:ResourceUsage)
                MATCH (e)<-[:EXECUTED]-(m:Model)
                WITH t, m.name as model, 
                     avg(r.peak_memory_mb) as avg_memory, 
                     avg(r.peak_cpu_percent) as avg_cpu,
                     avg(e.execution_time) as avg_time,
                     collect(r.resource_efficiency_score) as efficiency_scores
                RETURN t.id as task_id, t.category as category, t.difficulty as difficulty,
                       model, avg_memory, avg_cpu, avg_time, efficiency_scores
                ORDER BY avg_memory DESC
            """)
            
            data = [record.data() for record in result]
            df = pd.DataFrame(data)
            
            # Perform clustering analysis
            clusters = self._cluster_resource_patterns(df)
            
            return {
                'data': data,
                'clusters': clusters,
                'analysis': self._analyze_clusters(df, clusters)
            }
    
    def analyze_branching_strategy_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of branching strategies
        """
        with self.neo4j.driver.session() as session:
            result = session.run("""
                MATCH (e:Execution)-[:APPLIED_STRATEGY]->(s:BranchingStrategy)
                MATCH (e)-[:ON_TASK]->(t:Task)-[:BELONGS_TO]->(c:Category)
                MATCH (e)-[:USED_RESOURCES]->(r:ResourceUsage)
                RETURN t.id as task_id, t.difficulty as difficulty, c.name as category,
                       s.total_attempts as total_attempts, e.success as success,
                       e.execution_time as execution_time, r.peak_memory_mb as peak_memory,
                       s.level_1_attempts, s.level_2_attempts, s.level_3_attempts
            """)
            
            data = [record.data() for record in result]
            df = pd.DataFrame(data)
            
            # Analyze branching patterns
            branching_analysis = {
                'success_by_attempts': df.groupby('total_attempts')['success'].agg(['count', 'sum', 'mean']),
                'category_effectiveness': df.groupby('category')['success'].agg(['count', 'sum', 'mean']),
                'difficulty_scaling': df.groupby('difficulty')['total_attempts'].agg(['mean', 'std']),
                'resource_vs_attempts': df[['total_attempts', 'peak_memory', 'execution_time']].corr()
            }
            
            return {
                'data': data,
                'analysis': branching_analysis,
                'insights': self._generate_branching_insights(df)
            }
    
    def find_optimal_task_routing(self) -> Dict[str, Any]:
        """
        Find optimal routing of tasks to models based on characteristics
        """
        with self.neo4j.driver.session() as session:
            result = session.run("""
                MATCH (t:Task)<-[:ON_TASK]-(e1:Execution)<-[:EXECUTED]-(m1:Model {name: 'CodeLlama'})
                MATCH (t)<-[:ON_TASK]-(e2:Execution)<-[:EXECUTED]-(m2:Model {name: 'TinyLlama'})
                MATCH (e1)-[:USED_RESOURCES]->(r1:ResourceUsage)
                MATCH (e2)-[:USED_RESOURCES]->(r2:ResourceUsage)
                RETURN t.id as task_id, t.category as category, t.difficulty as difficulty,
                       t.complexity_score as complexity,
                       e1.success as large_success, e2.success as small_success,
                       e1.execution_time as large_time, e2.execution_time as small_time,
                       r1.peak_memory_mb as large_memory, r2.peak_memory_mb as small_memory,
                       (r2.peak_memory_mb * e2.execution_time) / (r1.peak_memory_mb * e1.execution_time) as efficiency_ratio
            """)
            
            data = [record.data() for record in result]
            df = pd.DataFrame(data)
            
            # Determine optimal routing rules
            routing_rules = self._generate_routing_rules(df)
            
            return {
                'data': data,
                'routing_rules': routing_rules,
                'performance_matrix': self._create_performance_matrix(df)
            }
    
    def analyze_task_similarity_networks(self) -> Dict[str, Any]:
        """
        Analyze networks of similar tasks
        """
        with self.neo4j.driver.session() as session:
            # Get similarity relationships
            result = session.run("""
                MATCH (t1:Task)-[sim:SIMILAR_RESOURCE_PATTERN]->(t2:Task)
                RETURN t1.id as source, t2.id as target, sim.strength as weight,
                       sim.memory_diff as memory_diff, sim.time_diff as time_diff,
                       t1.category as source_category, t2.category as target_category
                UNION ALL
                MATCH (t1:Task)-[sim:SIMILAR_CATEGORY]->(t2:Task)
                RETURN t1.id as source, t2.id as target, sim.strength as weight,
                       null as memory_diff, null as time_diff,
                       t1.category as source_category, t2.category as target_category
            """)
            
            edges = [record.data() for record in result]
            
            # Get task attributes
            result = session.run("""
                MATCH (t:Task)-[:BELONGS_TO]->(c:Category)
                RETURN t.id as task_id, t.difficulty as difficulty, 
                       c.name as category, t.complexity_score as complexity
            """)
            
            nodes = [record.data() for record in result]
            
            # Build NetworkX graph
            G = self._build_similarity_graph(nodes, edges)
            
            # Analyze network properties
            network_analysis = {
                'centrality_measures': self._calculate_centrality_measures(G),
                'community_detection': self._detect_communities(G),
                'network_metrics': self._calculate_network_metrics(G)
            }
            
            return {
                'graph': G,
                'nodes': nodes,
                'edges': edges,
                'analysis': network_analysis
            }
    
    def compare_model_performance_paths(self) -> Dict[str, Any]:
        """
        Analyze performance paths and decision trees for model selection
        """
        with self.neo4j.driver.session() as session:
            result = session.run("""
                MATCH (t:Task)<-[:ON_TASK]-(e:Execution)-[:COMPARED_WITH]->(e2:Execution)
                MATCH (e)<-[:EXECUTED]-(m1:Model)
                MATCH (e2)<-[:EXECUTED]-(m2:Model)
                MATCH (e)-[:USED_RESOURCES]->(r1:ResourceUsage)
                MATCH (e2)-[:USED_RESOURCES]->(r2:ResourceUsage)
                RETURN t.id as task_id, t.category as category, t.difficulty as difficulty,
                       m1.name as model1, m2.name as model2,
                       e.success as success1, e2.success as success2,
                       r1.peak_memory_mb as memory1, r2.peak_memory_mb as memory2,
                       e.execution_time as time1, e2.execution_time as time2
            """)
            
            data = [record.data() for record in result]
            df = pd.DataFrame(data)
            
            # Create decision tree analysis
            decision_analysis = {
                'success_patterns': self._analyze_success_patterns(df),
                'resource_trade_offs': self._analyze_resource_trade_offs(df),
                'category_preferences': self._analyze_category_preferences(df)
            }
            
            return {
                'data': data,
                'analysis': decision_analysis,
                'recommendations': self._generate_model_recommendations(df)
            }
    
    def generate_insights_report(self) -> str:
        """
        Generate a comprehensive insights report
        """
        report = []
        report.append("="*80)
        report.append("LLM RESOURCE COMPARISON - GRAPH ANALYTICS REPORT")
        report.append("="*80)
        
        # Model efficiency analysis
        efficiency_data = self.analyze_model_efficiency_clusters()
        report.append("\n🔍 MODEL EFFICIENCY CLUSTERS:")
        report.extend(self._format_efficiency_insights(efficiency_data))
        
        # Branching strategy analysis
        branching_data = self.analyze_branching_strategy_effectiveness()
        report.append("\n🌳 BRANCHING STRATEGY EFFECTIVENESS:")
        report.extend(self._format_branching_insights(branching_data))
        
        # Task routing analysis
        routing_data = self.find_optimal_task_routing()
        report.append("\n🗺️ OPTIMAL TASK ROUTING:")
        report.extend(self._format_routing_insights(routing_data))
        
        # Network analysis
        network_data = self.analyze_task_similarity_networks()
        report.append("\n🕸️ TASK SIMILARITY NETWORKS:")
        report.extend(self._format_network_insights(network_data))
        
        # Performance comparison
        comparison_data = self.compare_model_performance_paths()
        report.append("\n⚖️ MODEL PERFORMANCE COMPARISON:")
        report.extend(self._format_comparison_insights(comparison_data))
        
        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\n".join(report)
    
    # Helper methods for analysis
    def _cluster_resource_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster tasks by resource usage patterns"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features for clustering
        features = ['avg_memory', 'avg_cpu', 'avg_time']
        X = df[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        return {
            'cluster_labels': clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_analysis': pd.DataFrame(X).assign(cluster=clusters).groupby('cluster').agg(['mean', 'std'])
        }
    
    def _analyze_clusters(self, df: pd.DataFrame, clusters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        df_with_clusters = df.assign(cluster=clusters['cluster_labels'])
        
        cluster_summary = df_with_clusters.groupby('cluster').agg({
            'avg_memory': ['mean', 'std'],
            'avg_cpu': ['mean', 'std'], 
            'avg_time': ['mean', 'std'],
            'category': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
            'difficulty': lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
        })
        
        return {
            'cluster_summary': cluster_summary,
            'cluster_sizes': df_with_clusters['cluster'].value_counts(),
            'dominant_patterns': self._identify_dominant_patterns(df_with_clusters)
        }
    
    def _generate_branching_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate insights about branching strategy effectiveness"""
        insights = []
        
        # Success rate by attempt level
        success_by_attempts = df.groupby('total_attempts')['success'].mean()
        best_attempt_count = success_by_attempts.idxmax()
        best_success_rate = success_by_attempts.max()
        
        insights.append(f"Optimal attempt count: {best_attempt_count} (success rate: {best_success_rate:.2%})")
        
        # Category analysis
        category_success = df.groupby('category')['success'].agg(['count', 'mean'])
        best_category = category_success['mean'].idxmax()
        worst_category = category_success['mean'].idxmin()
        
        insights.append(f"Best performing category: {best_category} ({category_success.loc[best_category, 'mean']:.2%})")
        insights.append(f"Most challenging category: {worst_category} ({category_success.loc[worst_category, 'mean']:.2%})")
        
        return insights
    
    def _generate_routing_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate optimal routing rules for task assignment"""
        rules = {}
        
        # Rule 1: Success-based routing
        success_comparison = df[['category', 'difficulty', 'large_success', 'small_success']].groupby(['category', 'difficulty']).agg({
            'large_success': 'mean',
            'small_success': 'mean'
        })
        
        rules['success_based'] = success_comparison
        
        # Rule 2: Efficiency-based routing
        efficiency_comparison = df[df['efficiency_ratio'].notna()].groupby(['category', 'difficulty'])['efficiency_ratio'].mean()
        rules['efficiency_based'] = efficiency_comparison
        
        # Rule 3: Complexity threshold
        complexity_threshold = df[df['small_success'] == True]['complexity'].quantile(0.75)
        rules['complexity_threshold'] = complexity_threshold
        
        return rules
    
    def _create_performance_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a performance comparison matrix"""
        matrix = df.pivot_table(
            values=['large_success', 'small_success', 'efficiency_ratio'],
            index='category',
            columns='difficulty',
            aggfunc='mean'
        )
        return matrix
    
    def _build_similarity_graph(self, nodes: List[Dict], edges: List[Dict]) -> nx.Graph:
        """Build NetworkX graph from similarity data"""
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node['task_id'], 
                      category=node['category'],
                      difficulty=node['difficulty'],
                      complexity=node['complexity'])
        
        # Add edges
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], 
                      weight=edge['weight'],
                      similarity_type=edge.get('similarity_type', 'unknown'))
        
        return G
    
    def _calculate_centrality_measures(self, G: nx.Graph) -> Dict[str, Dict]:
        """Calculate various centrality measures"""
        return {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'closeness_centrality': nx.closeness_centrality(G),
            'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=1000)
        }
    
    def _detect_communities(self, G: nx.Graph) -> Dict[str, Any]:
        """Detect communities in the similarity graph"""
        import networkx.algorithms.community as nx_comm
        
        communities = list(nx_comm.greedy_modularity_communities(G))
        modularity = nx_comm.modularity(G, communities)
        
        return {
            'communities': communities,
            'modularity': modularity,
            'num_communities': len(communities)
        }
    
    def _calculate_network_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate overall network metrics"""
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'average_shortest_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else None
        }
    
    def _analyze_success_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze success patterns between models"""
        patterns = {}
        
        # Both succeed
        both_succeed = df[(df['success1'] == True) & (df['success2'] == True)]
        patterns['both_succeed'] = len(both_succeed) / len(df)
        
        # Only large succeeds
        only_large = df[(df['success1'] == True) & (df['success2'] == False)]
        patterns['only_large_succeeds'] = len(only_large) / len(df)
        
        # Only small succeeds
        only_small = df[(df['success1'] == False) & (df['success2'] == True)]
        patterns['only_small_succeeds'] = len(only_small) / len(df)
        
        # Both fail
        both_fail = df[(df['success1'] == False) & (df['success2'] == False)]
        patterns['both_fail'] = len(both_fail) / len(df)
        
        return patterns
    
    def _analyze_resource_trade_offs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze resource trade-offs between models"""
        trade_offs = {}
        
        # Memory efficiency
        memory_ratio = (df['memory2'] / df['memory1']).mean()
        trade_offs['memory_efficiency_ratio'] = memory_ratio
        
        # Time efficiency
        time_ratio = (df['time2'] / df['time1']).mean()
        trade_offs['time_efficiency_ratio'] = time_ratio
        
        # Success vs efficiency trade-off
        success_vs_efficiency = df[df['success1'] & df['success2']].copy()
        if not success_vs_efficiency.empty:
            efficiency_gain = ((success_vs_efficiency['memory1'] - success_vs_efficiency['memory2']) / 
                             success_vs_efficiency['memory1']).mean()
            trade_offs['efficiency_gain_when_both_succeed'] = efficiency_gain
        
        return trade_offs
    
    def _analyze_category_preferences(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze which model performs better in each category"""
        preferences = {}
        
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            large_success_rate = cat_data['success1'].mean()
            small_success_rate = cat_data['success2'].mean()
            
            if large_success_rate > small_success_rate:
                preferences[category] = 'large_model'
            elif small_success_rate > large_success_rate:
                preferences[category] = 'small_model'
            else:
                preferences[category] = 'tie'
        
        return preferences
    
    def _generate_model_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate model selection recommendations"""
        recommendations = []
        
        # Overall performance
        large_overall = df['success1'].mean()
        small_overall = df['success2'].mean()
        
        recommendations.append(f"Overall success rates - Large: {large_overall:.2%}, Small: {small_overall:.2%}")
        
        # Resource efficiency
        avg_memory_ratio = (df['memory2'] / df['memory1']).mean()
        avg_time_ratio = (df['time2'] / df['time1']).mean()
        
        recommendations.append(f"Small model uses {avg_memory_ratio:.1%} memory and {avg_time_ratio:.1%} time of large model")
        
        # Strategic recommendations
        if small_overall > 0.8 * large_overall:
            recommendations.append("✅ Small model recommended for resource-constrained environments")
        
        if large_overall - small_overall > 0.2:
            recommendations.append("⚠️ Large model significantly outperforms - consider hybrid approach")
        
        return recommendations
    
    def _identify_dominant_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify dominant resource usage patterns"""
        patterns = {}
        
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            dominant_category = cluster_data['category'].mode().iloc[0] if not cluster_data['category'].empty else 'unknown'
            avg_resources = {
                'memory': cluster_data['avg_memory'].mean(),
                'cpu': cluster_data['avg_cpu'].mean(),
                'time': cluster_data['avg_time'].mean()
            }
            
            patterns[f'cluster_{cluster}'] = {
                'dominant_category': dominant_category,
                'avg_resources': avg_resources,
                'size': len(cluster_data)
            }
        
        return patterns
    
    # Formatting methods for report generation
    def _format_efficiency_insights(self, data: Dict[str, Any]) -> List[str]:
        """Format efficiency insights for report"""
        insights = []
        clusters = data['analysis']['dominant_patterns']
        
        for cluster_id, info in clusters.items():
            insights.append(f"  {cluster_id}: {info['dominant_category']} tasks")
            insights.append(f"    Memory: {info['avg_resources']['memory']:.1f}MB")
            insights.append(f"    CPU: {info['avg_resources']['cpu']:.1f}%")
            insights.append(f"    Time: {info['avg_resources']['time']:.2f}s")
            insights.append(f"    Size: {info['size']} tasks")
            insights.append("")
        
        return insights
    
    def _format_branching_insights(self, data: Dict[str, Any]) -> List[str]:
        """Format branching insights for report"""
        return data['insights']
    
    def _format_routing_insights(self, data: Dict[str, Any]) -> List[str]:
        """Format routing insights for report"""
        insights = []
        rules = data['routing_rules']
        
        insights.append(f"  Complexity threshold for small model: {rules['complexity_threshold']:.2f}")
        insights.append("  Category-based efficiency ratios:")
        
        for category, ratio in rules['efficiency_based'].head().items():
            insights.append(f"    {category}: {ratio:.2f}")
        
        return insights
    
    def _format_network_insights(self, data: Dict[str, Any]) -> List[str]:
        """Format network insights for report"""
        insights = []
        metrics = data['analysis']['network_metrics']
        
        insights.append(f"  Network size: {metrics['num_nodes']} tasks, {metrics['num_edges']} connections")
        insights.append(f"  Network density: {metrics['density']:.3f}")
        insights.append(f"  Average clustering: {metrics['average_clustering']:.3f}")
        
        communities = data['analysis']['community_detection']
        insights.append(f"  Communities detected: {communities['num_communities']}")
        insights.append(f"  Modularity: {communities['modularity']:.3f}")
        
        return insights
    
    def _format_comparison_insights(self, data: Dict[str, Any]) -> List[str]:
        """Format comparison insights for report"""
        insights = []
        patterns = data['analysis']['success_patterns']
        
        insights.append(f"  Both models succeed: {patterns['both_succeed']:.2%}")
        insights.append(f"  Only large model succeeds: {patterns['only_large_succeeds']:.2%}")
        insights.append(f"  Only small model succeeds: {patterns['only_small_succeeds']:.2%}")
        insights.append(f"  Both models fail: {patterns['both_fail']:.2%}")
        
        recommendations = data['recommendations']
        insights.append("\n  Recommendations:")
        for rec in recommendations:
            insights.append(f"    {rec}")
        
        return insights
