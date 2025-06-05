"""
Visualization module for graph analytics results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Tuple

class GraphVisualizer:
    """
    Visualization tools for LLM resource comparison graph analytics
    """
    
    def __init__(self, style='seaborn-v0_8'):
        """
        Initialize visualizer with style preferences
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = px.colors.qualitative.Set1
    
    def plot_efficiency_clusters(self, efficiency_data: Dict[str, Any], save_path: str = None):
        """
        Plot resource efficiency clusters
        
        Args:
            efficiency_data: Output from analyze_model_efficiency_clusters()
            save_path: Optional path to save the plot
        """
        df = pd.DataFrame(efficiency_data['data'])
        clusters = efficiency_data['clusters']['cluster_labels']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resource Efficiency Clusters Analysis', fontsize=16, fontweight='bold')
        
        # Memory vs CPU scatter plot
        scatter = axes[0, 0].scatter(df['avg_memory'], df['avg_cpu'], 
                                   c=clusters, cmap='viridis', alpha=0.7, s=50)
        axes[0, 0].set_xlabel('Average Memory (MB)')
        axes[0, 0].set_ylabel('Average CPU (%)')
        axes[0, 0].set_title('Memory vs CPU Usage Clusters')
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        
        # Memory vs Time scatter plot
        scatter2 = axes[0, 1].scatter(df['avg_memory'], df['avg_time'], 
                                    c=clusters, cmap='viridis', alpha=0.7, s=50)
        axes[0, 1].set_xlabel('Average Memory (MB)')
        axes[0, 1].set_ylabel('Average Time (s)')
        axes[0, 1].set_title('Memory vs Execution Time Clusters')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
        
        # Category distribution by cluster
        df_with_clusters = df.assign(cluster=clusters)
        category_cluster = pd.crosstab(df_with_clusters['category'], df_with_clusters['cluster'])
        category_cluster.plot(kind='bar', ax=axes[1, 0], stacked=True)
        axes[1, 0].set_title('Category Distribution by Cluster')
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model comparison by cluster
        model_cluster = pd.crosstab(df_with_clusters['model'], df_with_clusters['cluster'])
        model_cluster.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Model Distribution by Cluster')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_branching_effectiveness(self, branching_data: Dict[str, Any], save_path: str = None):
        """
        Plot branching strategy effectiveness analysis
        
        Args:
            branching_data: Output from analyze_branching_strategy_effectiveness()
            save_path: Optional path to save the plot
        """
        df = pd.DataFrame(branching_data['data'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Branching Strategy Effectiveness Analysis', fontsize=16, fontweight='bold')
        
        # Success rate by total attempts
        success_by_attempts = df.groupby('total_attempts')['success'].agg(['count', 'sum', 'mean'])
        axes[0, 0].bar(success_by_attempts.index, success_by_attempts['mean'], alpha=0.7)
        axes[0, 0].set_xlabel('Total Attempts')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_title('Success Rate by Number of Attempts')
        
        # Success rate by category
        category_success = df.groupby('category')['success'].mean().sort_values(ascending=True)
        axes[0, 1].barh(range(len(category_success)), category_success.values, alpha=0.7)
        axes[0, 1].set_yticks(range(len(category_success)))
        axes[0, 1].set_yticklabels(category_success.index)
        axes[0, 1].set_xlabel('Success Rate')
        axes[0, 1].set_title('Success Rate by Category')
        
        # Resource usage vs attempts
        axes[1, 0].scatter(df['total_attempts'], df['peak_memory'], alpha=0.6, c=df['success'], cmap='RdYlGn')
        axes[1, 0].set_xlabel('Total Attempts')
        axes[1, 0].set_ylabel('Peak Memory (MB)')
        axes[1, 0].set_title('Memory Usage vs Attempts (Green=Success, Red=Failure)')
        
        # Difficulty vs attempts
        difficulty_attempts = df.groupby('difficulty')['total_attempts'].agg(['mean', 'std'])
        axes[1, 1].bar(difficulty_attempts.index, difficulty_attempts['mean'], 
                      yerr=difficulty_attempts['std'], alpha=0.7, capsize=5)
        axes[1, 1].set_xlabel('Difficulty')
        axes[1, 1].set_ylabel('Average Attempts')
        axes[1, 1].set_title('Average Attempts by Task Difficulty')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_task_similarity_network(self, network_data: Dict[str, Any], save_path: str = None):
        """
        Plot task similarity network
        
        Args:
            network_data: Output from analyze_task_similarity_networks()
            save_path: Optional path to save the plot
        """
        G = network_data['graph']
        centrality = network_data['analysis']['centrality_measures']
        communities = network_data['analysis']['community_detection']['communities']
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Task Similarity Network Analysis', fontsize=16, fontweight='bold')
        
        # Network layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Plot 1: Network with community colors
        community_colors = {}
        colors = plt.cm.Set1(np.linspace(0, 1, len(communities)))
        for i, community in enumerate(communities):
            for node in community:
                community_colors[node] = colors[i]
        
        node_colors = [community_colors.get(node, 'gray') for node in G.nodes()]
        node_sizes = [centrality['degree_centrality'][node] * 1000 + 100 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              alpha=0.8, ax=axes[0])
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=axes[0])
        
        # Add labels for high centrality nodes
        high_centrality_nodes = {node: centrality['degree_centrality'][node] 
                               for node in G.nodes() 
                               if centrality['degree_centrality'][node] > 0.1}
        nx.draw_networkx_labels(G, pos, labels=high_centrality_nodes, 
                               font_size=8, ax=axes[0])
        
        axes[0].set_title('Task Similarity Network (Communities)')
        axes[0].axis('off')
        
        # Plot 2: Centrality analysis
        degree_cent = list(centrality['degree_centrality'].values())
        betweenness_cent = list(centrality['betweenness_centrality'].values())
        
        axes[1].scatter(degree_cent, betweenness_cent, alpha=0.7, s=50)
        axes[1].set_xlabel('Degree Centrality')
        axes[1].set_ylabel('Betweenness Centrality')
        axes[1].set_title('Node Centrality Analysis')
        
        # Add task IDs for outliers
        for i, (node, dc, bc) in enumerate(zip(G.nodes(), degree_cent, betweenness_cent)):
            if dc > np.percentile(degree_cent, 90) or bc > np.percentile(betweenness_cent, 90):
                axes[1].annotate(f'T{node}', (dc, bc), fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison_matrix(self, comparison_data: Dict[str, Any], save_path: str = None):
        """
        Plot model performance comparison matrix
        
        Args:
            comparison_data: Output from compare_model_performance_paths()
            save_path: Optional path to save the plot
        """
        df = pd.DataFrame(comparison_data['data'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison Matrix', fontsize=16, fontweight='bold')
        
        # Success rate comparison by category
        success_comparison = df.groupby('category').agg({
            'success1': 'mean',
            'success2': 'mean'
        }).round(3)
        
        x = np.arange(len(success_comparison.index))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, success_comparison['success1'], width, 
                      label='Large Model', alpha=0.8)
        axes[0, 0].bar(x + width/2, success_comparison['success2'], width, 
                      label='Small Model', alpha=0.8)
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_title('Success Rate Comparison by Category')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(success_comparison.index, rotation=45, ha='right')
        axes[0, 0].legend()
        
        # Resource efficiency scatter
        df_success = df[(df['success1'] == True) & (df['success2'] == True)]
        if not df_success.empty:
            memory_ratio = df_success['memory2'] / df_success['memory1']
            time_ratio = df_success['time2'] / df_success['time1']
            
            scatter = axes[0, 1].scatter(memory_ratio, time_ratio, 
                                       c=df_success['difficulty'].astype('category').cat.codes, 
                                       cmap='viridis', alpha=0.7, s=50)
            axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal Time')
            axes[0, 1].axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Equal Memory')
            axes[0, 1].set_xlabel('Memory Ratio (Small/Large)')
            axes[0, 1].set_ylabel('Time Ratio (Small/Large)')
            axes[0, 1].set_title('Resource Efficiency (Both Models Succeed)')
            axes[0, 1].legend()
            plt.colorbar(scatter, ax=axes[0, 1], label='Difficulty')
        
        # Success pattern distribution
        patterns = comparison_data['analysis']['success_patterns']
        pattern_labels = ['Both Succeed', 'Only Large', 'Only Small', 'Both Fail']
        pattern_values = [patterns['both_succeed'], patterns['only_large_succeeds'], 
                         patterns['only_small_succeeds'], patterns['both_fail']]
        
        axes[1, 0].pie(pattern_values, labels=pattern_labels, autopct='%1.1f%%', 
                      startangle=90, colors=self.colors[:4])
        axes[1, 0].set_title('Success Pattern Distribution')
        
        # Category preferences heatmap
        preferences = comparison_data['analysis']['category_preferences']
        pref_df = pd.DataFrame(list(preferences.items()), columns=['Category', 'Preference'])
        pref_matrix = pd.crosstab(pref_df['Preference'], pref_df['Category'])
        
        if not pref_matrix.empty:
            sns.heatmap(pref_matrix.T, annot=True, cmap='RdYlBu', ax=axes[1, 1])
            axes[1, 1].set_title('Model Preferences by Category')
            axes[1, 1].set_xlabel('Preferred Model')
            axes[1, 1].set_ylabel('Category')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, analytics_data: Dict[str, Any]) -> go.Figure:
        """
        Create an interactive Plotly dashboard
        
        Args:
            analytics_data: Combined analytics data from all analyses
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Resource Efficiency', 'Success Rates', 
                          'Network Metrics', 'Branching Effectiveness'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Add efficiency data if available
        if 'efficiency_data' in analytics_data:
            df = pd.DataFrame(analytics_data['efficiency_data']['data'])
            
            # Resource efficiency scatter
            fig.add_trace(
                go.Scatter(
                    x=df['avg_memory'],
                    y=df['avg_cpu'],
                    mode='markers',
                    text=df['category'],
                    marker=dict(
                        size=df['avg_time'] * 10,
                        color=df['model'].astype('category').cat.codes,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Resource Usage'
                ),
                row=1, col=1
            )
        
        # Add success rate data if available
        if 'comparison_data' in analytics_data:
            comp_df = pd.DataFrame(analytics_data['comparison_data']['data'])
            success_by_cat = comp_df.groupby('category').agg({
                'success1': 'mean',
                'success2': 'mean'
            })
            
            fig.add_trace(
                go.Bar(
                    x=success_by_cat.index,
                    y=success_by_cat['success1'],
                    name='Large Model',
                    marker_color='blue'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=success_by_cat.index,
                    y=success_by_cat['success2'],
                    name='Small Model',
                    marker_color='orange'
                ),
                row=1, col=2
            )
        
        # Add network metrics if available
        if 'network_data' in analytics_data:
            network_metrics = analytics_data['network_data']['analysis']['network_metrics']
            
            metrics_names = list(network_metrics.keys())
            metrics_values = [v for v in network_metrics.values() if v is not None]
            
            fig.add_trace(
                go.Scatter(
                    x=metrics_names[:len(metrics_values)],
                    y=metrics_values,
                    mode='markers+lines',
                    name='Network Metrics',
                    marker=dict(size=10, color='green')
                ),
                row=2, col=1
            )
        
        # Add branching effectiveness if available
        if 'branching_data' in analytics_data:
            branch_df = pd.DataFrame(analytics_data['branching_data']['data'])
            success_by_attempts = branch_df.groupby('total_attempts')['success'].mean()
            
            fig.add_trace(
                go.Bar(
                    x=success_by_attempts.index,
                    y=success_by_attempts.values,
                    name='Success by Attempts',
                    marker_color='red'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="LLM Resource Comparison - Interactive Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def save_all_visualizations(self, analytics_results: Dict[str, Any], output_dir: str = "visualizations"):
        """
        Save all visualizations to files
        
        Args:
            analytics_results: Dictionary containing all analytics results
            output_dir: Directory to save visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot efficiency clusters
        if 'efficiency_data' in analytics_results:
            self.plot_efficiency_clusters(
                analytics_results['efficiency_data'], 
                f"{output_dir}/efficiency_clusters.png"
            )
        
        # Plot branching effectiveness
        if 'branching_data' in analytics_results:
            self.plot_branching_effectiveness(
                analytics_results['branching_data'],
                f"{output_dir}/branching_effectiveness.png"
            )
        
        # Plot network analysis
        if 'network_data' in analytics_results:
            self.plot_task_similarity_network(
                analytics_results['network_data'],
                f"{output_dir}/task_similarity_network.png"
            )
        
        # Plot model comparison
        if 'comparison_data' in analytics_results:
            self.plot_model_comparison_matrix(
                analytics_results['comparison_data'],
                f"{output_dir}/model_comparison_matrix.png"
            )
        
        # Save interactive dashboard
        if any(key in analytics_results for key in ['efficiency_data', 'comparison_data', 'network_data', 'branching_data']):
            dashboard = self.create_interactive_dashboard(analytics_results)
            dashboard.write_html(f"{output_dir}/interactive_dashboard.html")
        
        print(f"All visualizations saved to {output_dir}/")
    
    def generate_summary_report(self, analytics_results: Dict[str, Any]) -> str:
        """
        Generate a summary report with key insights
        
        Args:
            analytics_results: Dictionary containing all analytics results
            
        Returns:
            Formatted summary report string
        """
        report = []
        report.append("="*80)
        report.append("LLM RESOURCE COMPARISON - VISUAL ANALYTICS SUMMARY")
        report.append("="*80)
        
        # Efficiency insights
        if 'efficiency_data' in analytics_results:
            clusters = analytics_results['efficiency_data']['analysis']['cluster_sizes']
            report.append(f"\n🎯 EFFICIENCY CLUSTERS:")
            report.append(f"   Identified {len(clusters)} distinct resource usage patterns")
            for cluster, size in clusters.items():
                report.append(f"   Cluster {cluster}: {size} tasks")
        
        # Success rate insights
        if 'comparison_data' in analytics_results:
            patterns = analytics_results['comparison_data']['analysis']['success_patterns']
            report.append(f"\n📊 SUCCESS PATTERNS:")
            report.append(f"   Both models succeed: {patterns['both_succeed']:.1%}")
            report.append(f"   Only large succeeds: {patterns['only_large_succeeds']:.1%}")
            report.append(f"   Only small succeeds: {patterns['only_small_succeeds']:.1%}")
        
        # Network insights
        if 'network_data' in analytics_results:
            network_metrics = analytics_results['network_data']['analysis']['network_metrics']
            communities = analytics_results['network_data']['analysis']['community_detection']
            report.append(f"\n🕸️ NETWORK STRUCTURE:")
            report.append(f"   {network_metrics['num_nodes']} tasks with {network_metrics['num_edges']} similarities")
            report.append(f"   Network density: {network_metrics['density']:.3f}")
            report.append(f"   Communities detected: {communities['num_communities']}")
        
        # Branching insights
        if 'branching_data' in analytics_results:
            insights = analytics_results['branching_data']['insights']
            report.append(f"\n🌳 BRANCHING STRATEGY:")
            for insight in insights:
                report.append(f"   {insight}")
        
        report.append("\n" + "="*80)
        report.append("All visualizations generated and ready for analysis!")
        report.append("="*80)
        
        return "\n".join(report)
