# MBPP Task Evaluation Pipeline

A comprehensive pipeline for evaluating Language Models (LLMs) on MBPP (Mostly Basic Python Programming) tasks with resource tracking and performance comparison.

## Project Overview

This project implements an automated workflow that:

1. Imports MBPP tasks from a JSONL file into MongoDB
2. Processes tasks one by one through multiple models
3. Runs large models (DeepSeek-7B) in Docker containers to track resource usage
4. Evaluates solutions with a separate LLM evaluator model
5. Tries smaller models (Qwen-0.5B) with fallback options if solutions fail
6. Stores comprehensive evaluation results and resource metrics in MongoDB

## Architecture

The system is built with a modular architecture:

- **Database Layer**: MongoDB integration for storing tasks and results
- **Model Layer**: Abstraction for different LLM implementations
- **Orchestration Layer**: Workflow and Docker container management
- **Utilities**: Resource tracking, JSON processing, and configuration

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- MongoDB instance (or use the provided Docker container)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mbpp-evaluation.git
   cd mbpp-evaluation
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up model directories:
   ```
   mkdir -p models_data/deepseek models_data/qwen models_data/evaluator
   ```

4. Download the required models:
   - DeepSeek-7B into `models_data/deepseek/`
   - Qwen-0.5B into `models_data/qwen/`
   - A suitable evaluator model into `models_data/evaluator/`

5. Prepare the MBPP dataset:
   ```
   mkdir -p data
   # Place the mbpp.jsonl file in the data directory
   ```

## Configuration

Customize the configuration in `config/config.yaml` to match your environment:

- MongoDB connection settings
- Model paths and container configurations
- Resource monitoring options
- Docker container resource limits

## Usage

### Starting the Services

Start the required Docker containers:

```
docker-compose up -d
```

This will launch:
- MongoDB database
- DeepSeek model container
- Qwen model container
- Evaluator model container

### Running the Evaluation Pipeline

Import MBPP tasks and run the evaluation:

```
python main.py --import-data
```

To run only the evaluation without importing data:

```
python main.py
```

### Viewing Results

Results are stored in MongoDB in the following collections:
- `tasks`: The imported MBPP tasks
- `evaluation_results`: Results of model evaluations
- `resource_usage`: Resource usage metrics for each model run

## Docker Containers

The project includes Docker configurations for:

1. **DeepSeek Container**: Serves the large model (7B) via an API
2. **Qwen Container**: Serves the small model (0.5B) via an API
3. **MongoDB Container**: Stores tasks and evaluation results

## Project Structure

```
project_root/
├── config/                 # Configuration files
├── database/               # Database connection and operations
├── docker/                 # Docker configurations for models
├── models/                 # Model implementations
├── orchestrator/           # Workflow orchestration
├── utils/                  # Utility functions
├── docker-compose.yml      # Docker services configuration
├── main.py                 # Entry point
└── requirements.txt        # Python dependencies
```

## Resource Monitoring

The system tracks:
- CPU usage (percentage)
- Memory usage (MB)
- Duration of model execution
- Average and peak resource usage

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.