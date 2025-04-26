# LLM Resource Comparison

A framework for comparing resource consumption and performance of large vs. small language models on programming tasks.

## Project Overview

This project evaluates and compares the resource efficiency and solution quality of different sized language models (LLMs) on Python programming tasks from the MBPP (Mostly Basic Python Programming) dataset. It implements an automated workflow that:

1. Imports MBPP tasks from a JSONL file into MongoDB
2. Processes tasks using two different model sizes:
   - Large model (CodeLlama) for high accuracy
   - Small model (TinyLlama) with branching strategy for resource efficiency
3. Tracks resource usage (CPU, memory) of Ollama-based models
4. Validates generated solutions against test cases
5. Stores comprehensive evaluation results and resource metrics in MongoDB

## Architecture

The system uses a modular architecture:

- **Database Layer**: MongoDB for storing tasks and evaluation results
- **Model Layer**: 
  - OllamaModel as base class
  - LargeModel (CodeLlama) implementation
  - SmallModel (TinyLlama) implementation
- **Orchestration Layer**: 
  - WorkflowManager for high-level pipeline coordination
  - ModelManager for model instantiation and configuration
  - ModelExecutionManager for running model inferences
  - PromptFormatter for task-to-prompt conversion
  - BranchingStrategy for small model fallback approaches
  - ResourceManager for tracking resource consumption
  - SolutionEvaluator for validating the generated code
  - ResultProcessor for storing evaluation outcomes
  - DaytonaSandboxManager for secure code execution

## Prerequisites

- Python 3.8+
- MongoDB
- [Ollama](https://ollama.ai) - for running CodeLlama and TinyLlama models locally
- [Daytona](https://github.com/Daytona-Sandbox/daytona) SDK (optional) - for sandbox execution

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Pull required models via Ollama:
   ```
   ollama pull codellama
   ollama pull tinyllama
   ```
4. Ensure MongoDB is running locally

## Usage

Run the main script to start the evaluation:

```
python main.py
```

## Project Structure

```
project_root/
├── database/               # MongoDB connection and operations
│   └── mongo_client.py     # MongoDB client implementation
├── models/                 # Model implementations
│   ├── ollama_model.py     # Base class for Ollama models
│   ├── large_model.py      # CodeLlama implementation
│   └── small_model.py      # TinyLlama implementation
├── orchestrators/          # Workflow orchestration
│   ├── workflow_manager.py         # Overall workflow coordination
│   ├── model_manager.py            # Model instantiation and configuration
│   ├── model_execution_manager.py  # Model inference execution
│   ├── prompt_formatter.py         # Task-to-prompt conversion
│   ├── branching_strategy.py       # Small model fallback approaches
│   ├── resource_manager.py         # Resource consumption tracking
│   ├── solution_evaluator.py       # Code validation
│   ├── result_processor.py         # Outcome storage
│   └── daytona_sandbox_manager.py  # Secure code execution
├── resources/              # Resource tracking data
│   ├── ollama_codellama    # CodeLlama resource metrics
│   └── ollama_tinyllama    # TinyLlama resource metrics
├── utils/                  # Utility functions
│   └── ollama_resource_tracker.py  # Ollama process resource monitoring
├── main.py                 # Entry point
└── requirements.txt        # Python dependencies
```

## Resource Monitoring

The system tracks:
- CPU usage (percentage)
- Memory usage (MB)
- Duration of model execution
- Average and peak resource usage for both large and small models

## Research Focus

This project aims to answer key questions about LLM resource efficiency:
1. How much more resource-efficient are small models compared to large ones?
2. Can branching strategies with small models achieve similar accuracy to large models?
3. What is the optimal balance between resource usage and solution quality?
4. In which scenarios are small models sufficient, and when are large models necessary?

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.