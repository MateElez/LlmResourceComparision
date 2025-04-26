# LLM Resource Comparison Specifications

This document provides detailed technical specifications for the LLM Resource Comparison project.

## System Architecture

### Database Layer

- **Technology**: MongoDB
- **Collections**:
  - `tasks`: Stores MBPP programming tasks
  - `results`: Stores evaluation results and resource metrics
  - `models`: Stores model configurations
- **Implementation**: `mongo_client.py`

### Model Layer

#### Base Model (`ollama_model.py`)
- **Interface**: Abstract base class defining common model operations
- **Configuration**: Model name, endpoint URL, parameters
- **Methods**:
  - `generate`: Executes inference with provided prompt
  - `get_resource_usage`: Returns resource consumption metrics
  - `setup`: Initializes model and connections

#### Large Model (`large_model.py`)
- **Base Model**: CodeLlama via Ollama
- **Parameters**:
  - Temperature: 0.1
  - Max tokens: 2048
  - Top-p: 0.95
- **Resource Tracking**: Detailed CPU and memory usage
- **Implementation**: Extends `OllamaModel` with specializations for high-accuracy code generation

#### Small Model (`small_model.py`)
- **Base Model**: TinyLlama via Ollama
- **Parameters**:
  - Temperature: Range from 0.1 to 0.7 based on branching strategy
  - Max tokens: 1024
  - Top-p: 0.9
- **Resource Tracking**: Detailed CPU and memory usage
- **Implementation**: Extends `OllamaModel` with optimizations for resource efficiency

### Orchestration Layer

#### Workflow Manager (`workflow_manager.py`)
- **Responsibility**: Coordinates the entire evaluation pipeline
- **Process**:
  1. Retrieves tasks from database
  2. Initializes model managers
  3. Executes task processing workflow
  4. Handles error recovery and logging
  5. Aggregates and stores final results

#### Model Manager (`model_manager.py`)
- **Responsibility**: Manages model lifecycles
- **Functions**:
  - Initializes and configures model instances
  - Monitors model health
  - Implements model switching logic
  - Manages resource allocation

#### Model Execution Manager (`model_execution_manager.py`)
- **Responsibility**: Handles individual model execution runs
- **Functions**:
  - Submits prompts to models
  - Monitors execution time
  - Handles timeouts and retries
  - Captures and processes model outputs

#### Prompt Formatter (`prompt_formatter.py`)
- **Responsibility**: Converts tasks to model-specific prompts
- **Templates**:
  - Large Model: Detailed context with examples
  - Small Model: Concise instructions optimized for resource efficiency
- **Components**:
  - Task description formatter
  - Test case formatter
  - System instruction templates
  - Example solution templates

#### Branching Strategy (`branching_strategy.py`)
- **Responsibility**: Implements the small model fallback approach
- **Strategies**:
  - Exponential temperature increase (0.1 → 0.3 → 0.5 → 0.7)
  - Multi-attempt with different prompt formulations
  - Context reduction for resource efficiency
  - Progressive hint inclusion
- **Decision Logic**: Evaluates solution quality to determine when to continue or fall back

#### Resource Manager (`resource_manager.py`)
- **Responsibility**: Tracks and analyzes resource usage
- **Metrics Tracked**:
  - CPU usage (percentage)
  - Memory consumption (MB)
  - Execution time (seconds)
  - GPU utilization (if applicable)
- **Storage**: Writes time-series data to resource files

#### Solution Evaluator (`solution_evaluator.py`)
- **Responsibility**: Validates generated solutions
- **Testing Approach**:
  - Unit test execution
  - Code syntax validation
  - Output comparison
  - Error handling
- **Sandbox**: Utilizes Daytona for secure code execution

#### Result Processor (`result_processor.py`)
- **Responsibility**: Processes and stores evaluation results
- **Data Processing**:
  - Calculates success rates
  - Aggregates resource metrics
  - Computes efficiency scores
  - Generates comparative statistics
- **Storage**: Writes structured results to MongoDB

#### Daytona Sandbox Manager (`daytona_sandbox_manager.py`)
- **Responsibility**: Provides secure execution environment
- **Features**:
  - Isolated code execution
  - Resource limitation
  - Timeout enforcement
  - Standard libraries only
  - Output capture and validation

### Utilities

#### Ollama Resource Tracker (`ollama_resource_tracker.py`)
- **Responsibility**: Monitors Ollama process resource consumption
- **Implementation**:
  - Uses `psutil` for process monitoring
  - Tracks CPU and memory usage at regular intervals
  - Records peak and average usage
  - Generates time-series data for visualization

## Data Flow

1. **Task Retrieval**: Workflow Manager retrieves tasks from MongoDB
2. **Prompt Generation**: Prompt Formatter creates model-specific prompts
3. **Large Model Execution**:
   - Model Execution Manager submits to Large Model
   - Resource Manager tracks usage
   - Solution Evaluator validates outputs
4. **Small Model Execution** (parallel or sequential based on configuration):
   - Model Execution Manager submits to Small Model
   - Branching Strategy manages multiple attempts if needed
   - Resource Manager tracks usage
   - Solution Evaluator validates outputs
5. **Result Processing**:
   - Result Processor aggregates metrics
   - Comparative analysis between models
   - Stores structured results in MongoDB

## Research Methodology

### Evaluation Metrics

#### Performance Metrics
- **Success Rate**: Percentage of correctly solved tasks
- **Solution Quality**: Correctness, efficiency, and readability
- **First-attempt Success**: Success without branching/fallback

#### Resource Metrics
- **CPU Efficiency**: CPU usage per successful solution
- **Memory Efficiency**: Memory usage per successful solution
- **Time Efficiency**: Execution time per successful solution
- **Combined Efficiency Score**: Weighted combination of above metrics

### Experimentation Setup

- **Task Distribution**: 500 tasks from MBPP dataset
- **Hardware Standardization**: Consistent testing environment
- **Repeatability**: Multiple runs to account for variance
- **Baseline Comparison**: Large model (CodeLlama) as baseline
- **Variations**: Different branching strategies and small model configurations

## Implementation Details

### Programming Standards
- **Language**: Python 3.8+
- **Coding Style**: PEP 8
- **Documentation**: Google-style docstrings
- **Testing**: Unit tests for core components

### External Dependencies
- MongoDB for data storage
- Ollama for model serving
- Daytona SDK for secure execution
- psutil for resource monitoring

### Configuration
- Environment variables for sensitive information
- Configuration files for model parameters
- Command-line arguments for runtime options

## Future Enhancements

1. **Model Expansion**: Support for additional model sizes and architectures
2. **Advanced Branching**: Reinforcement learning for optimal branching decisions
3. **UI Dashboard**: Real-time monitoring and visualization
4. **Distributed Execution**: Support for parallel task processing
5. **Hyperparameter Optimization**: Automated tuning of model parameters