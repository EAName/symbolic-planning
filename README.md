# PDDL-INSTRUCT Framework

A comprehensive implementation of the PDDL-INSTRUCT framework for teaching Large Language Models (LLMs) to perform symbolic planning through logical chain-of-thought reasoning.

## Overview

This framework implements the core concepts from the paper "Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning" (arXiv:2509.13351). It provides:

- **PDDL Domain and Problem Parsing**: Parse PDDL domain and problem files
- **Logical Chain-of-Thought Reasoning**: Step-by-step reasoning about action applicability, state transitions, and plan validity
- **Instruction Templates**: Generate structured instruction templates for training LLMs
- **State Verification**: Rigorous verification of state transitions and action applicability
- **Evaluation Metrics**: Comprehensive benchmarking and evaluation tools

## Features

### 1. PDDL Parser (`pddl_parser.py`)
- Parse PDDL domain files (actions, predicates, types, requirements)
- Parse PDDL problem files (objects, initial state, goal state)
- Support for STRIPS planning domains

### 2. Logical Reasoning (`logical_reasoning.py`)
- Chain-of-thought reasoning for action applicability
- State transition reasoning
- Plan validation reasoning
- Step-by-step logical inference

### 3. Instruction Templates (`instruction_templates.py`)
- Generate instruction templates for different planning tasks
- Support for action applicability, state transition, plan validation, and plan generation
- Training data generation for LLM instruction tuning

### 4. State Verification (`state_verification.py`)
- Verify action applicability in given states
- Verify state transitions
- Verify plan execution
- Comprehensive error reporting and reasoning traces

### 5. Evaluation Framework (`evaluation.py`)
- Planning accuracy evaluation
- Reasoning accuracy evaluation
- Execution time measurement
- Plan length analysis
- Comprehensive benchmarking suite

## Installation

```bash
# Clone the repository
git clone https://github.com/EAName/symbolic-planning
cd symbolic-planning

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from pddl_parser import PDDLParser, load_pddl_file
from logical_reasoning import LogicalReasoner
from instruction_templates import PDDLInstructionGenerator, InstructionType

# Load domain and problem
parser = PDDLParser()
domain = parser.parse_domain(load_pddl_file("examples/blocksworld_domain.pddl"))
problem = parser.parse_problem(load_pddl_file("examples/blocksworld_problem.pddl"))

# Initialize components
reasoner = LogicalReasoner(domain)
instruction_generator = PDDLInstructionGenerator(domain)

# Test action applicability
state = problem.init_state.copy()
reasoning_chain = reasoner.reason_about_action_applicability("pickup", {"x": "a"}, state)

print(f"Action applicable: {reasoning_chain.success}")
print(f"Explanation: {reasoning_chain.explanation}")
```

### Running Examples

```bash
# Run the comprehensive demonstration
python example_usage.py

# Run evaluation benchmarks
python evaluation.py
```

### Example Domains

The framework includes several example domains:

1. **Blocksworld**: Classic blocks world planning domain
2. **Gripper**: Robot gripper domain for moving objects
3. **Logistics**: Package delivery domain with trucks and airplanes

## Architecture

### Core Components

1. **PDDL Parser**: Parses PDDL domain and problem files into structured data
2. **Logical Reasoner**: Implements chain-of-thought reasoning for planning tasks
3. **State Manager**: Manages state transitions and reasoning about states
4. **Instruction Generator**: Creates instruction templates for LLM training
5. **State Verifier**: Verifies correctness of state transitions and plans
6. **Evaluator**: Provides comprehensive evaluation metrics

### Reasoning Chain Structure

Each reasoning chain consists of:
- **Precondition Check**: Verify action preconditions are satisfied
- **Effect Application**: Apply action effects to current state
- **State Transition**: Update state based on effects
- **Goal Verification**: Check if goal conditions are met

## Evaluation Metrics

The framework provides several evaluation metrics:

1. **Planning Accuracy**: Percentage of correctly solved planning problems
2. **Reasoning Accuracy**: Accuracy of reasoning steps in the chain-of-thought
3. **Execution Time**: Time taken to execute plans
4. **Plan Length**: Length of generated plans
5. **Reasoning Steps**: Number of reasoning steps required
6. **Verification Success**: Success rate of plan verification

## Training Data Generation

The framework can generate training data for LLM instruction tuning:

```python
# Generate training examples
instruction_generator = PDDLInstructionGenerator(domain)
training_examples = instruction_generator.generate_training_examples(problem, num_examples=10)

# Export training data
instruction_generator.export_training_data([problem], "training_data.json")
```

## Benchmarking

Run comprehensive benchmarks:

```python
from evaluation import BenchmarkSuite, PDDLEvaluator

# Create benchmark suite
benchmark_suite = BenchmarkSuite()
benchmark_suite.add_benchmark(domain, problem, predicted_plan)

# Run benchmarks
results = benchmark_suite.run_benchmarks()

# Generate report
benchmark_suite.print_report()
```

## File Structure

```
symbolic-planning/
├── pddl_parser.py              # PDDL domain and problem parser
├── logical_reasoning.py        # Logical chain-of-thought reasoning
├── instruction_templates.py    # Instruction template generation
├── state_verification.py       # State transition verification
├── evaluation.py               # Evaluation metrics and benchmarking
├── example_usage.py            # Comprehensive usage examples
├── examples/                   # Example PDDL domains and problems
│   ├── blocksworld_domain.pddl
│   ├── blocksworld_problem.pddl
│   ├── gripper_domain.pddl
│   ├── gripper_problem.pddl
│   ├── logistics_domain.pddl
│   └── logistics_problem.pddl
└── README.md                   # This file
```

## Key Features

### Logical Chain-of-Thought Reasoning

The framework implements explicit logical inference steps:

1. **Precondition Analysis**: Check if action preconditions are satisfied
2. **State Checking**: Verify current state against preconditions
3. **Effect Application**: Apply positive and negative effects
4. **State Transition**: Update state based on effects
5. **Goal Verification**: Check if goal conditions are met

### Instruction Templates

Generate structured instructions for different planning tasks:

- Action Applicability: "Is action X applicable in state Y?"
- State Transition: "What happens when action X is applied to state Y?"
- Plan Validation: "Does plan P achieve goal G from initial state I?"
- Plan Generation: "Generate a plan to achieve goal G from initial state I"

### Verification and Validation

Comprehensive verification includes:

- Parameter type checking
- Precondition satisfaction verification
- Effect application verification
- State transition correctness
- Plan execution validation
- Goal satisfaction checking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite the original paper:

```bibtex
@article{pddl_instruct_2024,
  title={Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning},
  author={[Authors]},
  journal={arXiv preprint arXiv:2509.13351},
  year={2024}
}
```

## Acknowledgments

This implementation is based on the research paper "Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning" (arXiv:2509.13351). The framework provides a comprehensive implementation of the PDDL-INSTRUCT approach for enhancing LLM capabilities in symbolic planning.
