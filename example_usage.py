"""
Example Usage of PDDL-INSTRUCT Framework
Demonstrates how to use the framework with example domains and problems.
"""

import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pddl_parser import PDDLParser, load_pddl_file
from logical_reasoning import LogicalReasoner
from instruction_templates import PDDLInstructionGenerator, InstructionType
from state_verification import StateVerifier


def demonstrate_blocksworld():
    """Demonstrate the framework with blocksworld domain"""
    print("=" * 60)
    print("BLOCKSWORLD DOMAIN DEMONSTRATION")
    print("=" * 60)
    
    # Load domain and problem
    parser = PDDLParser()
    domain_path = "examples/blocksworld_domain.pddl"
    problem_path = "examples/blocksworld_problem.pddl"
    
    domain_text = load_pddl_file(domain_path)
    problem_text = load_pddl_file(problem_path)
    
    domain = parser.parse_domain(domain_text)
    problem = parser.parse_problem(problem_text)
    
    print(f"Domain: {domain.name}")
    print(f"Problem: {problem.name}")
    print(f"Initial State: {', '.join(sorted(problem.init_state))}")
    print(f"Goal State: {', '.join(sorted(problem.goal_state))}")
    print()
    
    # Initialize components
    reasoner = LogicalReasoner(domain)
    instruction_generator = PDDLInstructionGenerator(domain)
    verifier = StateVerifier(domain)
    
    # Test action applicability reasoning
    print("1. ACTION APPLICABILITY REASONING")
    print("-" * 40)
    
    state = problem.init_state.copy()
    action_name = "pickup"
    parameters = {"x": "a"}
    
    reasoning_chain = reasoner.reason_about_action_applicability(action_name, parameters, state)
    
    print(f"Action: {action_name}({', '.join(parameters.values())})")
    print(f"State: {', '.join(sorted(state))}")
    print(f"Applicable: {reasoning_chain.success}")
    print(f"Explanation: {reasoning_chain.explanation}")
    print()
    
    # Test state transition
    print("2. STATE TRANSITION")
    print("-" * 40)
    
    if reasoning_chain.success:
        print(f"New State: {', '.join(sorted(reasoning_chain.final_state))}")
        print()
    
    # Test plan validation
    print("3. PLAN VALIDATION")
    print("-" * 40)
    
    # Create a simple plan
    plan = [
        "pickup(a)",
        "stack(a, b)",
        "pickup(c)",
        "stack(c, a)"
    ]
    
    print(f"Plan: {', '.join(plan)}")
    
    plan_validation = reasoner.reason_about_plan_validity(plan, problem.init_state, problem.goal_state)
    
    print(f"Plan Valid: {plan_validation.success}")
    print(f"Explanation: {plan_validation.explanation}")
    print()
    
    # Test instruction generation
    print("4. INSTRUCTION GENERATION")
    print("-" * 40)
    
    instruction = instruction_generator.generate_instruction(
        InstructionType.ACTION_APPLICABILITY,
        domain_name=domain.name,
        current_state=', '.join(sorted(state)),
        action_name=action_name,
        parameters=parameters
    )
    
    print("Generated Instruction:")
    print(instruction)
    print()
    
    # Test verification
    print("5. STATE VERIFICATION")
    print("-" * 40)
    
    verification_report = verifier.verify_action_applicability(action_name, parameters, state)
    
    print(f"Verification Result: {verification_report.result.value}")
    print(f"Details: {verification_report.details}")
    if verification_report.errors:
        print(f"Errors: {verification_report.errors}")
    print()


def demonstrate_gripper():
    """Demonstrate the framework with gripper domain"""
    print("=" * 60)
    print("GRIPPER DOMAIN DEMONSTRATION")
    print("=" * 60)
    
    # Load domain and problem
    parser = PDDLParser()
    domain_path = "examples/gripper_domain.pddl"
    problem_path = "examples/gripper_problem.pddl"
    
    domain_text = load_pddl_file(domain_path)
    problem_text = load_pddl_file(problem_path)
    
    domain = parser.parse_domain(domain_text)
    problem = parser.parse_problem(problem_text)
    
    print(f"Domain: {domain.name}")
    print(f"Problem: {problem.name}")
    print(f"Initial State: {', '.join(sorted(problem.init_state))}")
    print(f"Goal State: {', '.join(sorted(problem.goal_state))}")
    print()
    
    # Initialize components
    reasoner = LogicalReasoner(domain)
    instruction_generator = PDDLInstructionGenerator(domain)
    
    # Test action applicability reasoning
    print("1. ACTION APPLICABILITY REASONING")
    print("-" * 40)
    
    state = problem.init_state.copy()
    action_name = "pick"
    parameters = {"obj": "ball1", "room": "rooma", "gripper": "left"}
    
    reasoning_chain = reasoner.reason_about_action_applicability(action_name, parameters, state)
    
    print(f"Action: {action_name}({', '.join(parameters.values())})")
    print(f"State: {', '.join(sorted(state))}")
    print(f"Applicable: {reasoning_chain.success}")
    print(f"Explanation: {reasoning_chain.explanation}")
    print()
    
    # Test plan generation instruction
    print("2. PLAN GENERATION INSTRUCTION")
    print("-" * 40)
    
    instruction = instruction_generator.generate_instruction(
        InstructionType.PLAN_GENERATION,
        problem_name=problem.name,
        domain_name=problem.domain_name,
        initial_state=', '.join(sorted(problem.init_state)),
        goal_state=', '.join(sorted(problem.goal_state))
    )
    
    print("Generated Instruction:")
    print(instruction)
    print()


def demonstrate_logistics():
    """Demonstrate the framework with logistics domain"""
    print("=" * 60)
    print("LOGISTICS DOMAIN DEMONSTRATION")
    print("=" * 60)
    
    # Load domain and problem
    parser = PDDLParser()
    domain_path = "examples/logistics_domain.pddl"
    problem_path = "examples/logistics_problem.pddl"
    
    domain_text = load_pddl_file(domain_path)
    problem_text = load_pddl_file(problem_path)
    
    domain = parser.parse_domain(domain_text)
    problem = parser.parse_problem(problem_text)
    
    print(f"Domain: {domain.name}")
    print(f"Problem: {problem.name}")
    print(f"Initial State: {', '.join(sorted(problem.init_state))}")
    print(f"Goal State: {', '.join(sorted(problem.goal_state))}")
    print()
    
    # Initialize components
    reasoner = LogicalReasoner(domain)
    instruction_generator = PDDLInstructionGenerator(domain)
    
    # Test action applicability reasoning
    print("1. ACTION APPLICABILITY REASONING")
    print("-" * 40)
    
    state = problem.init_state.copy()
    action_name = "load-truck"
    parameters = {"p": "obj11", "t": "tru1", "l": "pos1"}
    
    reasoning_chain = reasoner.reason_about_action_applicability(action_name, parameters, state)
    
    print(f"Action: {action_name}({', '.join(parameters.values())})")
    print(f"State: {', '.join(sorted(state))}")
    print(f"Applicable: {reasoning_chain.success}")
    print(f"Explanation: {reasoning_chain.explanation}")
    print()
    
    # Test plan validation instruction
    print("2. PLAN VALIDATION INSTRUCTION")
    print("-" * 40)
    
    # Create a simple plan
    plan = [
        "load-truck(obj11, tru1, pos1)",
        "drive-truck(tru1, pos1, pos2, cit1)",
        "unload-truck(obj11, tru1, pos2)"
    ]
    
    instruction = instruction_generator.generate_instruction(
        InstructionType.PLAN_VALIDATION,
        problem_name=problem.name,
        domain_name=problem.domain_name,
        initial_state=', '.join(sorted(problem.init_state)),
        goal_state=', '.join(sorted(problem.goal_state)),
        plan=', '.join(plan)
    )
    
    print("Generated Instruction:")
    print(instruction)
    print()


def demonstrate_training_data_generation():
    """Demonstrate training data generation"""
    print("=" * 60)
    print("TRAINING DATA GENERATION DEMONSTRATION")
    print("=" * 60)
    
    # Load blocksworld domain and problem
    parser = PDDLParser()
    domain_path = "examples/blocksworld_domain.pddl"
    problem_path = "examples/blocksworld_problem.pddl"
    
    domain_text = load_pddl_file(domain_path)
    problem_text = load_pddl_file(problem_path)
    
    domain = parser.parse_domain(domain_text)
    problem = parser.parse_problem(problem_text)
    
    # Initialize instruction generator
    instruction_generator = PDDLInstructionGenerator(domain)
    
    # Generate training examples
    print("Generating training examples...")
    training_examples = instruction_generator.generate_training_examples(problem, num_examples=5)
    
    print(f"Generated {len(training_examples)} training examples:")
    print()
    
    for i, example in enumerate(training_examples, 1):
        print(f"Example {i}: {example['type']}")
        print(f"Expected Output: {example['expected_output']}")
        print(f"Instruction: {example['instruction'][:200]}...")
        print()
    
    # Export training data
    print("Exporting training data...")
    instruction_generator.export_training_data([problem], "training_data.json", examples_per_problem=10)
    print("Training data exported to training_data.json")
    print()


def main():
    """Main demonstration function"""
    print("PDDL-INSTRUCT Framework Demonstration")
    print("Based on: Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning")
    print()
    
    try:
        # Demonstrate with different domains
        demonstrate_blocksworld()
        demonstrate_gripper()
        demonstrate_logistics()
        demonstrate_training_data_generation()
        
        print("=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
