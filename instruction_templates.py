"""
Instruction Templates for PDDL-INSTRUCT Framework
Generates structured instruction templates for training LLMs on symbolic planning tasks.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import random
from pddl_parser import PDDLDomain, PDDLProblem
from logical_reasoning import LogicalReasoner, InstructionGenerator


class InstructionType(Enum):
    """Types of instruction templates"""
    ACTION_APPLICABILITY = "action_applicability"
    STATE_TRANSITION = "state_transition"
    PLAN_VALIDATION = "plan_validation"
    PLAN_GENERATION = "plan_generation"
    GOAL_VERIFICATION = "goal_verification"
    PRECONDITION_CHECK = "precondition_check"
    EFFECT_APPLICATION = "effect_application"


@dataclass
class InstructionTemplate:
    """Represents an instruction template"""
    template_type: InstructionType
    template: str
    examples: List[Dict[str, Any]]
    reasoning_steps: List[str]
    expected_output_format: str


class PDDLInstructionGenerator:
    """Generates comprehensive instruction templates for PDDL-INSTRUCT"""
    
    def __init__(self, domain: PDDLDomain):
        self.domain = domain
        self.reasoner = LogicalReasoner(domain)
        self.instruction_generator = InstructionGenerator(domain)
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all instruction templates"""
        self.templates = {
            InstructionType.ACTION_APPLICABILITY: self._create_action_applicability_template(),
            InstructionType.STATE_TRANSITION: self._create_state_transition_template(),
            InstructionType.PLAN_VALIDATION: self._create_plan_validation_template(),
            InstructionType.PLAN_GENERATION: self._create_plan_generation_template(),
            InstructionType.GOAL_VERIFICATION: self._create_goal_verification_template(),
            InstructionType.PRECONDITION_CHECK: self._create_precondition_check_template(),
            InstructionType.EFFECT_APPLICATION: self._create_effect_application_template()
        }
    
    def _create_action_applicability_template(self) -> InstructionTemplate:
        """Create template for action applicability reasoning"""
        template = """You are an expert in symbolic planning. Given a PDDL domain, current state, and an action, determine if the action is applicable.

Domain: {domain_name}
Current State: {current_state}

Action: {action_name}
Parameters: {parameters}

Please reason step by step:
1. Identify the preconditions of the action
2. Check if each precondition is satisfied in the current state
3. Determine if the action is applicable
4. If applicable, describe what effects the action would have

Provide your reasoning in the following format:
Step 1: [Precondition analysis]
Step 2: [State checking]
Step 3: [Applicability determination]
Step 4: [Effect description]

Answer: [Yes/No] - [Brief explanation]"""

        reasoning_steps = [
            "Identify action preconditions",
            "Check precondition satisfaction in current state",
            "Determine action applicability",
            "Describe potential effects"
        ]

        return InstructionTemplate(
            template_type=InstructionType.ACTION_APPLICABILITY,
            template=template,
            examples=[],
            reasoning_steps=reasoning_steps,
            expected_output_format="Step-by-step reasoning followed by Yes/No answer"
        )
    
    def _create_state_transition_template(self) -> InstructionTemplate:
        """Create template for state transition reasoning"""
        template = """You are an expert in symbolic planning. Given a PDDL domain, current state, and an action, determine the resulting state after applying the action.

Domain: {domain_name}
Current State: {current_state}

Action: {action_name}
Parameters: {parameters}

Please reason step by step:
1. Verify the action is applicable in the current state
2. Identify the positive effects of the action
3. Identify the negative effects of the action
4. Apply effects to determine the new state

Provide your reasoning in the following format:
Step 1: [Applicability check]
Step 2: [Positive effects identification]
Step 3: [Negative effects identification]
Step 4: [State transition]

New State: [List of literals in the new state]"""

        reasoning_steps = [
            "Check action applicability",
            "Identify positive effects",
            "Identify negative effects",
            "Apply state transition"
        ]

        return InstructionTemplate(
            template_type=InstructionType.STATE_TRANSITION,
            template=template,
            examples=[],
            reasoning_steps=reasoning_steps,
            expected_output_format="Step-by-step reasoning followed by new state"
        )
    
    def _create_plan_validation_template(self) -> InstructionTemplate:
        """Create template for plan validation"""
        template = """You are an expert in symbolic planning. Given a PDDL problem and a proposed plan, validate whether the plan achieves the goal.

Problem: {problem_name}
Domain: {domain_name}
Initial State: {initial_state}
Goal State: {goal_state}

Proposed Plan: {plan}

Please reason step by step:
1. Start with the initial state
2. For each action in the plan:
   a. Check if the action is applicable in the current state
   b. Apply the action's effects to get the new state
3. Verify if the final state satisfies all goals

Provide your reasoning in the following format:
Step 1: [Initial state analysis]
Step 2: [Action-by-action analysis]
Step 3: [Goal verification]

Conclusion: [Valid/Invalid] - [Explanation]"""

        reasoning_steps = [
            "Analyze initial state",
            "Check each action's applicability",
            "Apply state transitions",
            "Verify goal satisfaction"
        ]

        return InstructionTemplate(
            template_type=InstructionType.PLAN_VALIDATION,
            template=template,
            examples=[],
            reasoning_steps=reasoning_steps,
            expected_output_format="Step-by-step reasoning followed by Valid/Invalid conclusion"
        )
    
    def _create_plan_generation_template(self) -> InstructionTemplate:
        """Create template for plan generation"""
        template = """You are an expert in symbolic planning. Given a PDDL problem, generate a plan that achieves the goal.

Problem: {problem_name}
Domain: {domain_name}
Initial State: {initial_state}
Goal State: {goal_state}

Please reason step by step:
1. Analyze the goal state to identify what needs to be achieved
2. Identify the differences between initial and goal states
3. Select appropriate actions to bridge these differences
4. Ensure the plan is valid and achieves the goal

Provide your reasoning in the following format:
Step 1: [Goal analysis]
Step 2: [State difference analysis]
Step 3: [Action selection]
Step 4: [Plan validation]

Plan: [Sequence of actions]"""

        reasoning_steps = [
            "Analyze goal requirements",
            "Identify state differences",
            "Select appropriate actions",
            "Validate plan correctness"
        ]

        return InstructionTemplate(
            template_type=InstructionType.PLAN_GENERATION,
            template=template,
            examples=[],
            reasoning_steps=reasoning_steps,
            expected_output_format="Step-by-step reasoning followed by action sequence"
        )
    
    def _create_goal_verification_template(self) -> InstructionTemplate:
        """Create template for goal verification"""
        template = """You are an expert in symbolic planning. Given a current state and a goal state, verify if the goal is satisfied.

Current State: {current_state}
Goal State: {goal_state}

Please reason step by step:
1. Identify each goal condition
2. Check if each goal condition is satisfied in the current state
3. Determine overall goal satisfaction

Provide your reasoning in the following format:
Step 1: [Goal condition identification]
Step 2: [Satisfaction checking]
Step 3: [Overall assessment]

Goal Satisfied: [Yes/No] - [Explanation]"""

        reasoning_steps = [
            "Identify goal conditions",
            "Check condition satisfaction",
            "Assess overall goal satisfaction"
        ]

        return InstructionTemplate(
            template_type=InstructionType.GOAL_VERIFICATION,
            template=template,
            examples=[],
            reasoning_steps=reasoning_steps,
            expected_output_format="Step-by-step reasoning followed by Yes/No answer"
        )
    
    def _create_precondition_check_template(self) -> InstructionTemplate:
        """Create template for precondition checking"""
        template = """You are an expert in symbolic planning. Given an action and a current state, check if the action's preconditions are satisfied.

Action: {action_name}
Parameters: {parameters}
Current State: {current_state}

Please reason step by step:
1. Identify all preconditions of the action
2. Check if each precondition is satisfied in the current state
3. Determine if all preconditions are met

Provide your reasoning in the following format:
Step 1: [Precondition identification]
Step 2: [Satisfaction checking]
Step 3: [Overall assessment]

Preconditions Satisfied: [Yes/No] - [Explanation]"""

        reasoning_steps = [
            "Identify action preconditions",
            "Check precondition satisfaction",
            "Assess overall precondition satisfaction"
        ]

        return InstructionTemplate(
            template_type=InstructionType.PRECONDITION_CHECK,
            template=template,
            examples=[],
            reasoning_steps=reasoning_steps,
            expected_output_format="Step-by-step reasoning followed by Yes/No answer"
        )
    
    def _create_effect_application_template(self) -> InstructionTemplate:
        """Create template for effect application"""
        template = """You are an expert in symbolic planning. Given an action and a current state, determine the effects of applying the action.

Action: {action_name}
Parameters: {parameters}
Current State: {current_state}

Please reason step by step:
1. Identify all effects of the action
2. Separate positive and negative effects
3. Apply effects to determine the new state

Provide your reasoning in the following format:
Step 1: [Effect identification]
Step 2: [Effect categorization]
Step 3: [State application]

Effects Applied: [List of effects]
New State: [List of literals in the new state]"""

        reasoning_steps = [
            "Identify action effects",
            "Categorize positive and negative effects",
            "Apply effects to current state"
        ]

        return InstructionTemplate(
            template_type=InstructionType.EFFECT_APPLICATION,
            template=template,
            examples=[],
            reasoning_steps=reasoning_steps,
            expected_output_format="Step-by-step reasoning followed by effects and new state"
        )
    
    def generate_instruction(self, template_type: InstructionType, **kwargs) -> str:
        """Generate a specific instruction using a template"""
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        template = self.templates[template_type]
        return template.template.format(**kwargs)
    
    def generate_training_examples(self, problem: PDDLProblem, num_examples: int = 10) -> List[Dict[str, Any]]:
        """Generate training examples for the given problem"""
        examples = []
        
        # Generate action applicability examples
        for _ in range(num_examples // 4):
            example = self._generate_action_applicability_example(problem)
            if example:
                examples.append(example)
        
        # Generate state transition examples
        for _ in range(num_examples // 4):
            example = self._generate_state_transition_example(problem)
            if example:
                examples.append(example)
        
        # Generate plan validation examples
        for _ in range(num_examples // 4):
            example = self._generate_plan_validation_example(problem)
            if example:
                examples.append(example)
        
        # Generate goal verification examples
        for _ in range(num_examples // 4):
            example = self._generate_goal_verification_example(problem)
            if example:
                examples.append(example)
        
        return examples
    
    def _generate_action_applicability_example(self, problem: PDDLProblem) -> Optional[Dict[str, Any]]:
        """Generate an action applicability example"""
        if not self.domain.actions:
            return None
        
        action_name = random.choice(list(self.domain.actions.keys()))
        action = self.domain.actions[action_name]
        
        # Generate random parameters
        parameters = {}
        for param_name, param_type in action.parameters:
            # Find objects of the correct type
            suitable_objects = [obj for obj, obj_type in problem.objects.items() 
                              if obj_type == param_type]
            if suitable_objects:
                parameters[param_name] = random.choice(suitable_objects)
        
        if not parameters:
            return None
        
        # Use initial state or generate a random state
        state = problem.init_state.copy()
        
        instruction = self.generate_instruction(
            InstructionType.ACTION_APPLICABILITY,
            domain_name=self.domain.name,
            current_state=', '.join(sorted(state)),
            action_name=action_name,
            parameters=parameters
        )
        
        # Generate reasoning chain
        reasoning_chain = self.reasoner.reason_about_action_applicability(action_name, parameters, state)
        
        return {
            "type": "action_applicability",
            "instruction": instruction,
            "reasoning_chain": reasoning_chain,
            "expected_output": "Yes" if reasoning_chain.success else "No"
        }
    
    def _generate_state_transition_example(self, problem: PDDLProblem) -> Optional[Dict[str, Any]]:
        """Generate a state transition example"""
        if not self.domain.actions:
            return None
        
        action_name = random.choice(list(self.domain.actions.keys()))
        action = self.domain.actions[action_name]
        
        # Generate random parameters
        parameters = {}
        for param_name, param_type in action.parameters:
            suitable_objects = [obj for obj, obj_type in problem.objects.items() 
                              if obj_type == param_type]
            if suitable_objects:
                parameters[param_name] = random.choice(suitable_objects)
        
        if not parameters:
            return None
        
        state = problem.init_state.copy()
        
        instruction = self.generate_instruction(
            InstructionType.STATE_TRANSITION,
            domain_name=self.domain.name,
            current_state=', '.join(sorted(state)),
            action_name=action_name,
            parameters=parameters
        )
        
        reasoning_chain = self.reasoner.reason_about_action_applicability(action_name, parameters, state)
        
        return {
            "type": "state_transition",
            "instruction": instruction,
            "reasoning_chain": reasoning_chain,
            "expected_output": ', '.join(sorted(reasoning_chain.final_state))
        }
    
    def _generate_plan_validation_example(self, problem: PDDLProblem) -> Optional[Dict[str, Any]]:
        """Generate a plan validation example"""
        # Generate a simple plan
        plan = self._generate_simple_plan(problem)
        if not plan:
            return None
        
        instruction = self.generate_instruction(
            InstructionType.PLAN_VALIDATION,
            problem_name=problem.name,
            domain_name=problem.domain_name,
            initial_state=', '.join(sorted(problem.init_state)),
            goal_state=', '.join(sorted(problem.goal_state)),
            plan=', '.join(plan)
        )
        
        reasoning_chain = self.reasoner.generate_reasoning_chain(problem, plan)
        
        return {
            "type": "plan_validation",
            "instruction": instruction,
            "reasoning_chain": reasoning_chain,
            "expected_output": "Valid" if reasoning_chain.success else "Invalid"
        }
    
    def _generate_goal_verification_example(self, problem: PDDLProblem) -> Optional[Dict[str, Any]]:
        """Generate a goal verification example"""
        # Use initial state or generate a random state
        state = problem.init_state.copy()
        
        instruction = self.generate_instruction(
            InstructionType.GOAL_VERIFICATION,
            current_state=', '.join(sorted(state)),
            goal_state=', '.join(sorted(problem.goal_state))
        )
        
        # Check goal satisfaction
        unsatisfied_goals = problem.goal_state - state
        goal_satisfied = len(unsatisfied_goals) == 0
        
        return {
            "type": "goal_verification",
            "instruction": instruction,
            "reasoning_chain": None,
            "expected_output": "Yes" if goal_satisfied else "No"
        }
    
    def _generate_simple_plan(self, problem: PDDLProblem) -> Optional[List[str]]:
        """Generate a simple plan for the problem"""
        # This is a simplified plan generation - in practice, you'd use a proper planner
        plan = []
        current_state = problem.init_state.copy()
        
        # Try to find actions that move us closer to the goal
        for _ in range(5):  # Limit plan length
            applicable_actions = []
            
            for action_name, action in self.domain.actions.items():
                # Try different parameter combinations
                for obj_name, obj_type in problem.objects.items():
                    if len(action.parameters) == 1 and action.parameters[0][1] == obj_type:
                        parameters = {action.parameters[0][0]: obj_name}
                        
                        # Check if action is applicable
                        reasoning_chain = self.reasoner.reason_about_action_applicability(
                            action_name, parameters, current_state
                        )
                        
                        if reasoning_chain.success:
                            applicable_actions.append((action_name, parameters))
            
            if not applicable_actions:
                break
            
            # Choose a random applicable action
            action_name, parameters = random.choice(applicable_actions)
            plan.append(f"{action_name}({', '.join(parameters.values())})")
            
            # Update state
            reasoning_chain = self.reasoner.reason_about_action_applicability(
                action_name, parameters, current_state
            )
            current_state = reasoning_chain.final_state
            
            # Check if goal is satisfied
            if problem.goal_state.issubset(current_state):
                break
        
        return plan if plan else None
    
    def export_templates(self, file_path: str):
        """Export all templates to a JSON file"""
        templates_data = {}
        
        for template_type, template in self.templates.items():
            templates_data[template_type.value] = {
                "template": template.template,
                "reasoning_steps": template.reasoning_steps,
                "expected_output_format": template.expected_output_format
            }
        
        with open(file_path, 'w') as f:
            json.dump(templates_data, f, indent=2)
    
    def export_training_data(self, problems: List[PDDLProblem], file_path: str, examples_per_problem: int = 10):
        """Export training data for multiple problems"""
        training_data = []
        
        for problem in problems:
            examples = self.generate_training_examples(problem, examples_per_problem)
            training_data.extend(examples)
        
        with open(file_path, 'w') as f:
            json.dump(training_data, f, indent=2)


# Example usage
if __name__ == "__main__":
    from pddl_parser import PDDLParser
    
    # Create a simple domain for testing
    parser = PDDLParser()
    domain_text = """
    (define (domain blocksworld)
      (:requirements :strips)
      (:predicates
        (on ?x ?y)
        (clear ?x)
        (ontable ?x)
        (handempty)
        (holding ?x)
      )
      (:action pickup
        :parameters (?x)
        :precondition (and (clear ?x) (ontable ?x) (handempty))
        :effect (and (holding ?x) (not (ontable ?x)) (not (clear ?x)) (not (handempty)))
      )
    )
    """
    
    domain = parser.parse_domain(domain_text)
    instruction_generator = PDDLInstructionGenerator(domain)
    
    # Generate an instruction
    instruction = instruction_generator.generate_instruction(
        InstructionType.ACTION_APPLICABILITY,
        domain_name=domain.name,
        current_state="(clear a), (ontable a), (handempty)",
        action_name="pickup",
        parameters={"x": "a"}
    )
    
    print("Generated Instruction:")
    print(instruction)
