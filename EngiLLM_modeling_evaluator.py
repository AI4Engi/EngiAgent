import pandas as pd
import json
import requests
import time
import random
from typing import Optional, Dict, List, Any, Tuple
import re
import os
import sys
from datetime import datetime
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

# Import unified LLM API call function
try:
    from specialized_agents import call_llm_api
    HAS_LLM_API = True
except ImportError:
    print("Warning: specialized_agents not available. Will use fallback implementation.")
    HAS_LLM_API = False

# Try to import OpenAI, if failed then skip (evaluator mainly uses Gemini API)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    print("Warning: OpenAI library not available. Only Gemini API will be supported.")
    HAS_OPENAI = False

# Set log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EngiLLMModelingEvaluator:
    """EngiLLM modeling quality evaluator - directly read evaluation criteria from CSV"""
    
    def __init__(self, evaluation_model: str = "gpt-4.1-nano"):
        # Evaluation specific model configuration
        self.evaluation_model = evaluation_model
        logger.info(f"Initialize EngiLLM evaluator, using evaluation model: {self.evaluation_model}")
        
        # Gemini API key pool (only as backup)
        self.GEMINI_API_KEYS = [
            # Input your own Gemini API key here
        ]
        
        # EngiLLM specific evaluation dimensions
        self.EVALUATION_DIMENSIONS = {
            'Information_Extraction': 'Information extraction quality',
            'Domain_Specific_Reasoning': 'Domain-specific reasoning',
            'Multi_Objective_Decision': 'Multi-objective decision-making',
            'Uncertainty_Handling': 'Uncertainty handling',
            'Feasibility': 'Feasibility evaluation',
            'Numerical_Solution': 'Numerical solution quality evaluation'
        }
        
    def load_evaluation_criteria_from_json(self, json_file: str = "simple_test_problems.json", problem_index: int = 0) -> Dict[str, str]:
        """
        Load evaluation criteria for a specific problem from a JSON file
        
        Args:
            json_file: JSON file path
            problem_index: Problem index (0-based)
            
        Returns:
            Dictionary containing evaluation criteria
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            problems = data.get('problems', [])
            
            if problem_index >= len(problems):
                raise ValueError(f"Problem index {problem_index} exceeds available problems ({len(problems)})")
            
            problem = problems[problem_index]
            evaluation_criteria = problem.get('evaluation_criteria', {})
            
            criteria = {
                'problem_id': problem.get('id', f'P{problem_index + 1}'),
                'problem': problem.get('description', ''),
                'information_extraction': evaluation_criteria.get('information_extraction', ''),
                'domain_specific_reasoning': evaluation_criteria.get('domain_specific_reasoning', ''),
                'multi_objective_decision': evaluation_criteria.get('multi_objective_decision', ''),
                'uncertainty_handling': evaluation_criteria.get('uncertainty_handling', ''),
                'feasibility': problem.get('feasibility_criteria', ''),  # Add feasibility standard
                'numerical_solution': '',  # Numerical solution uses default standard
                'field': problem.get('metadata', {}).get('field', ''),
                'reference': problem.get('metadata', {}).get('reference', '')
            }
            
            logger.info(f"Successfully loaded evaluation criteria for problem index {problem_index} ({criteria['problem_id']})")
            return criteria
            
        except Exception as e:
            logger.error(f"Error loading evaluation criteria from JSON: {str(e)}")
            raise

    def load_evaluation_criteria_from_csv(self, csv_file: str, problem_id: str = "P1") -> Dict[str, str]:
        """
        Load evaluation criteria for a specific problem from a CSV file
        
        Args:
            csv_file: CSV file path
            problem_id: Problem ID (e.g. "P1", "P2" etc.)
            
        Returns:
            Dictionary containing evaluation criteria
        """
        try:
            df = pd.read_csv(csv_file)
            
            # Find the corresponding problem row
            problem_row = df[df['ID'].str.strip() == problem_id.strip()]
            
            if problem_row.empty:
                raise ValueError(f"Problem ID '{problem_id}' not found in CSV file")
            
            row_data = problem_row.iloc[0]
            
            criteria = {
                'problem_id': problem_id,
                'problem': row_data['Problem'] if pd.notna(row_data['Problem']) else "",
                'information_extraction': row_data['Information Extraction'] if pd.notna(row_data['Information Extraction']) else "",
                'domain_specific_reasoning': row_data['Domain-Specific Reasoning'] if pd.notna(row_data['Domain-Specific Reasoning']) else "",
                'multi_objective_decision': row_data['Multi-Objective Decision-Making'] if pd.notna(row_data['Multi-Objective Decision-Making']) else "",
                'uncertainty_handling': row_data['Uncertainty Handling'] if pd.notna(row_data['Uncertainty Handling']) else "",
                'feasibility': row_data['Feasibility'] if pd.notna(row_data['Feasibility']) else "",
                'numerical_solution': row_data.get('Numerical Solution', '') if pd.notna(row_data.get('Numerical Solution', '')) else "",
                'field': row_data['Field'] if pd.notna(row_data['Field']) else "",
                'reference': row_data['Reference'] if pd.notna(row_data['Reference']) else ""
            }
            
            logger.info(f"Successfully loaded evaluation criteria for problem {problem_id}")
            logger.info(f"Problem field: {criteria['field']}")
            
            return criteria
            
        except Exception as e:
            logger.error(f"Error loading evaluation criteria: {str(e)}")
            raise

    def list_available_problems(self, csv_file: str) -> List[str]:
        """
        List all available problem IDs in the CSV file
        
        Args:
            csv_file: CSV file path
            
        Returns:
            Problem ID list
        """
        try:
            df = pd.read_csv(csv_file)
            problem_ids = df['ID'].dropna().str.strip().tolist()
            logger.info(f"Found {len(problem_ids)} problems in CSV file")
            return problem_ids
        except Exception as e:
            logger.error(f"Error listing problems: {str(e)}")
            return []
        
    def _truncate_content_if_needed(self, content: str, max_length: int = 30000) -> str:
        """Truncate content if it is too long"""
        if len(content) <= max_length:
            return content
            
        logger.warning(f"⚠️ Evaluation content too long ({len(content)} characters), truncating to {max_length} characters")
        # keep head and tail
        head_size = int(max_length * 0.6)
        tail_size = int(max_length * 0.3)
        truncated = (content[:head_size] + 
                    f"\n\n... [Content truncated, original length {len(content)} characters] ...\n\n" + 
                    content[-tail_size:])
        return truncated

    def call_llm_api(self, prompt: str, retry_attempts: int = 10, retry_delay: int = 30) -> Optional[str]:
        """
        Call the configured evaluation model for evaluation - using unified LLM API
        
        Args:
            prompt: Evaluation prompt
            retry_attempts: Retry attempts
            retry_delay: Retry delay
            
        Returns:
            API response result
        """
        if HAS_LLM_API:
            # Use unified call_llm_api function to call the configured evaluation model
            try:
                # Check and truncate overly long prompt
                prompt = self._truncate_content_if_needed(prompt, 50000)
                
                # Convert evaluation_model to call_llm_api supported format
                model_type = self._convert_evaluation_model_to_api_type(self.evaluation_model)
                
                logger.info(f"[{self.evaluation_model}] Call unified LLM API for evaluation")
                response = call_llm_api(prompt, model_type)
                if response:
                    logger.info(f"[{self.evaluation_model}] API call successful")
                    return response
                else:
                    logger.warning(f"[{self.evaluation_model}] API call returned empty result")
                    return None
            except Exception as e:
                logger.error(f"{self.evaluation_model} API call error: {str(e)}")
                return None
        else:
            logger.error("Unified LLM API is not available, cannot perform evaluation")
            return None
    
    def _convert_evaluation_model_to_api_type(self, evaluation_model: str) -> str:
        """
        Convert evaluation model name to call_llm_api supported model type
        
        Args:
            evaluation_model: Evaluation model name (e.g. "gpt-4.1-nano")
            
        Returns:
            call_llm_api supported model type
        """
        model_mapping = {
            "gpt-4.1-nano": "gpt4.1nano",
            "gpt-4o": "gpt4o", 
            "gpt-4": "gpt4",
            "gpt-5": "gpt5",
            "gemini": "gemini",
            "deepseek": "deepseek"
        }
        
        return model_mapping.get(evaluation_model, "gpt4o")  # Default to gpt4o
        
    def evaluate_feasibility_dimension(self, report_content: str, feasibility_requirements: str, 
                                      problem_context: str = "") -> Optional[Dict[str, any]]:
        """
        Specialized method for evaluating the Feasibility dimension
        
        Args:
            report_content: Report content
            feasibility_requirements: Feasibility requirements
            problem_context: Problem context
            
        Returns:
            Evaluation result dictionary, containing feasibility analysis and numerical solution judgment
        """
        if not feasibility_requirements.strip():
            logger.warning("No feasibility requirements provided")
            return None
        
        prompt = f"""
Please analyze whether the generated report satisfies the given feasibility requirements.

**Problem Context:**
{problem_context}

**Feasibility Requirements:**
{feasibility_requirements}

**Report Content:**
{report_content}

Please respond in the following JSON format:

{{
    "score": <score between 0-10, can use decimal points for precision>,
    "has_requirements": true,
    "overall_satisfaction": "fully_satisfied/partially_satisfied/not_satisfied/cannot_determine",
    "satisfaction_score": <0.0-1.0>,
    "has_numerical_solution": true/false,
    "numerical_solution_details": {{
        "has_objective_value": true/false,
        "has_variable_values": true/false,
        "solution_completeness": "complete/partial/missing",
        "solution_evidence": "specific evidence from the report"
    }},
    "detailed_analysis": [
        {{
            "requirement_number": 1,
            "requirement": "specific requirement description from feasibility requirements",
            "satisfied": true/false,
            "evidence": "relevant evidence from the report or why no violation found",
            "explanation": "detailed explanation of whether this requirement is violated or respected",
            "violation_severity": "none/minor/major"
        }}
    ],
    "violations_found": [
        {{
            "requirement_number": 1,
            "violation_description": "specific violation description",
            "severity": "minor/major",
            "reasoning": "detailed reasoning for why this is considered a violation"
        }}
    ],
    "constraints_respected": [
        {{
            "requirement_number": 2,
            "constraint_description": "constraint that is clearly respected",
            "evidence": "specific evidence from report"
        }}
    ],
    "detailed_breakdown": {{
        "strengths": ["List of specific strengths found in the report"],
        "weaknesses": ["List of specific weaknesses or missing elements"],
        "key_observations": ["Important observations about the report quality"]
    }},
    "justification": "Detailed explanation of the evaluation and scoring",
    "improvement_suggestions": ["Specific suggestions for improving feasibility compliance"],
    "overall_assessment": "Overall assessment explanation"
}}

**Analysis Points:**
1. **Primary Focus**: Check if the report contains FATAL VIOLATIONS that make the solution fundamentally infeasible
2. **Violation Definition - ONLY count as violations if it's FATAL**:
   - The solution explicitly contradicts a requirement AND makes it impossible to implement
   - The mathematical model violates fundamental physical/logical constraints
   - The proposed approach is fundamentally impossible or infeasible
3. **What is NOT a violation**:
   - Missing implementation details (as long as the approach is still feasible)
   - Incomplete descriptions (as long as the core solution is valid)
   - Technical details not explicitly mentioned (assumptions can fill gaps)
   - Minor inconsistencies that don't affect overall feasibility
4. **Satisfaction Logic**: 
   - "fully_satisfied" = No fatal violations found, solution is fundamentally feasible
   - "partially_satisfied" = ONLY if there are non-fatal issues that slightly reduce feasibility
   - "not_satisfied" = ONLY if there are fatal violations that make solution impossible
   - "cannot_determine" = Insufficient information to assess feasibility
5. **Requirement Numbering**: For each feasibility requirement, identify it by number (1, 2, 3, etc.) and analyze separately
6. **Evidence-Based Assessment**: Focus on what IS implemented rather than what's missing from the description
7. **Severity Classification**: 
   - "Minor" = Non-fatal issue, solution still feasible with reasonable assumptions
   - "Major" = Fatal violation that makes solution fundamentally impossible

**CRITICAL RULE**: If the solution can be made to work with reasonable assumptions or minor adjustments, it is NOT a violation. Only count violations that make the solution fundamentally impossible to implement.

**SCORING INSTRUCTION**: Your score should be calculated as (satisfied_constraints / total_constraints) × 10. Count each numbered requirement as one constraint. Only requirements with FATAL violations should be marked as unsatisfied.

Please analyze each numbered requirement separately and only flag FATAL violations that make the solution impossible.
"""

        try:
            response = self.call_llm_api(prompt)
            if response:
                # Try to extract JSON part
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                    try:
                        result = json.loads(json_str)
                        
                        # Validate result format, relax requirements
                        if 'score' in result:
                            # Ensure basic fields exist
                            if 'overall_satisfaction' not in result:
                                result['overall_satisfaction'] = 'cannot_determine'
                            if 'detailed_breakdown' not in result:
                                result['detailed_breakdown'] = {'strengths': [], 'weaknesses': [], 'key_observations': []}
                            if 'justification' not in result:
                                result['justification'] = 'Evaluation completed with partial information'
                            if 'violations_found' not in result:
                                result['violations_found'] = []
                            if 'constraints_respected' not in result:
                                result['constraints_respected'] = []
                            if 'detailed_analysis' not in result:
                                result['detailed_analysis'] = []
                            
                            # ✅ Recalculate feasibility score: satisfied constraints/total constraints
                            violations = result.get('violations_found', [])
                            respected = result.get('constraints_respected', [])
                            
                            # ✅ Count all violations (regardless of severity level)
                            total_violations = 0
                            fatal_violations = 0  # Only used for scoring calculation
                            
                            if isinstance(violations, list):
                                for v in violations:
                                    if isinstance(v, dict):
                                        total_violations += 1  # ✅ Count all violations
                                        if v.get('severity') == 'major':
                                            fatal_violations += 1  # Only used for scoring calculation
                                    elif isinstance(v, str):
                                        total_violations += 1  # ✅ Count all violations
                                        fatal_violations += 1  # Compatible with old format, assume all are fatal
                            
                            # Calculate actual total number of constraints from LLM returned data
                            all_req_numbers = set()
                            
                            # Collect all requirement_number from detailed analysis
                            detailed_analysis = result.get('detailed_analysis', [])
                            for item in detailed_analysis:
                                if isinstance(item, dict) and 'requirement_number' in item:
                                    all_req_numbers.add(item['requirement_number'])
                            
                            # Collect requirement_number from violations
                            for v in violations:
                                if isinstance(v, dict) and 'requirement_number' in v:
                                    all_req_numbers.add(v['requirement_number'])
                            
                            # Collect requirement_number from respected
                            for r in respected:
                                if isinstance(r, dict) and 'requirement_number' in r:
                                    all_req_numbers.add(r['requirement_number'])
                            
                            # Use actual recognized requirement number
                            total_requirements = len(all_req_numbers) if all_req_numbers else 6
                            
                            # ✅ Satisfied constraints = total - all violations (for CSV statistics)
                            satisfied_constraints_for_csv = max(0, total_requirements - total_violations)
                            
                            # Satisfied constraints = total - fatal violations (for scoring calculation)
                            satisfied_constraints = max(0, total_requirements - fatal_violations)
                            
                            # Recalculate score: full score if no fatal violations, otherwise calculate proportionally
                            if fatal_violations == 0:
                                new_score = 10.0  # No fatal violations is full score
                                result['overall_satisfaction'] = 'fully_satisfied'
                            else:
                                new_score = (satisfied_constraints / total_requirements) * 10.0
                                if satisfied_constraints >= total_requirements * 0.7:
                                    result['overall_satisfaction'] = 'partially_satisfied'
                                else:
                                    result['overall_satisfaction'] = 'not_satisfied'
                            
                            # Update score
                            original_score = result['score']
                            result['score'] = round(new_score, 1)
                            result['satisfaction_score'] = satisfied_constraints / total_requirements
                            
                            # ✅ Store actual statistics for summary (for CSV)
                            result['_scoring_details'] = {
                                'total_constraints': total_requirements,
                                'satisfied_constraints': satisfied_constraints_for_csv,  # ✅ Correct number for CSV statistics
                                'fatal_violations': fatal_violations,
                                'total_violations': total_violations  # ✅ Add total violations
                            }
                            
                            logger.info(f"Feasibility score recalculated: {satisfied_constraints}/{total_requirements} = {new_score:.1f} points (original: {original_score})")
                            logger.info(f"Successfully evaluated Feasibility dimension, final score: {result['score']}")
                            return result
                        else:
                            logger.warning(f"Feasibility evaluation result missing score field")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Feasibility JSON parsing error: {str(e)}")
                        logger.error(f"Response content: {response[:500]}...")
                        
                        # Try to extract basic score from response
                        try:
                            import re
                            score_match = re.search(r'"score":\s*([0-9.]+)', response)
                            if score_match:
                                score = float(score_match.group(1))
                                logger.info(f"Extracted score from error response: {score}")
                                return {
                                    'score': score,
                                    'overall_satisfaction': 'partially_satisfied',
                                    'satisfaction_score': score / 10.0,
                                    'detailed_breakdown': {'strengths': [], 'weaknesses': ['JSON format error'], 'key_observations': []},
                                    'justification': 'JSON parsing failed, only extracted basic score',
                                    'violations_found': [],
                                    'constraints_respected': []
                                }
                        except Exception:
                            pass
                        
                    return None
            
            return None
                    
        except Exception as e:
            logger.error(f"Error evaluating Feasibility dimension: {str(e)}")
            return None

    def evaluate_numerical_solution_dimension(self, report_content: str, criteria: str = "",
                                            problem_context: str = "") -> Optional[Dict[str, any]]:
        """
        Specialized method for evaluating the Numerical Solution dimension
        
        Args:
            report_content: Report content
            criteria: Evaluation standard (optional, with default standard)
            problem_context: Problem context
            
        Returns:
            Evaluation result dictionary, containing numerical solution quality analysis
        """
        # If no criteria is provided, use default standard
        if not criteria.strip():
            criteria = """
Evaluate the quality of the numerical solution generated, including:
1. Reasonableness and accuracy of the numerical solution
2. Whether specific decision variable values are provided
3. Whether the objective function value is reasonable
4. Completeness of the solution (covering all important variables)
5. Numerical precision and clarity of representation
6. Practical operability of the solution
"""
        
        prompt = f"""
Please evaluate the quality of the numerical solution in the generated report.

**Problem Context:**
{problem_context}

**Report Content:**
{report_content}

**Evaluation Criteria:**
{criteria}

Please respond in the following JSON format:

{{
    "score": <score between 0-10, can use decimal points for precision>,
    "has_numerical_solution": true/false,
    "solution_quality": {{
        "has_objective_value": true/false,
        "objective_value": "extracted value or null",
        "has_variable_values": true/false,
        "variable_count": <number of variables with values>,
        "solution_completeness": "complete/partial/missing",
        "numerical_precision": "high/medium/low",
        "practical_feasibility": "feasible/questionable/infeasible"
    }},
    "detailed_breakdown": {{
        "strengths": ["List of specific strengths in numerical solution"],
        "weaknesses": ["List of weaknesses or missing elements"],
        "key_observations": ["Important observations about solution quality"]
    }},
    "solution_evidence": {{
        "objective_evidence": "specific text showing objective value",
        "variables_evidence": "specific text showing variable values",
        "validation_evidence": "evidence of solution validation"
    }},
    "justification": "Detailed explanation of the scoring and analysis",
    "improvement_suggestions": ["Specific suggestions for improving numerical solution quality"],
    "overall_assessment": "Overall assessment of numerical solution quality"
}}

**Analysis Points:**
1. Extract and verify the objective function value
2. Identify and count decision variables with specific values
3. Assess the completeness and reasonableness of the solution
4. Evaluate numerical precision and presentation clarity
5. Determine practical implementability

Please carefully analyze the report content and provide detailed numerical solution assessment.
"""

        try:
            response = self.call_llm_api(prompt)
            if response:
                # Try to extract JSON part
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                    try:
                        result = json.loads(json_str)
                        
                        # Validate result format, relax requirements
                        if 'score' in result:
                            # Ensure basic fields exist
                            if 'has_numerical_solution' not in result:
                                result['has_numerical_solution'] = True
                            if 'detailed_breakdown' not in result:
                                result['detailed_breakdown'] = {'strengths': [], 'weaknesses': [], 'key_observations': []}
                            if 'justification' not in result:
                                result['justification'] = 'Evaluation completed with partial information'
                            if 'solution_quality' not in result:
                                result['solution_quality'] = {}
                            
                            logger.info(f"Successfully evaluated Numerical Solution dimension, score: {result['score']}")
                            return result
                        else:
                            logger.warning(f"Numerical Solution evaluation result missing score field")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Numerical Solution JSON parsing error: {str(e)}")
                        logger.error(f"Response content: {response[:200]}...")
                        
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating Numerical Solution dimension: {str(e)}")
            return None

    def evaluate_dimension(self, report_content: str, dimension: str, criteria: str, 
                          problem_context: str = "") -> Optional[Dict[str, any]]:
        """
        Evaluate the specific dimension of the report
        
        Args:
            report_content: Report content
            dimension: Evaluation dimension name
            criteria: Evaluation standard
            problem_context: Problem context
            
        Returns:
            Evaluation result dictionary
        """
        if not criteria.strip():
            logger.warning(f"No criteria provided for dimension: {dimension}")
            return None
        
        prompt = f"""
You are a professional engineering modeling competition judge with extensive experience in evaluating mathematical and engineering solutions. Please conduct a rigorous evaluation of the following EngiLLM system generated report based on the provided criteria.

**Problem Context:**
{problem_context}

**Report Content to Evaluate:**
{report_content}

**Evaluation Dimension:** {dimension}

**Evaluation Criteria:**
{criteria}

Please evaluate the report strictly according to the provided criteria and provide your assessment in the following JSON format:

{{
    "score": <score between 0-10, can use decimal points for precision>,
    "detailed_breakdown": {{
        "strengths": ["List of specific strengths found in the report"],
        "weaknesses": ["List of specific weaknesses or missing elements"],
        "key_observations": ["Important observations about the report quality"]
    }},
    "justification": "Detailed explanation of why this score was assigned, referencing specific parts of the report and how they align with the scoring criteria",
    "improvement_suggestions": ["Specific suggestions for improving the report in this dimension"]
}}

**Important Notes:**
- Be objective and consistent in your evaluation
- Reference specific sections or content from the report in your justification
- Consider both the technical accuracy and the methodology
- Compare the report content against the exact scoring criteria provided
- Look for evidence of the specific capabilities being evaluated in this dimension
"""

        try:
            response = self.call_llm_api(prompt)
            if response:
                # Try to extract JSON part
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                    try:
                        result = json.loads(json_str)
                        
                        # Validate result format
                        required_fields = ['score', 'detailed_breakdown', 'justification', 'improvement_suggestions']
                        if all(field in result for field in required_fields):
                            logger.info(f"Successfully evaluated dimension: {dimension}, Score: {result['score']}")
                            return result
                        else:
                            logger.warning(f"Missing required fields in evaluation result for {dimension}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error for {dimension}: {str(e)}")
                        logger.error(f"Response content: {response[:200]}...")
                        
                return None
                    
        except Exception as e:
            logger.error(f"Error evaluating dimension {dimension}: {str(e)}")
        return None

    def evaluate_json_dimension(self, json_data: Dict[str, Any], dimension: str, 
                               problem_context: str = "") -> Optional[Dict[str, any]]:
        """
        Evaluate the specific dimension of the JSON analysis
        
        Args:
            json_data: JSON analysis data
            dimension: Evaluation dimension name
            problem_context: Problem context
            
        Returns:
            Evaluation result dictionary
        """
        
        # Custom evaluation standards for EngiLLM project
        criteria_mapping = {
            'Information_Extraction': """
Evaluate the quality of JSON analysis in information extraction:
1. Completeness: Whether all key information (parameters, constraints, objectives, etc.) has been extracted from the original problem
2. Accuracy: Whether the extracted information is accurate and free of omissions or misinterpretations
3. Structuredness: Whether the information is well organized and classified
4. Traceability: Whether the information source can be clearly traced
Scoring standard: 9-10 points (excellent), 7-8 points (good), 5-6 points (average), 3-4 points (poor), 0-2 points (very poor)
            """,
            'Domain_Specific_Reasoning': """
Evaluate the quality of JSON analysis in domain-specific reasoning:
1. Engineering domain knowledge application: Whether the relevant engineering domain knowledge has been correctly applied
2. Modeling pattern selection: Whether the appropriate mathematical modeling method has been selected
3. Constraint understanding: Whether the engineering constraints have been correctly understood and expressed
4. Physical/logical consistency: Whether the model conforms to physical laws and logical rules
Scoring standard: 9-10 points (excellent), 7-8 points (good), 5-6 points (average), 3-4 points (poor), 0-2 points (very poor)
            """,
            'Multi_Objective_Decision': """
Evaluate the quality of JSON analysis in multi-objective decision modeling:
1. Target identification: Whether all relevant targets have been correctly identified
2. Target priority: Whether the priority and weight of the target have been reasonably set
3. Trade-off analysis: Whether the trade-off relationship between different targets has been analyzed
4. Decision framework: Whether a clear multi-objective decision framework has been established
Scoring standard: 9-10 points (excellent), 7-8 points (good), 5-6 points (average), 3-4 points (poor), 0-2 points (very poor)
            """,
            'Uncertainty_Handling': """
Evaluate the quality of JSON analysis in uncertainty handling:
1. Uncertainty identification: Whether the key uncertainty sources in the problem have been identified
2. Assumption rationality: Whether the assumptions made are reasonable and based on evidence
3. Sensitivity consideration: Whether the sensitivity analysis of the key parameters has been considered
4. Robust design: Whether suggestions for enhancing model robustness have been provided
Scoring standard: 9-10 points (excellent), 7-8 points (good), 5-6 points (average), 3-4 points (poor), 0-2 points (very poor)
            """
        }
        
        criteria = criteria_mapping.get(dimension, "General evaluation standard")
        
        # Convert JSON data to readable text
        json_text = json.dumps(json_data, indent=2, ensure_ascii=False)
        
        prompt = f"""
You are a professional engineering modeling expert with extensive experience in mathematical optimization and engineering modeling. Please conduct a rigorous evaluation of the following EngiLLM system generated JSON modeling analysis.

**Problem Context:**
{problem_context}

**JSON Modeling Analysis Content:**
```json
{json_text}
```

**Evaluation Dimension:** {self.EVALUATION_DIMENSIONS.get(dimension, dimension)}

**Evaluation Criteria:**
{criteria}

Please strictly evaluate the quality of the JSON analysis according to the provided criteria and return the evaluation results in the following JSON format:

{{
    "score": <score between 0-10, can use decimal points for precision>,
    "detailed_breakdown": {{
        "strengths": ["List of specific strengths found in the JSON analysis"],
        "weaknesses": ["List of specific weaknesses or missing elements in the JSON analysis"],
        "key_observations": ["Important observations about the quality of the JSON analysis"]
    }},
    "justification": "Detailed explanation of why this score was assigned, referencing specific parts of the JSON and the evaluation criteria",
    "improvement_suggestions": ["Specific suggestions for improving the quality of the JSON analysis"]
}}

**Important Notes:**
- Be objective and consistent in your evaluation
- Reference specific parts of the JSON in your justification
- Consider the technical accuracy and methodological completeness of the JSON analysis
- Compare the JSON analysis content against the exact evaluation criteria provided
"""

        try:
            response = self.call_llm_api(prompt)
            if response:
                # Try to extract JSON part
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                    try:
                        result = json.loads(json_str)
                        
                        # Validate result format
                        required_fields = ['score', 'detailed_breakdown', 'justification', 'improvement_suggestions']
                        if all(field in result for field in required_fields):
                            logger.info(f"Successfully evaluated dimension: {dimension}, score: {result['score']}")
                            return result
                        else:
                            logger.warning(f"Evaluation result missing required fields: {dimension}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error for {dimension}: {str(e)}")
                        logger.error(f"Response content: {response[:200]}...")
                        
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating dimension {dimension}: {str(e)}")
            return None

    def evaluate_single_json(self, json_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate a single JSON analysis data
        
        Args:
            json_data: JSON analysis data
            metadata: Metadata information
            
        Returns:
            Complete evaluation result
        """
        # Safely handle metadata
        if metadata is None:
            metadata = {}
        
        logger.info(f"Start evaluating JSON analysis: {metadata.get('report_file', 'unknown')}")
        
        # Extract problem context
        problem_context = ""
        if 'modeling_context' in json_data:
            problem_context = json_data['modeling_context'].get('problem_essence', '')
        
        # Execute evaluation for each dimension
        evaluation_results = {
            'metadata': {
                'source_file': metadata.get('report_file', 'unknown') if metadata else 'unknown',
                'evaluation_timestamp': datetime.now().isoformat(),
                'json_block_index': metadata.get('block_index', 0) if metadata else 0,
                'context_info': metadata.get('context', {}) if metadata else {}
            },
            'problem_context': problem_context,
            'dimension_scores': {},
            'overall_assessment': {}
        }
        
        total_score = 0
        valid_dimensions = 0
        
        for dimension in self.EVALUATION_DIMENSIONS.keys():
            logger.info(f"Evaluating dimension: {dimension}")
            
            result = self.evaluate_json_dimension(
                json_data=json_data,
                dimension=dimension,
                problem_context=problem_context
            )
            
            if result:
                evaluation_results['dimension_scores'][dimension] = result
                total_score += result['score']
                valid_dimensions += 1
                
                logger.info(f"Dimension {dimension} evaluation completed, score: {result['score']}")
            else:
                logger.warning(f"Dimension {dimension} evaluation failed")
                evaluation_results['dimension_scores'][dimension] = {
                    'score': 0,
                    'error': 'Evaluation failed'
                }
        
        # Calculate overall evaluation
        if valid_dimensions > 0:
            average_score = total_score / valid_dimensions
            evaluation_results['overall_assessment'] = {
                'average_score': round(average_score, 2),
                'total_possible_score': valid_dimensions * 10,
                'dimensions_evaluated': valid_dimensions,
                'overall_performance': self._get_performance_level(average_score)
            }
        else:
            evaluation_results['overall_assessment'] = {
                'error': 'No valid dimensions could be evaluated'
            }
        
        logger.info(f"Evaluation completed. Average score: {evaluation_results['overall_assessment'].get('average_score', 'N/A')}")
        
        return evaluation_results

    def evaluate_json_batch(self, json_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch evaluate JSON analysis data
        
        Args:
            json_data_list: JSON analysis data list
            
        Returns:
            Evaluation result list
        """
        evaluation_results = []
        
        logger.info(f"Start batch evaluating {len(json_data_list)} JSON analyses")
        
        for i, json_item in enumerate(json_data_list):
            logger.info(f"Evaluation progress: {i+1}/{len(json_data_list)}")
            
            # Extract JSON data and metadata
            json_data = json_item.get('json_data', {})
            metadata = {
                'report_file': json_item.get('report_file', 'unknown'),
                'block_index': json_item.get('block_index', 0),
                'context': json_item.get('context', {})
            }
            
            # Evaluate single JSON
            result = self.evaluate_single_json(json_data, metadata)
            evaluation_results.append(result)
        
        logger.info(f"Batch evaluation completed, evaluated {len(evaluation_results)} JSON analyses")
        return evaluation_results

    def _get_performance_level(self, score: float) -> str:
        """Determine performance level based on score"""
        if score >= 9:
            return "Excellent"
        elif score >= 7:
            return "Good"
        elif score >= 5:
            return "Average"
        elif score >= 3:
            return "Below Average"
        else:
            return "Poor"

    def save_evaluation_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation result list
            output_file: Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to: {output_file}")
            
            # Save a simplified summary
            summary_file = output_file.replace('.json', '_summary.txt')
            self._save_summary(results, summary_file)
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise

    def _save_summary(self, results: List[Dict[str, Any]], summary_file: str):
        """Save evaluation result summary"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("EngiLLM Modeling Quality Evaluation Summary Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Overall statistics
                total_evaluations = len(results)
                f.write(f"Evaluation total: {total_evaluations}\n")
                f.write(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Detailed evaluation results
                for i, result in enumerate(results, 1):
                    metadata = result.get('metadata', {})
                    overall = result.get('overall_assessment', {})
                    
                    f.write(f"Evaluation {i}: {metadata.get('source_file', 'unknown')}\n")
                    f.write(f"  Average score: {overall.get('average_score', 'N/A')}/10\n")
                    f.write(f"  Performance level: {overall.get('overall_performance', 'N/A')}\n")
                    f.write(f"  Evaluation dimension number: {overall.get('dimensions_evaluated', 'N/A')}\n")
                    
                    # Dimension scores
                    dimension_scores = result.get('dimension_scores', {})
                    for dimension, score_data in dimension_scores.items():
                        if isinstance(score_data, dict) and 'score' in score_data:
                            f.write(f"    {dimension}: {score_data['score']}/10\n")
                    f.write("\n")
                
                f.write("=" * 50 + "\n")
                
            logger.info(f"Evaluation summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation summary: {str(e)}")

    def generate_quality_curve(self, evaluation_results: List[Dict[str, Any]], 
                              solver_success_indices: List[int] = None,
                              output_file: str = "modeling_quality_curve.png") -> None:
        """
        Generate modeling quality change curve
        
        Args:
            evaluation_results: Evaluation result list
            solver_success_indices: Solver success indices list
            output_file: Output image file path
        """
        try:
            # Extract average score
            scores = []
            file_names = []
            
            for result in evaluation_results:
                overall = result.get('overall_assessment', {})
                score = overall.get('average_score', 0)
                scores.append(score)
                
                metadata = result.get('metadata', {})
                file_name = metadata.get('source_file', 'unknown')
                file_names.append(file_name)
            
            # Create chart
            plt.figure(figsize=(12, 8))
            
            # Draw quality curve
            x_positions = range(len(scores))
            plt.plot(x_positions, scores, 'b-o', linewidth=2, markersize=8, label='Modeling quality score')
            
            # Mark Solver success positions
            if solver_success_indices:
                success_scores = [scores[i] for i in solver_success_indices if i < len(scores)]
                success_positions = [i for i in solver_success_indices if i < len(scores)]
                plt.scatter(success_positions, success_scores, color='red', s=100, 
                           marker='*', label='Solver success', zorder=5)
            
            # Set chart properties
            plt.xlabel('Test sequence', fontsize=12)
            plt.ylabel('Modeling quality score', fontsize=12)
            plt.title('EngiLLM Modeling Quality Change Curve', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Set y-axis range
            plt.ylim(0, 10)
            
            # Set x-axis labels
            if len(file_names) <= 10:  # If there are not many files, display the full file name
                plt.xticks(x_positions, [name.replace('debug_problem_', '').replace('.md', '') 
                                       for name in file_names], rotation=45)
            else:
                plt.xticks(x_positions[::max(1, len(x_positions)//10)])
            
            # Add average line
            avg_score = np.mean(scores)
            plt.axhline(y=avg_score, color='green', linestyle='--', alpha=0.7, 
                       label=f'Average quality: {avg_score:.2f}')
            plt.legend()
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Modeling quality curve saved to: {output_file}")
            
            # Generate statistics information
            stats_info = f"""
Modeling quality statistics:
- Average quality score: {avg_score:.2f}
- Highest quality score: {max(scores):.2f}
- Lowest quality score: {min(scores):.2f}
- Standard deviation: {np.std(scores):.2f}
- Solver success times: {len(solver_success_indices) if solver_success_indices else 0}
            """
            
            stats_file = output_file.replace('.png', '_stats.txt')
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(stats_info)
            
            logger.info(f"Statistics information saved to: {stats_file}")
            
        except Exception as e:
            logger.error(f"Error generating quality curve: {str(e)}")
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='EngiLLM Modeling Quality Evaluator')
    parser.add_argument('--json_file', required=True, help='JSON analysis data file path')
    parser.add_argument('--output', help='Output file path (default: auto-generated)')
    parser.add_argument('--curve', action='store_true', help='Generate quality change curve')
    
    args = parser.parse_args()
    
    # Create evaluator instance
    evaluator = EngiLLMReportEvaluator()
    
    try:
        # Load JSON analysis data
        json_data_list = evaluator.load_json_analysis_data(args.json_file)
        
        # Batch evaluation
        evaluation_results = evaluator.evaluate_json_batch(json_data_list)
        
        # Determine output file name
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'engillm_modeling_evaluation_{timestamp}.json'
        
        # Save results
        evaluator.save_evaluation_results(evaluation_results, output_file)
        
        # Generate quality curve
        if args.curve:
            curve_file = output_file.replace('.json', '_quality_curve.png')
            evaluator.generate_quality_curve(evaluation_results, output_file=curve_file)
        
        # Print statistics information
        print("\n" + "=" * 60)
        print("EngiLLM modeling quality evaluation completed")
        print("=" * 60)
        
        total_evaluations = len(evaluation_results)
        print(f"Evaluation total: {total_evaluations}")
        
        # Calculate average score
        all_scores = []
        for result in evaluation_results:
            overall = result.get('overall_assessment', {})
            score = overall.get('average_score', 0)
            if isinstance(score, (int, float)):
                all_scores.append(score)
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"Overall average quality score: {avg_score:.2f}/10")
        
        print(f"\nDetailed results saved to: {output_file}")
        if args.curve:
            print(f"Quality curve saved to: {curve_file}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0


class EngiLLMReportEvaluator(EngiLLMModelingEvaluator):
    """Extended EngiLLM report evaluator, supports reading standards from JSON"""
    
    def __init__(self, evaluation_model: str = "gpt-4.1-nano"):
        """
        Initialize report evaluator
        
        Args:
            evaluation_model: Model name for evaluation, default is gpt-4.1-nano
        """
        super().__init__(evaluation_model)
        logger.info(f"EngiLLMReportEvaluator initialized, using evaluation model: {self.evaluation_model}")
    
    def _extract_problem_index_from_filename(self, report_file_path: str) -> int:
        """Extract problem index from filename"""
        try:
            # Try to extract problem number from filename
            # For example: debug_P5_free_gpt4o_20250907_151955.md -> 4 (P5 corresponds to index 4)
            import re
            from pathlib import Path
            filename = Path(report_file_path).name
            
            # Match P followed by numbers
            match = re.search(r'[_\-]P(\d+)[_\-]', filename)
            if match:
                problem_number = int(match.group(1))
                return problem_number - 1  # Convert to 0-based index
            
            # If no P number format is found, try other formats
            match = re.search(r'problem_(\d+)', filename)
            if match:
                return int(match.group(1))
                
            # Default to return 0
            logger.warning(f"Could not extract problem index from filename: {filename}, using index 0")
            return 0
            
        except Exception as e:
            logger.error(f"Error extracting problem index from filename: {e}")
            return 0

    def evaluate_engillm_report(self, report_file_path: str, json_file: str = "simple_test_problems.json", 
                               problem_index: int = None, file_type: str = "markdown", engillm_stats: Dict = None) -> Dict[str, any]:
        """
        Evaluate EngiLLM generated report
        
        Args:
            report_file_path: Report file path
            json_file: JSON file path (default: simple_test_problems.json)
            problem_index: Problem index (if None, extract from filename automatically)
            file_type: File type ("markdown" or other)
            
        Returns:
            Complete evaluation result
        """
        logger.info(f"Start evaluating EngiLLM report: {report_file_path}")
        
        # Determine problem index
        if problem_index is None:
            problem_index = self._extract_problem_index_from_filename(report_file_path)
        
        # Load evaluation criteria
        criteria = self.load_evaluation_criteria_from_json(json_file, problem_index)
        
        # Read report content
        try:
            with open(report_file_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            logger.info(f"Successfully read report file: {report_file_path}")
            logger.info(f"File size: {len(report_content)} characters")
        except Exception as e:
            logger.error(f"Error reading report file {report_file_path}: {str(e)}")
            raise
        
        # Evaluate dimension mapping - only evaluate dimensions with content
        evaluation_dimensions = {}
        
        if criteria['information_extraction'].strip():
            evaluation_dimensions['Information Extraction'] = criteria['information_extraction']
        
        if criteria['domain_specific_reasoning'].strip():
            evaluation_dimensions['Domain-Specific Reasoning'] = criteria['domain_specific_reasoning']
            
        if criteria['multi_objective_decision'].strip():
            evaluation_dimensions['Multi-Objective Decision-Making'] = criteria['multi_objective_decision']
            
        if criteria['uncertainty_handling'].strip():
            evaluation_dimensions['Uncertainty Handling'] = criteria['uncertainty_handling']
            
        # ✅ Add feasibility and numerical_solution evaluation
        if criteria.get('feasibility', '').strip():
            evaluation_dimensions['Feasibility'] = criteria['feasibility']
            
        # Create default evaluation criteria for numerical_solution (if not provided)
        numerical_criteria = criteria.get('numerical_solution', '').strip()
        if not numerical_criteria:
            # Use default numerical solution evaluation criteria
            numerical_criteria = """
Evaluate the quality of the numerical solution generated, including:
1. Reasonableness and accuracy of the numerical solution
2. Whether specific decision variable values are provided
3. Whether the objective function value is reasonable
4. Completeness of the solution (covering all important variables)
5. Numerical precision and clarity of representation
6. Practical operability of the solution
"""
        evaluation_dimensions['Numerical Solution'] = numerical_criteria
        
        # Execute evaluation
        evaluation_results = {
            'metadata': {
                'report_file': report_file_path,
                'criteria_source': json_file,
                'problem_id': criteria['problem_id'],
                'problem_index': problem_index,
                'problem_field': criteria['field'],
                'reference': criteria['reference'],
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluation_method': 'EngiLLM_GPT4o_JSON_criteria',
                'engillm_stats': engillm_stats or {}
            },
            'problem_context': criteria['problem'],
            'dimension_scores': {},
            'overall_assessment': {}
        }
        
        total_score = 0
        valid_dimensions = 0
        
        for dimension, criterion in evaluation_dimensions.items():
            if criterion.strip():  # Only evaluate dimensions with content
                logger.info(f"Evaluating dimension: {dimension}")
                
                # ✅ Use specialized evaluation methods for Feasibility and Numerical Solution
                if dimension == 'Feasibility':
                    result = self.evaluate_feasibility_dimension(
                        report_content=report_content,
                        feasibility_requirements=criterion,
                        problem_context=criteria['problem']
                    )
                elif dimension == 'Numerical Solution':
                    result = self.evaluate_numerical_solution_dimension(
                        report_content=report_content,
                        criteria=criterion,
                        problem_context=criteria['problem']
                    )
                else:
                    # Use general evaluation methods for other dimensions
                    result = self.evaluate_dimension(
                        report_content=report_content,
                        dimension=dimension,
                        criteria=criterion,
                        problem_context=criteria['problem']
                    )
                
                if result:
                    evaluation_results['dimension_scores'][dimension] = result
                    total_score += result['score']
                    valid_dimensions += 1
                    
                    logger.info(f"Dimension {dimension} evaluation completed, score: {result['score']}")
                else:
                    logger.warning(f"Dimension {dimension} evaluation failed")
                    evaluation_results['dimension_scores'][dimension] = {
                        'score': 0,
                        'error': 'Evaluation failed'
                    }
        
        # Calculate overall evaluation
        if valid_dimensions > 0:
            average_score = total_score / valid_dimensions
            evaluation_results['overall_assessment'] = {
                'average_score': round(average_score, 2),
                'total_possible_score': valid_dimensions * 10,
                'dimensions_evaluated': valid_dimensions,
                'overall_performance': self._get_performance_level(average_score)
            }
        else:
            evaluation_results['overall_assessment'] = {
                'error': 'No valid dimensions could be evaluated'
            }
        
        logger.info(f"Evaluation completed. Average score: {evaluation_results['overall_assessment'].get('average_score', 'N/A')}")
        
        return evaluation_results

    def evaluate_engillm_report_with_criteria(self, report_file_path: str, criteria: Dict[str, str], 
                                            file_type: str = "markdown", engillm_stats: Dict = None) -> Dict[str, any]:
        """
        Evaluate EngiLLM generated report using given evaluation criteria
        
        Args:
            report_file_path: Report file path
            criteria: Evaluation criteria dictionary (read from CSV)
            file_type: File type ("markdown" or other)
            engillm_stats: EngiLLM statistics
            
        Returns:
            Complete evaluation result
        """
        logger.info(f"Start evaluating EngiLLM report (using CSV criteria): {report_file_path}")
        
        # Read report content
        try:
            with open(report_file_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            logger.info(f"Successfully read report file: {report_file_path}")
            logger.info(f"File size: {len(report_content)} characters")
        except Exception as e:
            logger.error(f"Error reading report file {report_file_path}: {str(e)}")
            raise
        
        # Evaluate dimension mapping - only evaluate dimensions with content
        evaluation_dimensions = {}
        
        if criteria['information_extraction'].strip():
            evaluation_dimensions['Information Extraction'] = criteria['information_extraction']
        
        if criteria['domain_specific_reasoning'].strip():
            evaluation_dimensions['Domain-Specific Reasoning'] = criteria['domain_specific_reasoning']
            
        if criteria['multi_objective_decision'].strip():
            evaluation_dimensions['Multi-Objective Decision-Making'] = criteria['multi_objective_decision']
            
        if criteria['uncertainty_handling'].strip():
            evaluation_dimensions['Uncertainty Handling'] = criteria['uncertainty_handling']
            
        # ✅ Add feasibility and numerical_solution evaluation
        if criteria.get('feasibility', '').strip():
            evaluation_dimensions['Feasibility'] = criteria['feasibility']
            
        # Create default evaluation criteria for numerical_solution (if not provided)
        numerical_criteria = criteria.get('numerical_solution', '').strip()
        if not numerical_criteria:
            # Use default numerical solution evaluation criteria
            numerical_criteria = """
Evaluate the quality of the numerical solution generated, including:
1. Reasonableness and accuracy of the numerical solution
2. Whether specific decision variable values are provided
3. Whether the objective function value is reasonable
4. Completeness of the solution (covering all important variables)
5. Numerical precision and clarity of representation
6. Practical operability of the solution
"""
        evaluation_dimensions['Numerical Solution'] = numerical_criteria
        
        # Execute evaluation
        evaluation_results = {
            'metadata': {
                'report_file': report_file_path,
                'criteria_source': 'CSV',
                'problem_id': criteria['problem_id'],
                'problem_field': criteria['field'],
                'reference': criteria['reference'],
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluation_method': 'EngiLLM_GPT4o_CSV_criteria',
                'engillm_stats': engillm_stats or {}
            },
            'problem_context': criteria['problem'],
            'dimension_scores': {},
            'overall_assessment': {}
        }
        
        total_score = 0
        valid_dimensions = 0
        
        for dimension, criterion in evaluation_dimensions.items():
            if criterion.strip():  # Only evaluate dimensions with content
                logger.info(f"Evaluating dimension: {dimension}")
                
                # ✅ Use specialized evaluation methods for Feasibility and Numerical Solution
                if dimension == 'Feasibility':
                    result = self.evaluate_feasibility_dimension(
                        report_content=report_content,
                        feasibility_requirements=criterion,
                        problem_context=criteria['problem']
                    )
                elif dimension == 'Numerical Solution':
                    result = self.evaluate_numerical_solution_dimension(
                        report_content=report_content,
                        criteria=criterion,
                        problem_context=criteria['problem']
                    )
                else:
                    # Use general evaluation methods for other dimensions
                    result = self.evaluate_dimension(
                        report_content=report_content,
                        dimension=dimension,
                        criteria=criterion,
                        problem_context=criteria['problem']
                    )
                
                if result:
                    evaluation_results['dimension_scores'][dimension] = result
                    total_score += result['score']
                    valid_dimensions += 1
                    
                    logger.info(f"Dimension {dimension} evaluation completed, score: {result['score']}")
                else:
                    logger.warning(f"Dimension {dimension} evaluation failed")
                    evaluation_results['dimension_scores'][dimension] = {
                        'score': 0,
                        'error': 'Evaluation failed'
                    }
        
        # Calculate overall evaluation
        if valid_dimensions > 0:
            average_score = total_score / valid_dimensions
            evaluation_results['overall_assessment'] = {
                'average_score': round(average_score, 2),
                'total_possible_score': valid_dimensions * 10,
                'dimensions_evaluated': valid_dimensions,
                'overall_performance': self._get_performance_level(average_score)
            }
        else:
            evaluation_results['overall_assessment'] = {
                'error': 'No valid dimensions could be evaluated'
            }
        
        logger.info(f"Evaluation completed. Average score: {evaluation_results['overall_assessment'].get('average_score', 'N/A')}")
        
        return evaluation_results

    def _get_performance_level(self, score: float) -> str:
        """Determine performance level based on score"""
        if score >= 9:
            return "Excellent"
        elif score >= 7:
            return "Good"
        elif score >= 5:
            return "Average"
        elif score >= 3:
            return "Below Average"
        else:
            return "Poor"

    def save_evaluation_results(self, results: Dict[str, any], output_file: str):
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results
            output_file: Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to: {output_file}")
            
            # Save a simplified summary
            summary_file = output_file.replace('.json', '_summary.txt')
            self._save_summary(results, summary_file)
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise

    def _save_summary(self, results: Dict[str, any], summary_file: str):
        """Save evaluation results summary"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("EngiLLM Report Evaluation Summary\n")
                f.write("=" * 50 + "\n\n")
                
                # Basic information
                metadata = results.get('metadata', {})
                f.write(f"Report File: {metadata.get('report_file', 'N/A')}\n")
                f.write(f"Problem ID: {metadata.get('problem_id', 'N/A')}\n")
                f.write(f"Problem Field: {metadata.get('problem_field', 'N/A')}\n")
                f.write(f"Evaluation Time: {metadata.get('evaluation_timestamp', 'N/A')}\n\n")
                
                # Overall evaluation
                overall = results.get('overall_assessment', {})
                f.write("Overall Assessment:\n")
                f.write(f"  Average Score: {overall.get('average_score', 'N/A')}/10\n")
                f.write(f"  Performance Level: {overall.get('overall_performance', 'N/A')}\n")
                f.write(f"  Dimensions Evaluated: {overall.get('dimensions_evaluated', 'N/A')}\n\n")
                
                # Dimension scores
                f.write("Dimension Scores:\n")
                dimension_scores = results.get('dimension_scores', {})
                for dimension, score_data in dimension_scores.items():
                    if isinstance(score_data, dict) and 'score' in score_data:
                        f.write(f"  {dimension}: {score_data['score']}/10\n")
                    else:
                        f.write(f"  {dimension}: Error\n")
                
                # ✅ Add Feasibility simplified analysis
                feasibility_data = dimension_scores.get('Feasibility', {})
                if isinstance(feasibility_data, dict) and 'overall_satisfaction' in feasibility_data:
                    f.write("\nFeasibility Analysis:\n")
                    
                    # Display score - using stored actual data
                    scoring_details = feasibility_data.get('_scoring_details', {})
                    if scoring_details:
                        total_req = scoring_details.get('total_constraints', 0)
                        satisfied_req = scoring_details.get('satisfied_constraints', 0)
                        satisfaction_score = feasibility_data.get('satisfaction_score', 0)
                        f.write(f"  Score: {satisfied_req}/{total_req} = {satisfaction_score:.2f}\n")
                    
                    # Only display violated constraints
                    violations_found = feasibility_data.get('violations_found', [])
                    if violations_found and isinstance(violations_found, list):
                        f.write(f"  Unsatisfied Constraints:\n")
                        for violation in violations_found:
                            if isinstance(violation, dict):
                                req_num = violation.get('requirement_number', 'N/A')
                                description = violation.get('violation_description', 'No description')
                                reasoning = violation.get('reasoning', 'No reasoning')
                                f.write(f"    - Requirement #{req_num}: {description}\n")
                                f.write(f"      Reason: {reasoning}\n")
                            else:
                                f.write(f"    - {violation}\n")
                    else:
                        f.write(f"  Unsatisfied Constraints: None\n")
                
                # ✅ Add Numerical Solution detailed analysis
                numerical_data = dimension_scores.get('Numerical Solution', {})
                if isinstance(numerical_data, dict) and 'has_numerical_solution' in numerical_data:
                    f.write("\nNumerical Solution Analysis Details:\n")
                    f.write(f"  Has Numerical Solution: {numerical_data.get('has_numerical_solution', 'N/A')}\n")
                    
                    solution_quality = numerical_data.get('solution_quality', {})
                    if solution_quality:
                        f.write(f"  Has Objective Value: {solution_quality.get('has_objective_value', 'N/A')}\n")
                        f.write(f"  Objective Value: {solution_quality.get('objective_value', 'N/A')}\n")
                        f.write(f"  Has Variable Values: {solution_quality.get('has_variable_values', 'N/A')}\n")
                        f.write(f"  Variable Count: {solution_quality.get('variable_count', 'N/A')}\n")
                        f.write(f"  Solution Completeness: {solution_quality.get('solution_completeness', 'N/A')}\n")
                        f.write(f"  Numerical Precision: {solution_quality.get('numerical_precision', 'N/A')}\n")
                        f.write(f"  Practical Feasibility: {solution_quality.get('practical_feasibility', 'N/A')}\n")
                
                f.write("\n" + "=" * 50 + "\n")
                
            logger.info(f"Evaluation summary saved to: {summary_file}")
            
            # ✅ Record to CSV
            metadata = results.get('metadata', {})
            dimension_scores = results.get('dimension_scores', {})
            engillm_stats = metadata.get('engillm_stats', {})
            self._save_to_csv(metadata.get('problem_id', 'Unknown'), dimension_scores, metadata, engillm_stats)
            
        except Exception as e:
            logger.error(f"Error saving evaluation summary: {str(e)}")

    def _save_to_csv(self, problem_id: str, dimension_scores: Dict, metadata: Dict, engillm_stats: Dict = None):
        """Save evaluation results to CSV file"""
        try:
            csv_file = "engillm_evaluation_summary_v2.csv"  # ✅ Use new CSV file name
            file_exists = os.path.exists(csv_file)
            
            # Extract key information
            timestamp = metadata.get('evaluation_timestamp', datetime.now().isoformat())
            
            # EngiLLM statistics
            if engillm_stats:
                engillm_duration = engillm_stats.get('total_duration_seconds', 0)
                engillm_input_tokens = engillm_stats.get('input_tokens', 0)
                engillm_output_tokens = engillm_stats.get('output_tokens', 0)
                engillm_total_tokens = engillm_stats.get('total_tokens', 0)
            else:
                engillm_duration = 0
                engillm_input_tokens = 0
                engillm_output_tokens = 0
                engillm_total_tokens = 0
            
            # Four dimensions' scores - if not evaluated, return empty string
            def get_dimension_score(dimension_name):
                dim_data = dimension_scores.get(dimension_name, {})
                if isinstance(dim_data, dict) and 'score' in dim_data and dim_data['score'] is not None:
                    return f"{dim_data['score'] / 10:.3f}"
                else:
                    return ""  # If no evaluation criteria, return empty string
            
            info_extraction = get_dimension_score('Information Extraction')
            domain_reasoning = get_dimension_score('Domain-Specific Reasoning')
            multi_objective = get_dimension_score('Multi-Objective Decision-Making')
            uncertainty = get_dimension_score('Uncertainty Handling')
            
            # Feasibility score (split into multiple columns)
            feasibility_data = dimension_scores.get('Feasibility', {})
            scoring_details = feasibility_data.get('_scoring_details', {}) if isinstance(feasibility_data, dict) else {}
            if scoring_details:
                feasibility_satisfied = scoring_details.get('satisfied_constraints', 0)
                feasibility_total = scoring_details.get('total_constraints', 0)
                total_violations = scoring_details.get('total_violations', 0)  # Total violations
                fatal_violations = scoring_details.get('fatal_violations', 0)  # Fatal violations
            else:
                feasibility_satisfied = 0
                feasibility_total = 0
                total_violations = 0
                fatal_violations = 0
            
            # Whether there is a numerical solution
            numerical_data = dimension_scores.get('Numerical Solution', {})
            has_numerical_solution = numerical_data.get('has_numerical_solution', False) if isinstance(numerical_data, dict) else False
            
            # Unsatisfied constraints
            violations = feasibility_data.get('violations_found', []) if isinstance(feasibility_data, dict) else []
            unsatisfied_constraints = []
            for violation in violations:
                if isinstance(violation, dict):
                    req_num = violation.get('requirement_number', 'N/A')
                    description = violation.get('violation_description', 'No description')
                    reasoning = violation.get('reasoning', 'No reasoning')
                    unsatisfied_constraints.append(f"Req#{req_num}: {description} (Reason: {reasoning})")
                else:
                    unsatisfied_constraints.append(str(violation))
            
            unsatisfied_text = "; ".join(unsatisfied_constraints) if unsatisfied_constraints else "None"
            
            # CSV row data
            row_data = {
                'Problem_ID': problem_id,
                'Timestamp': timestamp,
                'EngiLLM_Duration_Seconds': f"{engillm_duration:.2f}",
                'EngiLLM_Input_Tokens': engillm_input_tokens,
                'EngiLLM_Output_Tokens': engillm_output_tokens,
                'EngiLLM_Total_Tokens': engillm_total_tokens,
                'Information_Extraction': info_extraction,
                'Domain_Reasoning': domain_reasoning,
                'Multi_Objective': multi_objective,
                'Uncertainty_Handling': uncertainty,
                'Has_Numerical_Solution': has_numerical_solution,
                'Feasibility_Total': feasibility_total,  # ✅ Total constraints
                'Feasibility_Satisfied': feasibility_satisfied,  # ✅ Satisfied constraints
                'Feasibility_Total_Violations': total_violations,  # ✅ Total violations
                'Feasibility_Fatal_Violations': fatal_violations,  # ✅ Fatal violations
                'Unsatisfied_Constraints': unsatisfied_text
            }
            
            # Write to CSV
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                
                # If file does not exist, write header row
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(row_data)
            
            logger.info(f"Evaluation results recorded to CSV: {csv_file}")
            
        except Exception as e:
            logger.error(f"Error recording evaluation results to CSV: {str(e)}")


def main_csv_evaluation():
    """JSON evaluation main function"""
    parser = argparse.ArgumentParser(description='EngiLLM Report Evaluator with JSON Criteria')
    parser.add_argument('--report', help='Path to the EngiLLM report file')
    parser.add_argument('--json', default='simple_test_problems.json', help='Path to the evaluation criteria JSON file')
    parser.add_argument('--index', type=int, help='Problem index in JSON file (0-based, auto-detected if not provided)')
    parser.add_argument('--type', choices=['markdown', 'text'], default='markdown', 
                       help='Type of report file (default: markdown)')
    parser.add_argument('--output', help='Output file path for evaluation results (default: auto-generated)')
    parser.add_argument('--list', action='store_true', help='List available problems in JSON file')
    
    args = parser.parse_args()
    
    # Create evaluator instance
    evaluator = EngiLLMReportEvaluator()
    
    try:
        # List available problems
        if args.list:
            try:
                with open(args.json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                problems = data.get('problems', [])
                print("\n" + "=" * 60)
                print("Available Problems in JSON:")
                print("=" * 60)
                for i, problem in enumerate(problems):
                    print(f"  Index {i}: {problem.get('id', f'P{i+1}')} - {problem.get('description', '')[:80]}...")
                print("=" * 60)
                print(f"Total: {len(problems)} problems")
                return 0
            except Exception as e:
                print(f"Error loading JSON file: {e}")
                return 1
        
        # Check if report file is provided
        if not args.report:
            parser.error("--report is required when not using --list")
        
        # Execute evaluation
        results = evaluator.evaluate_engillm_report(
            report_file_path=args.report,
            json_file=args.json,
            problem_index=args.index,
            file_type=args.type
        )
        
        # Determine output file name
        if args.output:
            output_file = args.output
        else:
            report_name = os.path.splitext(os.path.basename(args.report))[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'engillm_evaluation_{args.problem}_{report_name}_{timestamp}.json'
        
        # Save results
        evaluator.save_evaluation_results(results, output_file)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EngiLLM Report Evaluation Completed")
        print("=" * 60)
        
        overall = results.get('overall_assessment', {})
        print(f"Problem ID: {args.problem}")
        print(f"Average Score: {overall.get('average_score', 'N/A')}/10")
        print(f"Performance Level: {overall.get('overall_performance', 'N/A')}")
        print(f"Dimensions Evaluated: {overall.get('dimensions_evaluated', 'N/A')}")
        
        print("\nDimension Scores:")
        dimension_scores = results.get('dimension_scores', {})
        for dimension, score_data in dimension_scores.items():
            if isinstance(score_data, dict) and 'score' in score_data:
                print(f"  {dimension}: {score_data['score']}/10")
            else:
                print(f"  {dimension}: Error")
        
        print(f"\nDetailed results saved to: {output_file}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Check if using CSV evaluation mode
    if len(sys.argv) > 1 and ('--report' in sys.argv or '--list' in sys.argv):
        exit(main_csv_evaluation())
    else:
        exit(main())

