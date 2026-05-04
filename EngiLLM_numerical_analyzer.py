#!/usr/bin/env python3
"""
EngiLLM numerical solution analyzer
Adapted from numerical_solution_analyzer, specifically for EngiLLM project md reports and output format
"""

import re
import pandas as pd
import numpy as np
import json
import time
import random
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import google.generativeai as genai
from datetime import datetime

# Import unified LLM API call function
try:
    from specialized_agents import call_llm_api
    HAS_LLM_API = True
except ImportError:
    print("Warning: specialized_agents not available. Will use fallback implementation.")
    HAS_LLM_API = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EngiLLMNumericalAnalyzer:
    """EngiLLM numerical solution analyzer - using Gemini API for intelligent analysis"""
    
    def __init__(self):
        # Gemini API keys pool - consistent with other modules
        self.GEMINI_API_KEYS = [
            # Input your own Gemini API key here
        ]
        
        # EngiLLM specific analysis dimensions - using four evaluation dimensions from CSV
        self.ANALYSIS_DIMENSIONS = {
            'information_extraction': 'Information Extraction',
            'domain_specific_reasoning': 'Domain-Specific Reasoning', 
            'multi_objective_decision': 'Multi-Objective Decision-Making',
            'uncertainty_handling': 'Uncertainty Handling'
        }

    def load_evaluation_criteria_from_json(self, problem_index: int = 0, 
                                           json_file: str = "simple_test_problems.json") -> Dict[str, str]:
        """
        Load evaluation criteria for a specific problem from a JSON file
        
        Args:
            problem_index: Problem index (0-based)
            json_file: JSON file path
            
        Returns:
            Dictionary containing evaluation criteria
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            problems = data.get('problems', [])
            
            if problem_index >= len(problems):
                logger.warning(f"Problem index {problem_index} exceeds available problems ({len(problems)}), using index 0")
                problem_index = 0
            
            problem = problems[problem_index]
            evaluation_criteria = problem.get('evaluation_criteria', {})
            
            criteria = {
                'problem_id': problem.get('id', f'P{problem_index + 1}'),
                'problem_description': problem.get('description', ''),
                'information_extraction': evaluation_criteria.get('information_extraction', ''),
                'domain_specific_reasoning': evaluation_criteria.get('domain_specific_reasoning', ''),
                'multi_objective_decision': evaluation_criteria.get('multi_objective_decision', ''),
                'uncertainty_handling': evaluation_criteria.get('uncertainty_handling', '')
            }
            
            logger.info(f"Successfully loaded evaluation criteria for problem {criteria['problem_id']}")
            return criteria
            
        except Exception as e:
            logger.error(f"Error loading evaluation criteria from JSON: {str(e)}")
            # Return empty criteria as fallback
            return {
                'problem_id': f'P{problem_index + 1}',
                'problem_description': '',
                'information_extraction': '',
                'domain_specific_reasoning': '',
                'multi_objective_decision': '',
                'uncertainty_handling': ''
            }

    def _truncate_content_if_needed(self, content: str, max_length: int = 30000) -> str:
        """Truncate content if it is too long"""
        if len(content) <= max_length:
            return content
            
        logger.warning(f"⚠️ Content too long ({len(content)} characters), truncating to {max_length} characters")
        # Keep beginning and end
        head_size = int(max_length * 0.6)
        tail_size = int(max_length * 0.3)
        truncated = (content[:head_size] + 
                    f"\n\n... [Content truncated, original length {len(content)} characters] ...\n\n" + 
                    content[-tail_size:])
        return truncated

    def call_llm_api(self, prompt: str, retry_attempts: int = 10, retry_delay: int = 30) -> Optional[str]:
        """Call GPT-4o for analysis - using unified LLM API"""
        if HAS_LLM_API:
            try:
                # Check and truncate long prompt
                prompt = self._truncate_content_if_needed(prompt, 50000)
                
                logger.info("[GPT-4o] Starting numerical solution analysis")
                response = call_llm_api(prompt, "gpt4o")
                if response:
                    logger.info("[GPT-4o] Numerical solution analysis response obtained successfully")
                    return response
                else:
                    logger.warning("[GPT-4o] Numerical solution analysis returned empty result")
                    return None
            except Exception as e:
                logger.error(f"[GPT-4o] Numerical solution analysis API call error: {str(e)}")
                return None
        else:
            logger.error("Unified LLM API is not available, cannot perform numerical solution analysis")
            return None

    def _extract_problem_index_from_filename(self, report_file_path: str) -> int:
        """Extract problem index from filename"""
        try:
            # Try to extract problem number from filename
            # For example: debug_P5_free_gpt4o_20250907_151955.md -> 4 (P5 corresponds to index 4)
            import re
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

    def _analyze_with_criteria(self, report_content: str, dimension: str, 
                              criteria: str, evaluation_context: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze a specific dimension of the report using specific evaluation criteria
        
        Args:
            report_content: Report content
            dimension: Evaluation dimension
            criteria: Evaluation criteria
            evaluation_context: Evaluation context (contains problem description, etc.)
            
        Returns:
            Analysis result dictionary
        """
        if not criteria.strip():
            logger.warning(f"No criteria provided for dimension: {dimension}")
            return {
                'dimension': dimension,
                'score': 0,
                'analysis': 'No evaluation criteria provided',
                'error': 'Missing criteria'
            }
        
        prompt = f"""
You are a professional engineering modeling expert with extensive experience in mathematical optimization and engineering modeling. Please conduct a rigorous evaluation of the following EngiLLM system generated report based on the provided specific criteria.

**Problem Context:**
{evaluation_context['problem_description']}

**Report Content to Evaluate:**
{report_content}

**Evaluation Dimension:** {self.ANALYSIS_DIMENSIONS.get(dimension, dimension)}

**Specific Evaluation Criteria:**
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
            logger.info(f"[GPT-4o] Starting {dimension} dimension analysis")
            response = self.call_llm_api(prompt)
            
            if response:
                logger.info(f"[GPT-4o] {dimension} dimension analysis response obtained successfully")
                # Try to parse JSON response
                try:
                    # Extract JSON part
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    
                    if json_start != -1 and json_end != -1:
                        json_str = response[json_start:json_end]
                        result = json.loads(json_str)
                        
                        # Add dimension information
                        result['dimension'] = dimension
                        result['criteria_used'] = criteria[:200] + "..." if len(criteria) > 200 else criteria
                        
                        logger.info(f"Successfully parsed {dimension} analysis result, score: {result.get('score', 'N/A')}")
                        return result
                    else:
                        logger.warning(f"{dimension} analysis response did not find valid JSON")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Parsing {dimension} analysis response failed: {e}")
                    
                # If JSON parsing failed, return original response
                return {
                    'dimension': dimension,
                    'score': 0,
                    'analysis': response,
                    'error': 'JSON parsing failed'
                }
            else:
                logger.error(f"{dimension} dimension analysis failed: no response")
                return {
                    'dimension': dimension,
                    'score': 0,
                    'analysis': 'API call failed',
                    'error': 'No response from API'
                }
                
        except Exception as e:
            logger.error(f"{dimension} dimension analysis error: {str(e)}")
            return {
                'dimension': dimension,
                'score': 0,
                'analysis': f'Analysis failed: {str(e)}',
                'error': str(e)
            }

    def analyze_engillm_report(self, report_file_path: str, problem_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze the numerical solution quality of the EngiLLM generated md report
        
        Args:
            report_file_path: EngiLLM report file path (debug_problem_*.md)
            problem_info: Problem information dictionary, containing original problem description, etc.
            
        Returns:
            Analysis result dictionary
        """
        try:
            # Extract problem index from file path
            problem_index = self._extract_problem_index_from_filename(report_file_path)
            
            # Load evaluation criteria for this problem from JSON file
            evaluation_criteria = self.load_evaluation_criteria_from_json(problem_index)
            
            # Read EngiLLM report content
            report_content = self._read_engillm_report(report_file_path)
            
            # Extract key information from the report
            report_structure = self._parse_engillm_report_structure(report_content)
            
            # Use CSV evaluation criteria for four dimensions analysis
            information_extraction_analysis = self._analyze_with_criteria(
                report_content, 'information_extraction', 
                evaluation_criteria['information_extraction'], evaluation_criteria
            )
            
            domain_reasoning_analysis = self._analyze_with_criteria(
                report_content, 'domain_specific_reasoning',
                evaluation_criteria['domain_specific_reasoning'], evaluation_criteria
            )
            
            multi_objective_analysis = self._analyze_with_criteria(
                report_content, 'multi_objective_decision',
                evaluation_criteria['multi_objective_decision'], evaluation_criteria
            )
            
            uncertainty_analysis = self._analyze_with_criteria(
                report_content, 'uncertainty_handling',
                evaluation_criteria['uncertainty_handling'], evaluation_criteria
            )
            
            # Generate comprehensive analysis result - using CSV evaluation dimensions
            analysis_result = {
                'metadata': {
                    'report_file': report_file_path,
                    'problem_id': evaluation_criteria['problem_id'],
                    'problem_index': problem_index,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'report_length': len(report_content),
                    'analysis_method': 'engillm_gpt4o_csv_criteria',
                    'evaluation_criteria_source': 'simple_test_problems.json',
                    'problem_info': problem_info or {}
                },
                'evaluation_criteria': evaluation_criteria,
                'report_structure': report_structure,
                'information_extraction_analysis': information_extraction_analysis,
                'domain_reasoning_analysis': domain_reasoning_analysis,
                'multi_objective_analysis': multi_objective_analysis,
                'uncertainty_analysis': uncertainty_analysis,
                'overall_assessment': self._generate_csv_based_assessment(
                    information_extraction_analysis, domain_reasoning_analysis,
                    multi_objective_analysis, uncertainty_analysis, evaluation_criteria
                )
            }
            
            logger.info(f"EngiLLM numerical solution analysis completed: {report_file_path}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing EngiLLM report: {e}")
            return {
                'error': str(e),
                'metadata': {
                    'report_file': report_file_path,
                    'problem_id': self._extract_problem_id(report_file_path),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_method': 'engillm_gemini_api'
                }
            }

    def _read_engillm_report(self, file_path: str) -> str:
        """Read EngiLLM report file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Successfully read EngiLLM report file: {file_path}, length: {len(content)} characters")
            return content
        except Exception as e:
            logger.error(f"Error reading EngiLLM report file: {e}")
            raise

    def _extract_problem_id(self, file_path: str) -> str:
        """Extract problem ID from file path"""
        file_name = Path(file_path).name
        # Match format like debug_problem_20250831_P5_free.md
        match = re.search(r'debug_problem_\d+_([^_]+)_', file_name)
        if match:
            return match.group(1)
        # Match format like debug_problem_20250829_155101_free.md
        match = re.search(r'debug_problem_(\d{8}_\d{6})_', file_name)
        if match:
            return match.group(1)
        return Path(file_path).stem

    def _generate_csv_based_assessment(self, info_extraction: Dict, domain_reasoning: Dict,
                                     multi_objective: Dict, uncertainty: Dict, 
                                     criteria: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate comprehensive assessment based on CSV evaluation criteria
        
        Args:
            info_extraction: Information extraction analysis result
            domain_reasoning: Domain reasoning analysis result
            multi_objective: Multi-objective decision analysis result
            uncertainty: Uncertainty handling analysis result
            criteria: Evaluation criteria
            
        Returns:
            Comprehensive assessment result
        """
        try:
            # Collect scores from each dimension
            scores = []
            dimension_results = {
                'Information Extraction': info_extraction,
                'Domain-Specific Reasoning': domain_reasoning,
                'Multi-Objective Decision-Making': multi_objective,
                'Uncertainty Handling': uncertainty
            }
            
            valid_dimensions = 0
            total_score = 0
            
            for dimension_name, result in dimension_results.items():
                if isinstance(result, dict) and 'score' in result:
                    score = result.get('score', 0)
                    if isinstance(score, (int, float)) and score > 0:
                        scores.append(score)
                        total_score += score
                        valid_dimensions += 1
            
            # Calculate overall assessment
            if valid_dimensions > 0:
                average_score = total_score / valid_dimensions
                
                # Determine grade
                if average_score >= 9:
                    grade = "Excellent"
                elif average_score >= 7:
                    grade = "Good"
                elif average_score >= 5:
                    grade = "Average"
                elif average_score >= 3:
                    grade = "Below Average"
                else:
                    grade = "Poor"
                
                # Generate conclusion
                conclusion = f"Based on CSV evaluation criteria, the report shows {grade.lower()} performance across {valid_dimensions} dimensions."
                
                return {
                    'overall_score': round(average_score, 2),
                    'overall_grade': grade,
                    'total_possible_score': valid_dimensions * 10,
                    'dimensions_evaluated': valid_dimensions,
                    'dimension_scores': {
                        name: result.get('score', 0) if isinstance(result, dict) else 0
                        for name, result in dimension_results.items()
                    },
                    'conclusion': conclusion,
                    'evaluation_summary': {
                        'problem_id': criteria.get('problem_id', 'Unknown'),
                        'criteria_source': 'CSV-based evaluation standards',
                        'methodology': 'GPT-4o evaluation using specific domain criteria'
                    }
                }
            else:
                return {
                    'overall_score': 0,
                    'overall_grade': 'Failed',
                    'total_possible_score': 40,
                    'dimensions_evaluated': 0,
                    'dimension_scores': {},
                    'conclusion': 'No valid dimension scores could be obtained',
                    'error': 'All dimensions failed evaluation'
                }
                
        except Exception as e:
            logger.error(f"Error generating comprehensive assessment: {e}")
            return {
                'overall_score': 0,
                'overall_grade': 'Error',
                'conclusion': f'Assessment generation failed: {str(e)}',
                'error': str(e)
            }

    def _parse_engillm_report_structure(self, report_content: str) -> Dict[str, Any]:
        """Parse EngiLLM report structure information"""
        structure = {
            'has_problem_description': False,
            'has_analysis_json': False,
            'has_model_code': False,
            'has_solution_results': False,
            'has_verification_results': False,
            'has_success_records': False,
            'sections_found': []
        }
        
        # Check each key part
        if '## Original problem description' in report_content:
            structure['has_problem_description'] = True
            structure['sections_found'].append('problem_description')
        
        if '## Problem analysis result' in report_content or '```json' in report_content:
            structure['has_analysis_json'] = True
            structure['sections_found'].append('analysis_json')
        
        if '```python' in report_content or 'pyomo' in report_content.lower():
            structure['has_model_code'] = True
            structure['sections_found'].append('model_code')
        
        if '## Solution result' in report_content or 'solution_results' in report_content:
            structure['has_solution_results'] = True
            structure['sections_found'].append('solution_results')
        
        if '## Solution verification' in report_content or 'verification_results' in report_content:
            structure['has_verification_results'] = True
            structure['sections_found'].append('verification_results')
        
        if '## Historical successful solving records' in report_content or 'success_records' in report_content:
            structure['has_success_records'] = True
            structure['sections_found'].append('success_records')
        
        # Extract numerical information
        structure['numerical_values_count'] = len(re.findall(r'\d+\.?\d*', report_content))
        structure['equations_count'] = len(re.findall(r'=', report_content))
        structure['code_blocks_count'] = len(re.findall(r'```', report_content)) // 2
        
        return structure

    def _analyze_numerical_solution_quality(self, report_content: str, structure: Dict, problem_info: Dict) -> Dict[str, Any]:
        """Analyze numerical solution quality"""
        
        # Limit report length to avoid API limit
        max_length = 50000
        if len(report_content) > max_length:
            # Prioritize keeping solution-related sections
            sections_to_keep = ['Final solution result', 'Solution result', 'Success package', 'Objective function value', 'objective_value']
            report_sample = self._extract_relevant_sections(report_content, sections_to_keep, max_length)
        else:
            report_sample = report_content
        
        prompt = f"""
You are a professional engineering optimization expert, please analyze the quality of the numerical solution in the following EngiLLM system generated solution report.

Report content:
{report_sample}

Report structure information:
- Contains problem description: {structure['has_problem_description']}
- Contains analysis JSON: {structure['has_analysis_json']}
- Contains model code: {structure['has_model_code']}
- Contains solution results: {structure['has_solution_results']}
- Contains verification results: {structure['has_verification_results']}
- Contains success records: {structure['has_success_records']}

Please analyze the numerical solution quality in the following JSON format:
{{
    "has_numerical_solution": true/false,
    "solution_quality_score": 0.0-10.0,
    "confidence_score": 0.0-1.0,
    "numerical_evidence": [
        {{"type": "objective_value", "value": "specific value", "context": "context"}},
        {{"type": "decision_variable_value", "value": "specific value", "context": "context"}},
        {{"type": "constraint_satisfaction", "value": "specific value", "context": "context"}}
    ],
    "solution_characteristics": {{
        "solution_type": "optimal/feasible/approximate/no solution",
        "solver_status": "success/failure/partial success",
        "termination_condition": "optimality/feasibility/time limit/other",
        "objective_value_found": true/false,
        "decision_variables_found": true/false
    }},
    "quality_assessment": {{
        "completeness": "complete/partial complete/incomplete",
        "accuracy": "high/medium/low/cannot determine",
        "reasonableness": "reasonable/partial reasonable/unreasonable/cannot determine",
        "consistency": "consistent/partial consistent/inconsistent/cannot determine"
    }},
    "detailed_analysis": "Detailed analysis of the numerical solution quality in the report, including specific numerical results, solving status, variable values, etc.",
    "improvement_suggestions": ["Improvement suggestion 1", "Improvement suggestion 2"]
}}

Evaluation criteria:
1. Check if there are specific numerical results (objective function value, variable value, etc.)
2. Evaluate the execution status and termination condition of the solver
3. Verify the reasonableness and consistency of the numerical results
4. Analyze the completeness and accuracy of the solution
5. Consider the multi-round debugging process of the EngiLLM system

**Important Notes:**
- If the solver status is 'optimal' or the termination condition is 'optimal', even if the objective function value shows 'None' or 'cannot get', it should be considered that there is a numerical solution
- Find specific numerical values in the report, such as "Optimal total cost is X", "Target value: X", specific numerical values of decision variables, etc.
- If there are success packages (success package 1, success package 2, etc.) and contain solving results, it should be considered that there is a numerical solution

Please carefully analyze and provide professional evaluation.
"""
        
        try:
            response = self.call_llm_api(prompt)
            if response:
                return self._parse_gemini_json_response(response, 'numerical_solution_quality')
            else:
                return self._fallback_numerical_analysis(report_content, structure)
        except Exception as e:
            logger.error(f"Error analyzing numerical solution quality: {e}")
            return self._fallback_numerical_analysis(report_content, structure)

    def _analyze_solution_feasibility(self, report_content: str, structure: Dict, problem_info: Dict) -> Dict[str, Any]:
        """Analyze the feasibility of the solution"""
        
        max_length = 45000
        if len(report_content) > max_length:
            sections_to_keep = ['constraint', 'feasible', 'verification']
            report_sample = self._extract_relevant_sections(report_content, sections_to_keep, max_length)
        else:
            report_sample = report_content
        
        prompt = f"""
You are a professional engineering optimization expert, please analyze the feasibility of the solution in the following EngiLLM system generated solution report.

Report content:
{report_sample}

Please analyze the feasibility in the following JSON format:
{{
    "feasibility_status": "feasible/infeasible/partial feasible/cannot determine",
    "feasibility_score": 0.0-10.0,
    "constraint_satisfaction": {{
        "all_constraints_checked": true/false,
        "satisfied_constraints_count": number,
        "violated_constraints_count": number,
        "constraint_details": [
            {{"constraint_type": "constraint type", "status": "satisfied/violated/not checked", "evidence": "evidence"}}
        ]
    }},
    "feasibility_evidence": [
        "evidence 1: constraint satisfaction",
        "evidence 2: feasibility verification process",
        "evidence 3: boundary condition check"
    ],
    "validation_methods": [
        "validation method 1",
        "validation method 2"
    ],
    "feasibility_issues": [
        "problem 1: specific feasibility problem",
        "problem 2: constraint violation"
    ],
    "detailed_analysis": "Detailed analysis of the feasibility of the solution, including constraint satisfaction, boundary conditions, physical reasonableness, etc.",
    "recommendations": ["Improvement suggestion 1", "Improvement suggestion 2"]
}}

分析要点：
1. Check if all constraints are satisfied
2. Verify if the solution is in the feasible domain
3. Analyze the physical and engineering reasonableness
4. Evaluate the boundary condition handling
5. Check the numerical stability

Please give professional evaluation of the feasibility.
"""
        
        try:
            response = self.call_llm_api(prompt)
            if response:
                return self._parse_gemini_json_response(response, 'feasibility_analysis')
            else:
                return self._fallback_feasibility_analysis(report_content, structure)
        except Exception as e:
            logger.error(f"Error analyzing feasibility: {e}")
            return self._fallback_feasibility_analysis(report_content, structure)

    def _analyze_solution_completeness(self, report_content: str, structure: Dict, problem_info: Dict) -> Dict[str, Any]:
        """Analyze the completeness of the solution"""
        
        max_length = 40000
        if len(report_content) > max_length:
            report_sample = report_content[:max_length//2] + "\n\n... [TRUNCATED] ...\n\n" + report_content[-max_length//2:]
        else:
            report_sample = report_content
        
        prompt = f"""
You are a professional engineering optimization expert, please analyze the completeness of the solution in the following EngiLLM system generated solution report.

Report content:
{report_sample}

Please analyze the completeness in the following JSON format:
{{
    "completeness_score": 0.0-10.0,
    "required_elements": {{
        "problem_formulation": {{"present": true/false, "quality": "high/medium/low"}},
        "mathematical_model": {{"present": true/false, "quality": "high/medium/low"}},
        "solution_method": {{"present": true/false, "quality": "high/medium/low"}},
        "numerical_results": {{"present": true/false, "quality": "high/medium/low"}},
        "result_interpretation": {{"present": true/false, "quality": "high/medium/low"}},
        "validation_verification": {{"present": true/false, "quality": "high/medium/low"}}
    }},
    "missing_elements": [
        "missing element 1",
        "missing element 2"
    ],
    "strengths": [
        "strength 1: specific description",
        "strength 2: specific description"
    ],
    "completeness_assessment": "complete/basic complete/partial complete/incomplete",
    "detailed_analysis": "Detailed analysis of the completeness of the report, including the existence and quality of each necessary component",
    "improvement_suggestions": ["Improvement suggestion 1", "Improvement suggestion 2"]
}}

Evaluation criteria:
1. Check if the problem description is clear and complete
2. Check if the mathematical model is correctly established
3. Check if the solution method is appropriate
4. Check if the numerical results are sufficient
5. Check if the result interpretation is complete
6. Check if the validation process is complete

Please give a comprehensive evaluation of the completeness.
"""
        
        try:
            response = self.call_llm_api(prompt)
            if response:
                return self._parse_gemini_json_response(response, 'completeness_analysis')
            else:
                return self._fallback_completeness_analysis(report_content, structure)
        except Exception as e:
            logger.error(f"Error analyzing completeness: {e}")
            return self._fallback_completeness_analysis(report_content, structure)

    def _analyze_result_validation(self, report_content: str, structure: Dict, problem_info: Dict) -> Dict[str, Any]:
        """Analyze the quality of the result validation"""
        
        max_length = 40000
        if len(report_content) > max_length:
            sections_to_keep = ['validation', 'check', 'reasonableness']
            report_sample = self._extract_relevant_sections(report_content, sections_to_keep, max_length)
        else:
            report_sample = report_content
        
        prompt = f"""
Please analyze the quality of the result validation in the following EngiLLM system generated solution report.

Report content:
{report_sample}

Please analyze the validation quality in the following JSON format:
{{
    "validation_score": 0.0-10.0,
    "validation_methods_used": [
        "validation method 1",
        "validation method 2"
    ],
    "validation_coverage": {{
        "solution_accuracy": {{"verified": true/false, "method": "validation method"}},
        "constraint_satisfaction": {{"verified": true/false, "method": "validation method"}},
        "result_reasonableness": {{"verified": true/false, "method": "validation method"}},
        "sensitivity_analysis": {{"verified": true/false, "method": "validation method"}},
        "boundary_conditions": {{"verified": true/false, "method": "validation method"}}
    }},
    "validation_results": [
        {{"aspect": "validation aspect", "result": "passed/failed/partially passed", "details": "specific result"}}
    ],
    "validation_quality": "excellent/good/average/poor",
    "validation_gaps": [
        "validation gap 1",
        "validation gap 2"
    ],
    "detailed_analysis": "Detailed analysis of the quality of the result validation, including the appropriateness of the validation methods, the coverage of the validation, and the validation results.",
    "recommendations": ["validation improvement suggestion 1", "validation improvement suggestion 2"]
}}

Evaluation criteria:
1. Check if the validation methods are scientific and reasonable
2. Check if the validation coverage is comprehensive
3. Check if the validation process is rigorous
4. Check if the validation results are credible
5. Check if there is sensitivity analysis
6. Check if the boundary conditions are checked

Please give professional evaluation of the validation quality.
"""
        
        try:
            response = self.call_llm_api(prompt)
            if response:
                return self._parse_gemini_json_response(response, 'validation_analysis')
            else:
                return self._fallback_validation_analysis(report_content, structure)
        except Exception as e:
            logger.error(f"Error analyzing validation quality: {e}")
            return self._fallback_validation_analysis(report_content, structure)

    def _extract_relevant_sections(self, content: str, keywords: List[str], max_length: int) -> str:
        """Extract relevant sections content"""
        lines = content.split('\n')
        relevant_lines = []
        current_length = 0
        
        for line in lines:
            if current_length >= max_length:
                break
            
            # Check if it contains keywords
            if any(keyword.lower() in line.lower() for keyword in keywords):
                # Add context
                line_idx = lines.index(line)
                start_idx = max(0, line_idx - 5)
                end_idx = min(len(lines), line_idx + 6)
                
                context_lines = lines[start_idx:end_idx]
                for context_line in context_lines:
                    if context_line not in relevant_lines and current_length < max_length:
                        relevant_lines.append(context_line)
                        current_length += len(context_line)
        
        if not relevant_lines:
            # If no relevant content is found, return the beginning and end
            return content[:max_length//2] + "\n\n... [CONTENT TRUNCATED] ...\n\n" + content[-max_length//2:]
        
        return '\n'.join(relevant_lines)

    def _parse_gemini_json_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse Gemini's JSON response"""
        try:
            # Clean the response text
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            result = json.loads(cleaned_response)
            result['analysis_method'] = 'gemini_api'
            logger.info(f"Successfully parsed Gemini {analysis_type} analysis result")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini {analysis_type} response: {e}")
            return self._create_fallback_result(analysis_type, f"JSON parsing failed: {response[:200]}...")

    def _create_fallback_result(self, analysis_type: str, message: str) -> Dict[str, Any]:
        """Create fallback analysis result"""
        fallback_results = {
            'numerical_solution_quality': {
                'has_numerical_solution': False,
                'solution_quality_score': 0.0,
                'confidence_score': 0.1,
                'detailed_analysis': f'Fallback analysis method: {message}',
                'analysis_method': 'fallback'
            },
            'feasibility_analysis': {
                'feasibility_status': 'cannot determine',
                'feasibility_score': 0.0,
                'detailed_analysis': f'Fallback analysis method: {message}',
                'analysis_method': 'fallback'
            },
            'completeness_analysis': {
                'completeness_score': 0.0,
                'completeness_assessment': 'cannot determine',
                'detailed_analysis': f'Fallback analysis method: {message}',
                'analysis_method': 'fallback'
            },
            'validation_analysis': {
                'validation_score': 0.0,
                'validation_quality': 'cannot determine',
                'detailed_analysis': f'Fallback analysis method: {message}',
                'analysis_method': 'fallback'
            }
        }
        return fallback_results.get(analysis_type, {'error': message, 'analysis_method': 'fallback'})

    def _fallback_numerical_analysis(self, report_content: str, structure: Dict) -> Dict[str, Any]:
        """Fallback numerical solution analysis"""
        has_numbers = structure['numerical_values_count'] > 20
        has_results = structure['has_solution_results']
        
        return {
            'has_numerical_solution': has_numbers and has_results,
            'solution_quality_score': 3.0 if has_numbers and has_results else 1.0,
            'confidence_score': 0.3,
            'detailed_analysis': 'Fallback analysis method: based on structure analysis',
            'analysis_method': 'fallback'
        }

    def _fallback_feasibility_analysis(self, report_content: str, structure: Dict) -> Dict[str, Any]:
        """Fallback feasibility analysis"""
        return {
            'feasibility_status': 'cannot determine',
            'feasibility_score': 0.0,
            'detailed_analysis': 'Fallback analysis method: API call failed',
            'analysis_method': 'fallback'
        }

    def _fallback_completeness_analysis(self, report_content: str, structure: Dict) -> Dict[str, Any]:
        """Fallback completeness analysis"""
        score = len(structure['sections_found']) * 1.5  # Based on the number of found sections
        return {
            'completeness_score': min(score, 10.0),
            'completeness_assessment': 'based on structure analysis',
            'detailed_analysis': f"Fallback analysis method: found {len(structure['sections_found'])} key sections",
            'analysis_method': 'fallback'
        }

    def _fallback_validation_analysis(self, report_content: str, structure: Dict) -> Dict[str, Any]:
        """Fallback validation analysis"""
        return {
            'validation_score': 0.0,
            'validation_quality': 'cannot determine',
            'detailed_analysis': 'Fallback analysis method: API call failed',
            'analysis_method': 'fallback'
        }

    def _generate_overall_assessment(self, numerical_analysis: Dict, feasibility_analysis: Dict, 
                                   completeness_analysis: Dict, validation_analysis: Dict) -> Dict[str, Any]:
        """Generate overall assessment"""
        
        # Calculate the scores of each dimension
        scores = {
            'numerical_quality': numerical_analysis.get('solution_quality_score', 0.0),
            'feasibility': feasibility_analysis.get('feasibility_score', 0.0),
            'completeness': completeness_analysis.get('completeness_score', 0.0),
            'validation': validation_analysis.get('validation_score', 0.0)
        }
        
        # Calculate the weighted average score
        weights = {'numerical_quality': 0.35, 'feasibility': 0.25, 'completeness': 0.25, 'validation': 0.15}
        overall_score = sum(scores[dim] * weights[dim] for dim in scores)
        
        # Determine the overall grade
        if overall_score >= 8.0:
            overall_grade = "excellent"
        elif overall_score >= 6.0:
            overall_grade = "good"
        elif overall_score >= 4.0:
            overall_grade = "average"
        elif overall_score >= 2.0:
            overall_grade = "poor"
        else:
            overall_grade = "very poor"
        
        # Generate conclusion
        has_solution = numerical_analysis.get('has_numerical_solution', False)
        feasibility_status = feasibility_analysis.get('feasibility_status', 'cannot determine')
        completeness_level = completeness_analysis.get('completeness_assessment', 'cannot determine')
        
        conclusion = f"Numerical solution quality: {overall_grade} ({overall_score:.2f}/10); "
        conclusion += f"Contains numerical solution: {'yes' if has_solution else 'no'}; "
        conclusion += f"Feasibility: {feasibility_status}; "
        conclusion += f"Completeness: {completeness_level}"
        
        return {
            'overall_score': round(overall_score, 2),
            'overall_grade': overall_grade,
            'dimension_scores': scores,
            'dimension_weights': weights,
            'conclusion': conclusion,
            'summary': {
                'has_numerical_solution': has_solution,
                'solution_confidence': numerical_analysis.get('confidence_score', 0.0),
                'feasibility_status': feasibility_status,
                'completeness_level': completeness_level,
                'validation_quality': validation_analysis.get('validation_quality', 'cannot determine')
            }
        }

    def save_analysis_results(self, results: Dict[str, Any], output_file: str):
        """Save analysis results to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"EngiLLM numerical solution analysis results saved to: {output_file}")
            
            # Save simplified summary
            summary_file = output_file.replace('.json', '_summary.txt')
            self._save_summary(results, summary_file)
            
        except Exception as e:
            logger.error(f"Save analysis results failed: {e}")

    def _save_summary(self, results: Dict[str, Any], summary_file: str):
        """Save analysis results summary"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("EngiLLM numerical solution analysis summary report\n")
                f.write("=" * 50 + "\n\n")
                
                # Basic information
                metadata = results.get('metadata', {})
                f.write(f"Report file: {metadata.get('report_file', 'N/A')}\n")
                f.write(f"Problem ID: {metadata.get('problem_id', 'N/A')}\n")
                f.write(f"Analysis time: {metadata.get('analysis_timestamp', 'N/A')}\n\n")
                
                # Comprehensive assessment
                overall = results.get('overall_assessment', {})
                f.write("Comprehensive assessment:\n")
                f.write(f"  Overall score: {overall.get('overall_score', 'N/A')}/10\n")
                f.write(f"  Overall grade: {overall.get('overall_grade', 'N/A')}\n")
                f.write(f"  Conclusion: {overall.get('conclusion', 'N/A')}\n\n")
                
                # Dimension scores
                scores = overall.get('dimension_scores', {})
                f.write("Dimension scores:\n")
                for dim, score in scores.items():
                    f.write(f"  {dim}: {score}/10\n")
                
                f.write("\n" + "=" * 50 + "\n")
                
            logger.info(f"Analysis summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Save analysis summary failed: {e}")

    def analyze_json_data(self, json_tracking_file: str, output_file: str = None) -> Dict[str, Any]:
        """Analyze JSON tracking data, extract numerical solution and feasibility information"""
        try:
            # Load JSON tracking data
            with open(json_tracking_file, 'r', encoding='utf-8') as f:
                tracking_data = json.load(f)
            
            # Extract successful solving records
            json_records = tracking_data.get('json_records', [])
            solver_success_indices = tracking_data.get('solver_success_indices', [])
            
            logger.info(f"Load JSON tracking data: {len(json_records)} records, {len(solver_success_indices)} successful solving")
            
            results = {
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'tracking_file': json_tracking_file,
                    'total_records': len(json_records),
                    'solver_success_count': len(solver_success_indices)
                },
                'numerical_analysis': {},
                'feasibility_analysis': {},
                'summary': {}
            }
            
            # Analyze successful solving records
            if solver_success_indices:
                latest_success_idx = solver_success_indices[-1]  # Latest successful record
                if latest_success_idx < len(json_records):
                    success_record = json_records[latest_success_idx]
                    
                    # Extract model code for numerical solution analysis
                    model_code = success_record.get('model_code', '')
                    if model_code:
                        results['numerical_analysis'] = self._analyze_numerical_solution_from_code(model_code)
                    
                    # If there is problem analysis, do feasibility analysis
                    problem_analysis = success_record.get('problem_analysis', {})
                    if problem_analysis:
                        results['feasibility_analysis'] = self._analyze_feasibility_from_analysis(problem_analysis)
            
            # Generate summary
            results['summary'] = self._generate_json_analysis_summary(
                results['numerical_analysis'], 
                results['feasibility_analysis'],
                tracking_data
            )
            
            # Save results
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"JSON analysis results saved to: {output_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"JSON tracking data analysis failed: {e}")
            return {
                'error': str(e),
                'metadata': {'analysis_timestamp': datetime.now().isoformat()}
            }

    def _analyze_numerical_solution_from_code(self, model_code: str) -> Dict[str, Any]:
        """Analyze numerical solution from model code"""
        
        # Check numerical results in the code
        numerical_patterns = [
            r'(?:optimal|result|solution|value)\s*[=:]\s*([0-9]+\.?[0-9]*)',
            r'([0-9]+\.?[0-9]*)\s*(?:optimal|result|solution)',
            r'x\w*\s*=\s*([0-9]+\.?[0-9]*)',
            r'obj\w*\s*=\s*([0-9]+\.?[0-9]*)'
        ]
        
        numerical_values = []
        for pattern in numerical_patterns:
            matches = re.findall(pattern, model_code, re.IGNORECASE)
            numerical_values.extend(matches)
        
        has_solution = len(numerical_values) > 0
        confidence = min(0.8, len(numerical_values) * 0.2) if has_solution else 0.1
        
        return {
            'has_numerical_solution': has_solution,
            'confidence_score': confidence,
            'numerical_values_found': [
                {'value': val, 'context': 'extracted from code'} 
                for val in numerical_values[:5]  # Limit the first 5
            ],
            'solution_type': 'numerical solution' if has_solution else 'theoretical modeling',
            'analysis_method': 'code_extraction'
        }

    def _analyze_feasibility_from_analysis(self, problem_analysis: Dict) -> Dict[str, Any]:
        """Extract feasibility information from problem analysis"""
        
        # Check constraints
        constraints = problem_analysis.get('core_model_elements', {}).get('constraints', [])
        has_constraints = len(constraints) > 0
        
        # Check assumptions
        assumptions = problem_analysis.get('extended_analysis_and_robustness', {}).get('key_assumptions', [])
        
        return {
            'has_requirements': has_constraints,
            'constraint_count': len(constraints),
            'assumption_count': len(assumptions),
            'overall_satisfaction': 'partially satisfied' if has_constraints else 'cannot determine',
            'satisfaction_score': 0.7 if has_constraints else 0.3,
            'analysis_method': 'json_extraction'
        }

    def _generate_json_analysis_summary(self, numerical_analysis: Dict, feasibility_analysis: Dict, tracking_data: Dict) -> Dict[str, Any]:
        """Generate JSON analysis summary"""
        
        summary = {
            'has_numerical_solution': numerical_analysis.get('has_numerical_solution', False),
            'solution_confidence': numerical_analysis.get('confidence_score', 0.0),
            'solution_type': numerical_analysis.get('solution_type', 'unknown'),
            'numerical_values_count': len(numerical_analysis.get('numerical_values_found', [])),
            'total_iterations': tracking_data.get('total_records', 0),
            'solver_success_rate': tracking_data.get('solver_success_count', 0) / max(1, tracking_data.get('total_records', 1))
        }
        
        if feasibility_analysis.get('has_requirements', False):
            summary['feasibility_summary'] = {
                'has_feasibility_requirements': True,
                'overall_satisfaction': feasibility_analysis.get('overall_satisfaction', 'cannot determine'),
                'satisfaction_score': feasibility_analysis.get('satisfaction_score', 0.0),
                'constraint_count': feasibility_analysis.get('constraint_count', 0)
            }
        else:
            summary['feasibility_summary'] = {
                'has_feasibility_requirements': False
            }
        
        # Generate conclusion
        if numerical_analysis.get('has_numerical_solution', False):
            confidence = numerical_analysis.get('confidence_score', 0.0)
            if confidence > 0.8:
                solution_assessment = "contains clear numerical solution"
            elif confidence > 0.5:
                solution_assessment = "contains partial numerical solution"
            else:
                solution_assessment = "may contain numerical solution"
        else:
            solution_assessment = "no clear numerical solution found"
        
        feasibility_satisfaction = feasibility_analysis.get('overall_satisfaction', 'cannot determine')
        
        summary['conclusion'] = f"{solution_assessment}; Feasibility satisfaction: {feasibility_satisfaction}; Solver success rate: {summary['solver_success_rate']:.1%}"
        
        return summary


def main():
    """Test function"""
    analyzer = EngiLLMNumericalAnalyzer()
    
    # Test existing reports (find in new directory structure)
    debug_reports_dir = Path("experiments/outputs/debug_reports")
    test_reports = []
    
    if debug_reports_dir.exists():
        test_reports = list(debug_reports_dir.glob("debug_*.md"))
    
    # If no reports found, try other locations
    if not test_reports:
        # Find markdown files in current directory
        test_reports = list(Path(".").glob("debug_*.md"))
    
    # Also find json analysis files
    json_files = list(Path(".").glob("json_analysis_*.md"))
    test_reports.extend(json_files)
    
    if test_reports:
        print(f"🔍 Found {len(test_reports)} report files for analysis")
        for report_path in test_reports[:3]:  # Limit analysis to the first 3
            print(f"\n📊 Analyzing report: {report_path}")
            
            try:
                results = analyzer.analyze_engillm_report(str(report_path))
                
                if 'error' not in results:
                    overall = results.get('overall_assessment', {})
                    print(f"  ✅ Analysis completed")
                    print(f"  Overall score: {overall.get('overall_score', 'N/A')}/10")
                    print(f"  Overall grade: {overall.get('overall_grade', 'N/A')}")
                    print(f"  Conclusion: {overall.get('conclusion', 'N/A')}")
                else:
                    print(f"  ❌ Analysis failed: {results.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"  ❌ Error analyzing report: {e}")
    else:
        print("❌ No report files found for analysis")
        print("💡 Please ensure there are debug report files or run EngiLLM to generate reports")


if __name__ == "__main__":
    main()
