import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
# import google.generativeai as genai  # No longer needed, using unified LLM API
from base_agents import Message, MessageType
# Import LLM API call function
from specialized_agents import call_llm_api

# Debug record data structure
@dataclass
class DebugAttempt:
    timestamp: datetime
    agent: str
    error_type: str
    strategy: str
    success: bool
    duration: float
    coordinator_mode: str
    attempt_number: int
    error_details: str
    fix_applied: str

class DebugSummary:
    """Debug record and analysis system"""
    
    def __init__(self, problem_id: str = None, model_type: str = "gemini"):
        self.debug_records: List[DebugAttempt] = []
        self.session_stats = {}
        self.model_type = model_type  # Save model type
        
        # Problem related information
        self.problem_id = problem_id or f"problem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.problem_description = ""
        self.analysis_result = {}
        self.final_model_code = ""
        self.solution_result = {}
        self.verification_result = {}
        self.coordinator_mode = ""
        self.start_time = datetime.now()
        self.end_time = None
        # New: Cumulative success packages (maintained and synchronized by Coordinator)
        self.solution_history = []
        
        # File naming - Initially only contains problem_id, coordinator_mode updated in set_problem_info
        self.summary_file = f"debug_{self.problem_id}.md"
        self.logger = logging.getLogger(__name__)
    
    def record_attempt(self, attempt_info: Dict[str, Any]) -> None:
        """Record debugging attempt"""
        record = DebugAttempt(
            timestamp=datetime.now(),
            agent=attempt_info.get("agent", "unknown"),
            error_type=attempt_info.get("error_type", "unknown"),
            strategy=attempt_info.get("strategy", "unknown"),
            success=attempt_info.get("success", False),
            duration=attempt_info.get("duration", 0.0),
            coordinator_mode=attempt_info.get("coordinator_mode", "unknown"),
            attempt_number=attempt_info.get("attempt_number", 0),
            error_details=attempt_info.get("error_details", ""),
            fix_applied=attempt_info.get("fix_applied", "")
        )
        self.debug_records.append(record)
        self.logger.info(f"Record debugging attempt: {record.agent} - {record.error_type} - {'success' if record.success else 'failure'}")
    
    def get_agent_success_rate(self, agent: str, recent_only: bool = True) -> float:
        """Get success rate for specific Agent"""
        if recent_only:
            # Only consider recent 20 attempts
            recent_records = self.debug_records[-20:] if len(self.debug_records) > 20 else self.debug_records
        else:
            recent_records = self.debug_records
        
        agent_records = [r for r in recent_records if r.agent == agent]
        if not agent_records:
            return 0.5  # Default success rate
        
        success_count = sum(1 for r in agent_records if r.success)
        return success_count / len(agent_records)
    
    def set_problem_info(self, problem_description: str, coordinator_mode: str, model_name: str = "gemini") -> None:
        """Set basic problem information"""
        self.problem_description = problem_description
        self.coordinator_mode = coordinator_mode
        self.model_name = model_name
        
        # Update filename to include coordinator mode and model name, preventing test results from different modes and models from overwriting each other
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_file = f"debug_{self.problem_id}_{coordinator_mode}_{model_name}_{timestamp}.md"
    
    def set_analysis_result(self, analysis_result: Dict[str, Any]) -> None:
        """Set problem analysis result"""
        self.analysis_result = analysis_result
    
    def set_final_model(self, model_code: str) -> None:
        """Set final model code"""
        self.final_model_code = model_code
    
    def set_solution_result(self, solution_result: Dict[str, Any]) -> None:
        """Set solution result"""
        self.solution_result = solution_result
        self.end_time = datetime.now()
    
    def set_verification_result(self, verification_result: Dict[str, Any]) -> None:
        """Set solution verification result"""
        self.verification_result = verification_result
        self.logger.info("Solution verification result saved to debug summary")
    
    def get_total_duration(self) -> float:
        """Get total duration (from start to current time or workflow end time)"""
        # Use current time instead of solution_result's end_time to ensure complete process time is included
        end_time = datetime.now()
        return (end_time - self.start_time).total_seconds()
    
    def get_error_type_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get error type statistics"""
        error_stats = {}
        for record in self.debug_records:
            if record.error_type not in error_stats:
                error_stats[record.error_type] = {
                    "total": 0,
                    "success": 0,
                    "agents_used": set(),
                    "avg_duration": 0.0
                }
            
            stats = error_stats[record.error_type]
            stats["total"] += 1
            if record.success:
                stats["success"] += 1
            stats["agents_used"].add(record.agent)
            stats["avg_duration"] = (stats["avg_duration"] * (stats["total"] - 1) + record.duration) / stats["total"]
        
        # set to list for JSON serialization
        for error_type in error_stats:
            error_stats[error_type]["agents_used"] = list(error_stats[error_type]["agents_used"])
            error_stats[error_type]["success_rate"] = error_stats[error_type]["success"] / error_stats[error_type]["total"]
        
        return error_stats
    
    def generate_markdown_summary(self) -> str:
        """Generate complete Markdown format modeling report"""
        
        # Basic information
        total_duration = self.get_total_duration()
        total_attempts = len(self.debug_records)
        successful_attempts = sum(1 for r in self.debug_records if r.success)
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0
        
        # Generate complete report
        summary = f"""# Optimization modeling problem solving report

## Problem basic information
- **Problem ID**: {self.problem_id}
- **Coordinator mode**: {self.coordinator_mode}
- **Start time**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **End time**: {self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else 'In progress'}
- **Total duration**: {total_duration:.2f} seconds
- **Final status**: {'Success' if self.solution_result else 'Failure or in progress'}

## Original problem description
```
{self.problem_description}
```

## Problem analysis result

### Complete analysis JSON


```json
{self._format_analysis_json()}
```

## Final Pyomo model code
```python
{self.final_model_code if self.final_model_code else 'No final model code generated'}
```

## Solution result"""
        
        # Solution result - Prioritize displaying the complete real output
        if self.solution_result:
            if isinstance(self.solution_result, dict):
                summary += f"""
- **Solving status**: {self.solution_result.get('status', 'Unknown')}
- **Objective function value**: {self.solution_result.get('objective_value', 'N/A')}
- **Solver used**: {self.solution_result.get('solver_used', 'N/A')}
- **Solving time**: {self.solution_result.get('solve_time', 'N/A')} seconds
"""
                
                # Prioritize displaying the complete real solving output
                if 'full_output' in self.solution_result and self.solution_result['full_output']:
                    summary += f"\n### Complete solving output (real code output)\n```\n{self.solution_result['full_output']}\n```\n"
                
                # If there are variable values, also display (but mark that this is formatted)
                if 'decision_variables_values' in self.solution_result:
                    var_values = self.solution_result['decision_variables_values']
                    if var_values:
                        summary += f"\n### Formatted variable value summary ({len(var_values)} variables)\n"
                        summary += "*(Below is formatted display, please refer to the real code output above for complete output)*\n\n"
                        
                        # Smart grouping and sorting of variables
                        var_groups = {}
                        for var_name, var_value in var_values.items():
                            # Group by variable name
                            base_name = var_name.split('_')[0] if '_' in var_name else var_name
                            if base_name not in var_groups:
                                var_groups[base_name] = []
                            var_groups[base_name].append((var_name, var_value))
                        
                        # Display variables by group (limit display quantity to avoid too long）
                        displayed_count = 0
                        max_display = 20  # Maximum display quantity to avoid too long
                        
                        for group_name, variables in sorted(var_groups.items()):
                            if displayed_count >= max_display:
                                remaining = len(var_values) - displayed_count
                                summary += f"\n*... There are {remaining} variables, please refer to the complete output above*\n"
                                break
                                
                            if len(variables) > 1:
                                summary += f"\n#### {group_name.upper()} Series Variables:\n"
                                for var_name, var_value in sorted(variables[:5]):  # Each group displays up to 5
                                    if displayed_count >= max_display:
                                        break
                                    if isinstance(var_value, (int, float)) and var_value != 0:
                                        summary += f"- **{var_name}**: {var_value:.3f}\n"
                                    elif var_value == 0:
                                        summary += f"- **{var_name}**: {var_value:.0f}\n"
                                    else:
                                        summary += f"- **{var_name}**: {str(var_value)}\n"
                                    displayed_count += 1
                                if len(variables) > 5:
                                    summary += f"- *... There are {len(variables) - 5} variables*\n"
                            else:
                                var_name, var_value = variables[0]
                                if isinstance(var_value, (int, float)) and var_value != 0:
                                    summary += f"- **{var_name}**: {var_value:.3f}\n"
                                elif var_value == 0:
                                    summary += f"- **{var_name}**: {var_value:.0f}\n"
                                else:
                                    summary += f"- **{var_name}**: {str(var_value)}\n"
                                displayed_count += 1
                elif 'variables' in self.solution_result:
                    # Compatible with old version
                    summary += "\n### Variable values\n"
                    for var_name, var_value in self.solution_result['variables'].items():
                        summary += f"- **{var_name}**: {var_value}\n"
            else:
                summary += f"\n{str(self.solution_result)}"
        else:
            summary += "\nNo solving result obtained"
        
        # Append: All successful solving records
        if hasattr(self, 'solution_history') and self.solution_history:
            summary += "\n## Historical successful solving records\n"
            for idx, item in enumerate(self.solution_history, 1):
                status_note = "(Continued after verification: undecided)" if item.get("continued_after_verification") is None else ("(Continued after verification: yes)" if item.get("continued_after_verification") else "(Continued after verification: no)")
                summary += f"\n### Success package {idx} {status_note}\n"
                # Analysis
                summary += "\n**Modeling analysis (JSON)**\n\n```json\n"
                try:
                    summary += json.dumps(item.get("analysis", {}), ensure_ascii=False, indent=2)
                except Exception:
                    summary += str(item.get("analysis", {}))
                summary += "\n```\n"
                # Code
                summary += "\n**Model code (Pyomo)**\n\n```python\n"
                summary += item.get("model_code", "")
                summary += "\n```\n"
                # Solving result
                summary += "\n**Solving result**\n\n"
                try:
                    sol = item.get("solution", {})
                    summary += f"- Status: {sol.get('status', 'unknown')}\n"
                    summary += f"- Termination condition: {sol.get('termination_condition', 'N/A')}\n"
                    summary += f"- Objective value: {sol.get('objective_value', 'N/A')}\n"
                    summary += f"- Solver used: {sol.get('solver_used', 'N/A')}\n"
                    # Original solving complete output (no secondary processing)
                    if 'full_output' in sol and sol['full_output']:
                        summary += f"\n**Original solving complete output**\n\n````\n{sol['full_output']}\n````\n"
                except Exception:
                    pass
        
        # Solution verification result
        summary += f"""

## Solution verification"""
        
        if self.verification_result:
            verification_status = self.verification_result.get('verification_status', 'unknown')
            quality_score = self.verification_result.get('quality_score', 0.0)
            confidence = self.verification_result.get('confidence', 0.0)
            
            # Status icons
            status_icons = {
                'satisfactory': '✅',
                'needs_improvement': '⚠️',
                'failed': '❌'
            }
            status_icon = status_icons.get(verification_status, '❓')
            
            summary += f"""

**Verification status**: {status_icon} {verification_status.upper()}
**Quality score**: {quality_score:.3f}/1.0 ({quality_score*100:.1f}%)
**Confidence**: {confidence:.3f}/1.0

### Detailed analysis

#### Solution feasibility check
{self.verification_result.get('analysis', {}).get('solution_feasibility', 'No detailed analysis')}

#### Model consistency analysis
{self.verification_result.get('analysis', {}).get('model_consistency', 'No detailed analysis')}

#### Result reasonableness assessment
{self.verification_result.get('analysis', {}).get('result_reasonableness', 'No detailed analysis')}

#### Optimization quality judgment
{self.verification_result.get('analysis', {}).get('optimization_quality', 'No detailed analysis')}

### Improvement suggestions"""
            
            recommendations = self.verification_result.get('recommendations', [])
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    summary += f"\n{i}. {rec}"
            else:
                summary += "\n- No specific improvement suggestions"
            
            # If it is fallback mode, add explanation
            if self.verification_result.get('fallback_mode', False):
                summary += "\n\n> **Note**: This verification result is based on simplified analysis (API unavailable)"
            
            # If there is LLM decision analysis (Free mode), add to report
            if 'llm_decision_analysis' in self.verification_result:
                decision_analysis = self.verification_result['llm_decision_analysis']
                summary += f"""

### LLM final decision analysis (Free mode)

#### Overall evaluation
{decision_analysis.get('overall_assessment', 'No evaluation')}

#### Solution quality
{decision_analysis.get('solution_quality', 'No evaluation')}

#### Improvement potential
{decision_analysis.get('improvement_potential', 'No analysis')}

#### Recommended strategy
{decision_analysis.get('recommended_strategy', 'No recommendation')}

#### User recommendations"""
                
                user_recommendations = decision_analysis.get('user_recommendations', [])
                if user_recommendations:
                    for i, rec in enumerate(user_recommendations, 1):
                        summary += f"\n{i}. {rec}"
                else:
                    summary += "\n- No specific recommendations"
                
                summary += f"""

#### Final conclusion
{decision_analysis.get('conclusion', 'No conclusion')}"""
                
                if decision_analysis.get('analysis_mode') == 'fallback_analysis':
                    summary += "\n\n> **Note**: This decision analysis is based on simplified logic (LLM API unavailable)"
                
        else:
            summary += "\n\n- Solution verification result unavailable"
        
        # Debug statistics
        summary += f"""

## Debug process statistics
- **Total debug attempts**: {total_attempts}
- **Successful attempts**: {successful_attempts}
- **Overall success rate**: {success_rate:.2%}
"""
        
        # Agent performance statistics - using unified statistics method
        if total_attempts > 0:
            agent_stats = self._get_agent_statistics()
            
            summary += """
### Agent performance statistics
| Agent | Total attempts | Successful attempts | Success rate |
|-------|--------|----------|--------|
"""
            for agent, stats in agent_stats.items():
                agent_success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
                summary += f"| {agent} | {stats['total']} | {stats['success']} | {agent_success_rate:.2%} |\n"
            
            # Detailed debug records
            summary += """
### Detailed debug records
"""
            for i, record in enumerate(self.debug_records, 1):
                status = "✅" if record.success else "❌"
                summary += f"""
#### {i}. {record.timestamp.strftime('%H:%M:%S')} - {record.agent} {status}
- **Error type**: {record.error_type}
- **Strategy**: {record.strategy}
- **Duration**: {record.duration:.2f}s
- **Attempt number**: {record.attempt_number}
"""
                if record.error_details:
                    summary += f"- **Error details**: {record.error_details}\n"
                if record.fix_applied:
                    summary += f"- **Fix applied**: {record.fix_applied}\n"
        
        return summary
    
    def save_summary_to_file(self) -> None:
        """Save summary to file"""
        try:
            from pathlib import Path
            
            # Ensure saving to the correct directory
            debug_dir = Path("experiments/outputs/debug_reports")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Build complete path - check if it is already a complete path
            if Path(self.summary_file).is_absolute() or str(self.summary_file).startswith('experiments/'):
                # Already a complete path, use directly
                full_path = Path(self.summary_file)
            else:
                # Relative file name, need to concatenate directory
                full_path = debug_dir / self.summary_file
            
            summary_content = self.generate_markdown_summary()
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            # Update summary_file to complete path, so coordinator can return correct path
            self.summary_file = str(full_path)
            
            self.logger.info(f"Debug summary saved to {self.summary_file}")
        except Exception as e:
            self.logger.error(f"Save debug summary failed: {e}")
    
    def get_csv_summary_data(self) -> Dict[str, Any]:
        """Get CSV summary data"""
        total_duration = self.get_total_duration()
        total_attempts = len(self.debug_records)
        successful_attempts = sum(1 for r in self.debug_records if r.success)
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0
        
        # Extract key information from solution result
        objective_value = "N/A"
        solver_used = "N/A"
        final_status = "Failed"
        
        if self.solution_result and isinstance(self.solution_result, dict):
            objective_value = self.solution_result.get('objective_value', 'N/A')
            solver_used = self.solution_result.get('solver_used', 'N/A')
            final_status = self.solution_result.get('status', 'Unknown')
        
        # Statistics of each Agent usage - using unified statistics method
        agent_stats = self._get_agent_statistics()
        
        # Extract solve time and complete output information
        solve_time = "N/A"
        has_full_output = False
        full_output_length = 0
        num_variables = 0
        
        if self.solution_result and isinstance(self.solution_result, dict):
            solve_time = self.solution_result.get('solve_time', 'N/A')
            if 'full_output' in self.solution_result:
                has_full_output = True
                full_output_length = len(self.solution_result['full_output'])
            if 'decision_variables_values' in self.solution_result:
                num_variables = len(self.solution_result['decision_variables_values'])

        return {
            "problem_id": self.problem_id,
            "coordinator_mode": self.coordinator_mode,
            "start_time": self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else '',
            "total_duration": round(total_duration, 2),
            "final_status": final_status,
            "objective_value": objective_value,
            "solver_used": solver_used,
            "solve_time": solve_time,  # Add solve time
            "total_debug_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "debug_success_rate": round(success_rate, 4),
            "analyzer_attempts": agent_stats.get('analyzer', {}).get('total', 0),
            "analyzer_success": agent_stats.get('analyzer', {}).get('success', 0),
            "modeler_attempts": agent_stats.get('modeler', {}).get('total', 0),
            "modeler_success": agent_stats.get('modeler', {}).get('success', 0),
            "verifier_attempts": agent_stats.get('verifier', {}).get('total', 0),
            "verifier_success": agent_stats.get('verifier', {}).get('success', 0),
            "solver_attempts": agent_stats.get('solver', {}).get('total', 0),
            "solver_success": agent_stats.get('solver', {}).get('success', 0),
            "problem_description_length": len(self.problem_description),
            "model_code_length": len(self.final_model_code),
            "has_decision_variables": self._extract_decision_variables_count() > 0,
            "num_constraints": self._extract_constraints_count(),
            "num_parameters": self._extract_parameters_count(),
            "num_extracted_variables": num_variables,  # Add extracted variable quantity
            "has_full_output": has_full_output,  # Add whether there is a complete output mark
            "full_output_length": full_output_length  # Add complete output length
        }
    
    def _extract_decision_variables_count(self) -> int:
        """Extract decision variable quantity, compatible with new and old JSON formats"""
        if not self.analysis_result:
            return 0
        
        # New format: core_model_elements.decision_variables
        if 'core_model_elements' in self.analysis_result:
            decision_vars = self.analysis_result['core_model_elements'].get('decision_variables', [])
            return len(decision_vars) if isinstance(decision_vars, list) else 0
        
        # Compatible with old format: modeling_elements.decision_variables
        if 'modeling_elements' in self.analysis_result:
            decision_vars = self.analysis_result['modeling_elements'].get('decision_variables', [])
            return len(decision_vars) if isinstance(decision_vars, list) else 0
        
        # Compatible with old format: decision_variables
        decision_vars = self.analysis_result.get('decision_variables', [])
        return len(decision_vars) if isinstance(decision_vars, list) else 0
    
    def _extract_constraints_count(self) -> int:
        """Extract constraint quantity, compatible with new and old JSON formats"""
        if not self.analysis_result:
            return 0
        
        # New format: core_model_elements.constraints
        if 'core_model_elements' in self.analysis_result:
            constraints = self.analysis_result['core_model_elements'].get('constraints', [])
            return len(constraints) if isinstance(constraints, list) else 0
        
        # Compatible with old format: modeling_elements.constraints
        if 'modeling_elements' in self.analysis_result:
            constraints = self.analysis_result['modeling_elements'].get('constraints', [])
            return len(constraints) if isinstance(constraints, list) else 0
        
        # Compatible with old format: constraints
        constraints = self.analysis_result.get('constraints', [])
        return len(constraints) if isinstance(constraints, list) else 0
    
    def _extract_parameters_count(self) -> int:
        """Extract parameter quantity, compatible with new and old JSON formats"""
        if not self.analysis_result:
            return 0
        
        # New format: core_model_elements.parameters
        if 'core_model_elements' in self.analysis_result:
            parameters = self.analysis_result['core_model_elements'].get('parameters', [])
            return len(parameters) if isinstance(parameters, list) else 0
        
        # Compatible with old format: modeling_elements.parameters
        if 'modeling_elements' in self.analysis_result:
            parameters = self.analysis_result['modeling_elements'].get('parameters', [])
            return len(parameters) if isinstance(parameters, list) else 0
        
        # Compatible with old format: parameters
        parameters = self.analysis_result.get('parameters', [])
        return len(parameters) if isinstance(parameters, list) else 0
    
    def _format_analysis_json(self) -> str:
        """Format analysis result JSON, ensure clear display"""
        import json
        
        if not self.analysis_result:
            return "No analysis result generated"
        
        try:
            # Use JSON formatting, ensure readability
            formatted_json = json.dumps(self.analysis_result, indent=2, ensure_ascii=False)
            return formatted_json
        except Exception as e:
            # If JSON serialization fails, return string representation
            return f"Analysis result formatting failed: {str(e)}\nOriginal result: {str(self.analysis_result)}"
    
    def _get_agent_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get Agent statistics, avoid duplicate calculation"""
        agent_stats = {}
        for record in self.debug_records:
            if record.agent not in agent_stats:
                agent_stats[record.agent] = {"total": 0, "success": 0}
            agent_stats[record.agent]["total"] += 1
            if record.success:
                agent_stats[record.agent]["success"] += 1
        return agent_stats

class LLMDebugRouter:
    """LLM-driven intelligent debugging router"""
    
    def __init__(self, api_available: bool = True, model_type: str = "gemini"):
        self.api_available = api_available
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        if self.api_available:
            self.logger.info("LLM DebugRouter initialized successfully, using unified API call")
    
    def route_debug_request(self, error_info: Dict[str, Any], debug_history: List[DebugAttempt], 
                          success_rates: Dict[str, float]) -> Dict[str, Any]:
        """Intelligent routing debugging request"""
        
        if not self.api_available:
            return {
            "target_agent": "analyzer",
            "strategy": "Rebuild Model",
            "priority": "high",
            "estimated_success_rate": 0.5,
            "should_continue": True,
            "reasoning": f"JSON parsing failed. Return to analyzer."
        }
        
        try:
            return self._llm_based_routing(error_info, debug_history, success_rates)
        except Exception as e:
            self.logger.error(f"LLM routing failed: {e}")
            return {
            "target_agent": "analyzer",
            "strategy": "Rebuild Model",
            "priority": "high",
            "estimated_success_rate": 0.5,
            "should_continue": True,
            "reasoning": f"JSON parsing failed. Return to analyzer."
        }
    
    def _llm_based_routing(self, error_info: Dict[str, Any], debug_history: List[DebugAttempt], 
                          success_rates: Dict[str, float]) -> Dict[str, Any]:
        """Intelligent routing based on LLM"""
        
        # Prepare history record summary
        recent_history = debug_history[-12:] if len(debug_history) > 5 else debug_history
        history_summary = []
        for record in recent_history:
            history_summary.append({
                "agent": record.agent,
                "error_type": record.error_type,
                "strategy": record.strategy,
                "success": record.success,
                "duration": record.duration
            })
        
        # Build LLM prompt
        prompt = f"""You are an optimization problem debugging expert. Based on the following information, decide the best debugging strategy:

Current Error Information:
{json.dumps(error_info, indent=2, ensure_ascii=False)}

Historical Debugging Record Summary:
{json.dumps(history_summary, indent=2, ensure_ascii=False)}

Historical Success Rates by Agent:
- ModelingAgent Success Rate: {success_rates.get('modeler', 0.5):.2f}
- AnalyzerAgent Success Rate: {success_rates.get('analyzer', 0.5):.2f}

Please analyze the current error type (SyntaxError/Infeasible/NoSolution, etc.), combine with historical debugging records including error information, failure types, success rates, and agent response times, and return a JSON format decision:
```json
{{
    "target_agent": "modeler|analyzer",  // Forward to ModelingAgent or AnalyzerAgent
    "strategy": "Specific modification strategy description",
    "priority": "high|medium|low",  // Priority level
    "estimated_success_rate": 0.0-1.0,  // Estimated success rate
    "should_continue": true|false,  // Whether to continue trying
    "reasoning": "Decision reasoning based on historical records and current error"
}}
```

**Decision Principles:**
1. **Analyze Current Error Type**: SyntaxError/environment errors prioritize modeler, model structure/constraint issues prioritize analyzer
2. **Intelligent Decision**: Choose the most suitable agent based on historical records and current problem characteristics
3. **Switch Agent**: If modeler keeps making the same type of error, switch to analyzer
"""
        
        # Use unified LLM API call
        response_text = call_llm_api(prompt, self.model_type)
        
        # Strategy 1: Extract ```json``` block
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            if json_end != -1:
                response_text = response_text[json_start:json_end].strip()
                print("🔧 Extracting JSON from ```json``` block")
            else:
                # If no end marker, take to the end
                response_text = response_text[json_start:].strip()
                print("🔧 Extracting JSON from ```json``` block (no end marker)")
        
        # Strategy 2: Extract largest complete JSON object
        elif '{' in response_text:
            json_start = response_text.find('{')
            brace_count = 0
            json_end = -1
            
            for i, char in enumerate(response_text[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end != -1:
                response_text = response_text[json_start:json_end]
                print("🔧 Extracting largest complete JSON object")
            else:
                # Fall back to original method
                json_end = response_text.rfind('}') + 1
                response_text = response_text[json_start:json_end]
                print("🔧 Extracting JSON fragment from text")
        
        # Clean up common JSON format issues
        response_text = response_text.replace('```', '').strip()
        # Handle potential encoding issues
        response_text = response_text.replace('"', '"').replace('"', '"')
        
        try:
            decision = json.loads(response_text)
            self.logger.info(f"LLM routing decision: {decision.get('target_agent')} - {decision.get('strategy')}")
            return decision
        except json.JSONDecodeError as e:
            self.logger.error(f"LLM response JSON parsing failed: {e}")
            return {
            "target_agent": "analyzer",
            "strategy": "Rebuild Model",
            "priority": "high",
            "estimated_success_rate": 0.5,
            "should_continue": True,
            "reasoning": f"JSON parsing failed. Return to analyzer."
        }
    

class ErrorClassifier:
    """Error classifier"""
    
    @staticmethod
    def classify_error(error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Classify errors and provide detailed information"""
        error_msg = str(error_info.get("error", "")).lower()
        error_type = "unknown"
        severity = "medium"
        suggested_agent = "analyzer"
        
        # Syntax error
        if any(keyword in error_msg for keyword in ["syntax", "invalid", "unexpected", "indentation"]):
            error_type = "SyntaxError"
            severity = "low"
            suggested_agent = "modeler"
        
        # Constraint error
        elif any(keyword in error_msg for keyword in ["constraint", "infeasible", "unbounded", "conflicting"]):
            error_type = "ConstraintError"
            severity = "high"
            suggested_agent = "analyzer"
        
        # Solver error
        elif any(keyword in error_msg for keyword in ["solver", "optimization", "convergence"]):
            error_type = "SolverError"
            severity = "medium"
            suggested_agent = "modeler"
        
        # Import error
        elif any(keyword in error_msg for keyword in ["import", "module", "package"]):
            error_type = "ImportError"
            severity = "low"
            suggested_agent = "modeler"
        
        # Model structure error
        elif any(keyword in error_msg for keyword in ["model", "variable", "objective", "structure"]):
            error_type = "ModelStructureError"
            severity = "high"
            suggested_agent = "analyzer"
        
        return {
            "error_type": error_type,
            "severity": severity,
            "suggested_agent": suggested_agent,
            "classification_confidence": 0.8 if error_type != "unknown" else 0.3
        } 