import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class JSONTrackingManager:
    """JSON analysis tracking manager - record all JSON analysis during solving process"""
    
    def __init__(self, problem_id: str = None, enable_tracking: bool = True):
        """
        Initialize JSON tracking manager
        
        Args:
            problem_id: problem ID
            enable_tracking: whether to enable tracking function
        """
        self.problem_id = problem_id or f"problem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_tracking = enable_tracking
        self.json_records = []  # store all JSON records
        self.solver_success_indices = []  # record JSON index when Solver succeeds
        
        # file path
        # use unified output directory
        from pathlib import Path
        output_dir = Path("engillm_outputs/json_tracking")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # ensure file name clearly shows problem ID
        self.tracking_file = str(output_dir / f"json_tracking_{self.problem_id}_{timestamp}_{timestamp}.json")
        self.analysis_md_file = f"json_analysis_{self.problem_id}_{timestamp}.md"
        
        logger.info(f"JSON tracking manager initialized - problem ID: {self.problem_id}, enable tracking: {enable_tracking}")
    
    def record_json_analysis(self, json_data: Dict[str, Any], context: Dict[str, Any] = None) -> None:
        """
        record one JSON analysis
        
        Args:
            json_data: JSON analysis data
            context: context information
        """
        if not self.enable_tracking:
            return
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'record_index': len(self.json_records),
            'json_data': json_data,
            'context': context or {},
            'is_solver_success': False  # default is False, may be updated later
        }
        
        self.json_records.append(record)
        logger.info(f"record JSON analysis #{len(self.json_records)}")
    
    def mark_solver_success(self, json_index: int = None) -> None:
        """
        mark JSON corresponding to solver success
        
        Args:
            json_index: JSON record index, if None, use the latest record
        """
        if not self.enable_tracking:
            return
        
        if json_index is None:
            json_index = len(self.json_records) - 1
        
        if 0 <= json_index < len(self.json_records):
            self.json_records[json_index]['is_solver_success'] = True
            if json_index not in self.solver_success_indices:
                self.solver_success_indices.append(json_index)
            logger.info(f"mark JSON record #{json_index} as solver success")
        else:
            logger.warning(f"invalid JSON index: {json_index}")
    
    def get_all_json_records(self) -> List[Dict[str, Any]]:
        """get all JSON records"""
        return self.json_records.copy()
    
    def get_solver_success_indices(self) -> List[int]:
        """get Solver successful JSON index list"""
        return self.solver_success_indices.copy()
    
    def save_tracking_data(self, output_file: str = None) -> None:
        """
        save tracking data to file
        
        Args:
            output_file: output file path, if None, use default path
        """
        if not self.enable_tracking:
            return
        
        output_file = output_file or self.tracking_file
        
        tracking_data = {
            'problem_id': self.problem_id,
            'tracking_enabled': self.enable_tracking,
            'total_records': len(self.json_records),
            'solver_success_count': len(self.solver_success_indices),
            'solver_success_indices': self.solver_success_indices,
            'json_records': self.json_records,
            'export_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON tracking data saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"save JSON tracking data failed: {e}")
            raise
    
    def generate_analysis_markdown(self, output_file: str = None) -> None:
        """
        generate Markdown report containing all JSON analysis
        
        Args:
            output_file: output file path, if None, use default path
        """
        if not self.enable_tracking:
            return
        
        output_file = output_file or self.analysis_md_file
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# JSON analysis tracking report\n\n")
                f.write(f"## Basic information\n")
                f.write(f"- **problem ID**: {self.problem_id}\n")
                f.write(f"- **generation time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **JSON record total**: {len(self.json_records)}\n")
                f.write(f"- **Solver success count**: {len(self.solver_success_indices)}\n")
                
                if self.solver_success_indices:
                    f.write(f"- **Solver successful JSON index**: {', '.join(map(str, self.solver_success_indices))}\n")
                
                f.write(f"\n## Detailed JSON analysis record\n\n")
                
                for i, record in enumerate(self.json_records):
                    success_marker = " ⭐ **Solver success**" if record.get('is_solver_success', False) else ""
                    f.write(f"### JSON record #{i+1}{success_marker}\n\n")
                    
                    f.write(f"**timestamp**: {record['timestamp']}\n\n")
                    
                    # context information
                    context = record.get('context', {})
                    if context:
                        f.write(f"**context information**:\n")
                        for key, value in context.items():
                            f.write(f"- {key}: {value}\n")
                        f.write(f"\n")
                    
                    # JSON data
                    f.write(f"**JSON analysis data**:\n\n")
                    f.write(f"```json\n")
                    f.write(json.dumps(record['json_data'], indent=2, ensure_ascii=False))
                    f.write(f"\n```\n\n")
                    
                    f.write(f"---\n\n")
            
            logger.info(f"JSON analysis Markdown report saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"generate JSON analysis Markdown report failed: {e}")
            raise
    
    def load_tracking_data(self, input_file: str) -> None:
        """
        load tracking data from file
        
        Args:
            input_file: input file path
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                tracking_data = json.load(f)
            
            self.problem_id = tracking_data.get('problem_id', self.problem_id)
            self.json_records = tracking_data.get('json_records', [])
            self.solver_success_indices = tracking_data.get('solver_success_indices', [])
            
            logger.info(f"successfully load JSON tracking data: {len(self.json_records)} records")
            
        except Exception as e:
            logger.error(f"load JSON tracking data failed: {e}")
            raise
    
    def clear_records(self) -> None:
        """clear all records"""
        self.json_records.clear()
        self.solver_success_indices.clear()
        logger.info("clear all JSON tracking records")
    
    def get_statistics(self) -> Dict[str, Any]:
        """get tracking statistics information"""
        return {
            'total_records': len(self.json_records),
            'solver_success_count': len(self.solver_success_indices),
            'solver_success_rate': len(self.solver_success_indices) / len(self.json_records) if self.json_records else 0,
            'tracking_enabled': self.enable_tracking,
            'problem_id': self.problem_id
        }

class JSONTrackingIntegrator:
    """JSON tracking integrator - integrate with existing system"""
    
    @staticmethod
    def integrate_with_debug_summary(debug_summary, json_tracking_manager: JSONTrackingManager):
        """
        integrate JSON tracking manager with DebugSummary
        
        Args:
            debug_summary: DebugSummary instance
            json_tracking_manager: JSONTrackingManager instance
        """
        # save original set_analysis_result method
        original_set_analysis_result = debug_summary.set_analysis_result
        
        def enhanced_set_analysis_result(analysis_result):
            # call original method
            result = original_set_analysis_result(analysis_result)
            
            # record JSON analysis, include access source information
            if json_tracking_manager.enable_tracking:
                # ✅ use coordinator recorded agent source first
                import inspect
                previous_agent = "unknown"
                
                # try to find coordinator instance from call stack, get its current_agent_source
                for frame_info in inspect.stack():
                    frame_locals = frame_info.frame.f_locals
                    # find coordinator instance
                    if 'self' in frame_locals:
                        coordinator_obj = frame_locals['self']
                        if hasattr(coordinator_obj, 'current_agent_source') and hasattr(coordinator_obj, 'coordinator_mode'):
                            previous_agent = coordinator_obj.current_agent_source
                            break
                
                # if not found coordinator recorded source, back to call stack analysis
                if previous_agent == "unknown":
                    # check call stack, find previous agent information
                    for frame_info in inspect.stack():
                        frame_locals = frame_info.frame.f_locals
                        function_name = frame_info.function.lower()
                        filename = frame_info.filename.lower()
                        
                        # check if from verification related call
                        if ('verification' in filename or 'verifier' in filename or
                            any('verification' in str(val).lower() for val in frame_locals.values() if isinstance(val, str))):
                            previous_agent = "verifier"
                            break
                        
                        # check if from solver related call
                        if ('solver' in filename or
                            any('solver' in str(val).lower() for val in frame_locals.values() if isinstance(val, str))):
                            previous_agent = "solver"
                            break
                        
                        # check specific coordinator method to infer source
                        if 'coordinator' in filename:
                            # from verifier error handling analyzer retry
                            if function_name in ['handle_verifier_error', '_restart_fixed_mode_debug_cycle']:
                                previous_agent = "verifier"
                                break
                            # from solver error handling analyzer retry - extended check range
                            elif function_name in ['_execute_analyzer_retry', 'handle_solver_error', '_execute_fixed_debug_cycle', '_execute_modeling_retry']:
                                previous_agent = "solver"
                                break
                            # check error type parameter to infer source
                            elif 'error_type' in frame_locals:
                                error_type_val = str(frame_locals.get('error_type', '')).lower()
                                if any(err_type in error_type_val for err_type in ['solvererror', 'constrainterror', 'syntaxerror']):
                                    previous_agent = "solver"
                                    break
                                elif 'verification' in error_type_val:
                                    previous_agent = "verifier"
                                    break
                            # Free mode restart decision (based on verification result)
                            elif 'restart' in function_name and any('verif' in str(val).lower() for val in frame_locals.values() if isinstance(val, str)):
                                previous_agent = "verifier"
                                break
                            # Solution Verifier related restart (Fixed mode)
                            elif any('solution_verifier' in str(val).lower() for val in frame_locals.values() if isinstance(val, str)):
                                previous_agent = "solution verifier"
                                break
                    
                    # if still not found clear source, check if it is initial analysis
                    if previous_agent == "unknown":
                        # check if it is initial problem analysis (no debug record or record few)
                        if not hasattr(debug_summary, 'debug_records') or len(debug_summary.debug_records) <= 1:
                            previous_agent = "initial"
                        else:
                            # note: coordinator cannot be considered as source agent
                            # if cannot determine source, keep as "unknown" instead of setting to "coordinator"
                            previous_agent = "unknown"
                
                context = {
                    'source': 'ProblemAnalyzer',
                    'stage': 'analysis',
                    'problem_id': debug_summary.problem_id,
                    'previous_agent': previous_agent,  # new: record previous agent
                    'access_type': 'retry' if previous_agent in ['solver', 'verifier'] else 'initial'
                }
                json_tracking_manager.record_json_analysis(analysis_result, context)
            
            return result
        
        # replace method
        debug_summary.set_analysis_result = enhanced_set_analysis_result
        logger.info("JSON tracking integrated into DebugSummary (include access source tracking)")
    
    @staticmethod
    def integrate_with_coordinator(coordinator, json_tracking_manager: JSONTrackingManager):
        """
        integrate JSON tracking manager with Coordinator
        
        Args:
            coordinator: Coordinator instance
            json_tracking_manager: JSONTrackingManager instance
        """
        # save original handle_response_message method
        original_handle_response = coordinator.handle_response_message
        
        def enhanced_handle_response(message):
            # call original method
            result = original_handle_response(message)
            
            # if solver success response, mark JSON
            # check multiple success cases: normal response, contains solution message etc.
            if (json_tracking_manager.enable_tracking and 
                message.sender == "solver"):
                
                # check message type and content, judge if it is success solving
                is_success = False
                
                if message.msg_type.value == "response":
                    is_success = True
                elif hasattr(message, 'content') and isinstance(message.content, dict):
                    # check if content contains success solving flag
                    content = message.content
                    if (content.get('status') == 'success' or 
                        'objective_value' in content or
                        'solution' in content or
                        content.get('solver_status') in ['optimal', 'feasible']):
                        is_success = True
                
                if is_success:
                    json_tracking_manager.mark_solver_success()
                    logger.info(f"mark solver success solving point: JSON record #{len(json_tracking_manager.json_records)-1}")
            
            return result
        
        # replace method
        coordinator.handle_response_message = enhanced_handle_response
        logger.info("JSON tracking integrated into Coordinator")


def create_json_tracking_manager(problem_id: str = None, enable_tracking: bool = True) -> JSONTrackingManager:
    """
    create JSON tracking manager factory function
    
    Args:
        problem_id: problem ID
        enable_tracking: whether to enable tracking
        
    Returns:
        JSONTrackingManager instance
    """
    return JSONTrackingManager(problem_id=problem_id, enable_tracking=enable_tracking)


# usage example
if __name__ == "__main__":
    # create tracking manager
    tracker = create_json_tracking_manager("test_problem", enable_tracking=True)
    
    # simulate record some JSON analysis
    sample_json1 = {
        "modeling_context": {"problem_essence": "test problem 1"},
        "core_model_elements": {"decision_variables": []}
    }
    
    sample_json2 = {
        "modeling_context": {"problem_essence": "test problem 2"},
        "core_model_elements": {"decision_variables": []}
    }
    
    tracker.record_json_analysis(sample_json1, {"iteration": 1})
    tracker.record_json_analysis(sample_json2, {"iteration": 2})
    
    # mark second JSON as solver success
    tracker.mark_solver_success(1)
    
    # save data
    tracker.save_tracking_data("test_tracking.json")
    tracker.generate_analysis_markdown("test_analysis.md")
    
    # print statistics information
    stats = tracker.get_statistics()
    print("tracking statistics information:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

