from base_agents import BaseAgent, Message, MessageType
from debug_framework import DebugSummary, LLMDebugRouter, ErrorClassifier, DebugAttempt
from json_tracking_manager import JSONTrackingManager, JSONTrackingIntegrator
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json
import time
import traceback
import re
from specialized_agents import call_llm_api
# Coordinator classes are defined in this file

class BaseCoordinator(BaseAgent):
    """Base Coordinator Class"""
    
    def __init__(self, agent_id: str = "coordinator", mode: str = "fixed", enable_json_tracking: bool = False, model_type: str = "gemini", problem_id: str = None):
        super().__init__(agent_id)
        # Delayed import to avoid circular dependencies
        from specialized_agents import ProblemAnalyzer, ModelingAgent, VerificationAgent, SolverAgent, SolutionVerificationAgent
        
        self.model_type = model_type.lower()
        self.agents = {
            "analyzer": ProblemAnalyzer("analyzer", self.model_type),
            "modeler": ModelingAgent("modeler", self.model_type),
            "verifier": VerificationAgent("verifier", self.model_type),
            "solver": SolverAgent("solver", self.model_type),
            "solution_verifier": SolutionVerificationAgent("solution_verifier", self.model_type)
        }
        
        # Basic configuration
        self.coordinator_mode = mode
        self.workflow_state = "idle"
        self.solution_history = []
        self.pending_messages = []
        
        # Debug status flag
        self.in_debug_mode = False  # Mark whether in debug mechanism
        
        # Debug framework
        self.debug_summary = DebugSummary(problem_id=problem_id, model_type=self.model_type)
        self.error_classifier = ErrorClassifier()
        
        # JSON tracking functionality
        self.enable_json_tracking = enable_json_tracking
        self.json_tracking_manager = None
        if enable_json_tracking:
            self.json_tracking_manager = JSONTrackingManager(problem_id=problem_id, enable_tracking=True)
            self.logger.info("JSON tracking functionality enabled")
        
        # ✅ Add agent source tracking
        self.current_agent_source = "initial"  # Track the source agent currently calling analyzer
        
        # Remove static success rate statistics, change to dynamic calculation
        
        self.logger.info(f"CoordinatorAgent initialization completed, mode: {mode}, JSON tracking: {enable_json_tracking}")
    
    def select_coordinator_mode(self, problem_input: str) -> str:
        """Autonomously select coordinator mode based on input (Only used in initial selection)"""
        
        # Analyze problem complexity indicators
        complexity_indicators = {
            "high": ["nonlinear", "integer programming", "multi-objective", "stochastic", "dynamic", "stochastic", "nonlinear", "integer", "multi-objective"],
            "medium": ["constraint", "optimization", "planning", "scheduling", "constraint", "optimization", "scheduling"],
            "low": ["simple", "linear", "single-objective", "simple", "linear", "single"]
        }
        
        problem_lower = problem_input.lower()
        complexity_score = 0
        
        # Calculate complexity score
        for indicator in complexity_indicators["high"]:
            if indicator in problem_lower:
                complexity_score += 3
        
        for indicator in complexity_indicators["medium"]:
            if indicator in problem_lower:
                complexity_score += 2
        
        for indicator in complexity_indicators["low"]:
            if indicator in problem_lower:
                complexity_score += 1
        
        # Make decision based on complexity score (don't use historical success rate, as this is initial selection)
        if complexity_score >= 6:
            selected_mode = "free"  # High complexity uses free mode
        elif complexity_score >= 3:
            selected_mode = "fixed"  # Medium complexity uses fixed mode
        else:
            selected_mode = "fixed"  # Low complexity uses fixed mode
        
        self.logger.info(f"Selected coordinator mode based on problem complexity ({complexity_score}): {selected_mode}")
        return selected_mode
    
    
    def get_current_success_rates(self) -> Dict[str, float]:
        """Get current success rates for each Agent (calculated based on debug records)"""
        return {
            "modeler": self.debug_summary.get_agent_success_rate("modeler"),
            "analyzer": self.debug_summary.get_agent_success_rate("analyzer"),
            "verifier": self.debug_summary.get_agent_success_rate("verifier"),
            "solver": self.debug_summary.get_agent_success_rate("solver")
        }
    
    def process_message(self, message: Message) -> None:
        try:
            if message.msg_type == MessageType.ERROR:
                self.handle_error_message(message)
            elif message.msg_type == MessageType.RESPONSE:
                self.handle_response_message(message)
            elif message.msg_type == MessageType.RETRY:
                self.handle_retry_message(message)
        except Exception as e:
            self.handle_error(e, {"message": message.to_dict()})
    
    def is_solver_error(self, message: Message) -> bool:
        """Determine if it's a solver stage error, only solver errors enter debug mechanism"""
        return (message.sender == "solver" and 
                message.msg_type == MessageType.ERROR)
    
    def handle_pre_solver_error(self, message: Message) -> None:
        """Handle pre-solver errors: Record + Direct RETRY (except verification)"""
        error_info = message.content
        sender = message.sender
        
        self.logger.info(f"Handling pre-solver error: {sender} -> {error_info.get('error_type', 'Unknown')}")
        
        # Special handling for verification: transfer to analyzer for re-extraction
        # ⭐ But first check if code execution error was misidentified as modeling mismatch
        if sender == "verifier" and error_info.get("mismatch_detected", False):
            # Build detailed retry information
            retry_data = {
                "action": "re_extract_from_mismatch",
                "original_problem": self.original_problem,
                "mismatch_info": error_info,
                "verification_details": error_info.get("mismatch_details", {}),
                "previous_analysis": getattr(self.debug_summary, 'analysis_result', {}),
                "rejected_model_code": getattr(self.debug_summary, 'final_model_code', ""),
                "strategy": "Re-analyze based on verification mismatch details"
            }

            # Note: Remove secondary flattening on Coordinator side, trust top-level fields already provided by VerificationAgent
            
            retry_msg = Message("coordinator", "analyzer", retry_data, MessageType.RETRY)
            
            # Record analyzer processing time
            self.current_agent_source = "verifier"  # Set source as verifier error
            analyzer_start_time = time.time()
            response = self.agents["analyzer"].process_message(retry_msg)
            analyzer_duration = time.time() - analyzer_start_time
            
            # Record analyzer retry debug attempt (with actual processing time)
            self.debug_summary.record_attempt({
                "agent": "analyzer",
                "error_type": "mismatch_retry",
                "strategy": "Re-analyze from modeling mismatch",
                "success": response and response.msg_type == MessageType.RESPONSE,
                "duration": analyzer_duration,
                "coordinator_mode": self.coordinator_mode,
                "attempt_number": 0,
                "error_details": str(error_info.get("error", ""))[:200],
                "fix_applied": "Modeling mismatch, transferred to analyzer for re-extraction"
            })
            
            if response:
                self.pending_messages.append(response)
            return
        
        # For other agents, direct RETRY
        if sender in self.agents:
            retry_data = {
                "action": "direct_retry",
                "error_info": error_info
            }
            
            # For analyzer retry, need to include original problem and context
            if sender == "analyzer":
                retry_data["original_problem"] = self.original_problem
                retry_data["strategy"] = "Re-analyze based on error information"
                retry_data["previous_analysis"] = getattr(self.debug_summary, 'analysis_result', {})
            
            # For modeler retry, need to include analysis result and error details
            elif sender == "modeler":
                retry_data["analysis_result"] = getattr(self.debug_summary, 'analysis_result', {})
                retry_data["strategy"] = "Re-model based on error information"
                retry_data["previous_model_code"] = getattr(self.debug_summary, 'final_model_code', "")
            
            # For verifier retry, need to include pyomo code and original problem
            elif sender == "verifier":
                # verifier's error info now contains actual verified pyomo code
                retry_data["pyomo_code"] = error_info.get("pyomo_code", "")
                # If no code in error info (old version compatibility), get from debug_summary
                if not retry_data["pyomo_code"] and hasattr(self, 'debug_summary') and self.debug_summary.final_model_code:
                    retry_data["pyomo_code"] = self.debug_summary.final_model_code
                retry_data["original_problem"] = self.original_problem
                retry_data["strategy"] = "Re-verify based on error information"
            
            retry_msg = Message("coordinator", sender, retry_data, MessageType.RETRY)
            
            # Record agent retry processing time
            retry_start_time = time.time()
            response = self.agents[sender].process_message(retry_msg)
            processing_duration = time.time() - retry_start_time
            
            # Record retry attempt
            attempt_info = {
                "agent": sender,
                "error_type": error_info.get("error_type", "unknown"),
                "strategy": "direct_retry",
                "success": response and response.msg_type == MessageType.RESPONSE,
                "duration": processing_duration,
                "coordinator_mode": self.coordinator_mode,
                "attempt_number": 0,
                "error_details": str(error_info.get("error", ""))[:200],
                "fix_applied": "Direct RETRY processing"
            }
            self.debug_summary.record_attempt(attempt_info)
            
            if response:
                self.pending_messages.append(response)
        else:
            self.logger.error(f"Unknown sender: {sender}")
            error_msg = Message(
                "coordinator",
                "coordinator", 
                {
                    "error": f"Unknown sender: {sender}", 
                    "error_type": "CoordinatorError", 
                    "stage": "retry_routing"
                }, 
                MessageType.ERROR
            )
            self.pending_messages.append(error_msg)
    
    def handle_response_message(self, message: Message) -> None:
        """Handle normal response messages"""
        sender = message.sender
        content = message.content
        
        print(f"\n📩 [Coordinator] Received response message from {sender}")
        
        if sender == "analyzer":
            # Analysis completed, start modeling
            start_time = time.time()
            print("🎯 Analysis phase completed, preparing to start modeling phase")
            self.workflow_state = "modeling"
            self.logger.info("Analysis completed, starting modeling...")
            
            # Save analysis result
            if isinstance(content, dict):
                self.debug_summary.set_analysis_result(content)
            
            modeling_msg = Message("coordinator", "modeler", content, MessageType.QUERY)
            response = self.agents["modeler"].process_message(modeling_msg)
            if response:
                self.pending_messages.append(response)
            
            # Record actual processing time after analyzer completion
            duration = time.time() - start_time
            self.debug_summary.record_attempt({
                "agent": sender,
                "error_type": "success",
                "strategy": "Analysis completed, forwarding to modeling",
                "success": True,
                "duration": duration,
                "coordinator_mode": self.coordinator_mode,
                "attempt_number": 0,
                "error_details": "",
                "fix_applied": "Normal execution, starting modeling phase"
            })
                
        elif sender == "modeler":
            # Modeling completed, start verification
            start_time = time.time()
            print("🎯 Modeling phase completed, preparing to start verification phase")
            self.workflow_state = "verifying"
            self.logger.info("Modeling completed, starting verification...")
            
            # Save model code
            if isinstance(content, str):
                self.debug_summary.set_final_model(content)
            
            verification_msg = Message("coordinator", "verifier", content, MessageType.QUERY, 
                                      metadata={"original_problem": self.original_problem})
            response = self.agents["verifier"].process_message(verification_msg)
            if response:
                self.pending_messages.append(response)
            
            # Record actual processing time after modeler completion
            duration = time.time() - start_time
            self.debug_summary.record_attempt({
                "agent": sender,
                "error_type": "success",
                "strategy": "Modeling completed, forwarding to verification",
                "success": True,
                "duration": duration,
                "coordinator_mode": self.coordinator_mode,
                "attempt_number": 0,
                "error_details": "",
                "fix_applied": "Normal execution, starting verification phase"
            })
                
        elif sender == "verifier":
            # Verification completed, start solving
            start_time = time.time()
            print("🎯 Verification phase completed, preparing to start solving phase")
            self.workflow_state = "solving"
            self.logger.info("Verification completed, starting solving...")
            solving_msg = Message("coordinator", "solver", content, MessageType.QUERY)
            response = self.agents["solver"].process_message(solving_msg)
            if response:
                self.pending_messages.append(response)
            
            # Record actual processing time after verifier completion
            duration = time.time() - start_time
            self.debug_summary.record_attempt({
                "agent": sender,
                "error_type": "success",
                "strategy": "Verification completed, forwarding to solving",
                "success": True,
                "duration": duration,
                "coordinator_mode": self.coordinator_mode,
                "attempt_number": 0,
                "error_details": "",
                "fix_applied": "Normal execution, starting solving phase"
            })
                
        elif sender == "solver":
            # Solving completed, start solution verification
            start_time = time.time()
            print("🎯 Solving phase completed, starting solution verification")
            self.workflow_state = "verifying_solution"
            self.in_debug_mode = False  # Exit debug mode
            self.logger.info("Solving completed, starting solution verification...")
            
            # Save solution result - Add defensive check to ensure it's not an accidental simulation result
            if isinstance(content, dict):
                # Defensive check: Make sure this is a real solver result, not an accidental simulation result
                if content.get('status') == 'simulated' and content.get('solver_used') == 'intelligent_simulation':
                    self.logger.warning("⚠️ Detected simulation result from solver, this might be an error!")
                    self.logger.warning(f"⚠️ Content: {content}")
                    # Don't save simulation result, record as error instead
                    error_msg = f"Solver returned unexpected simulation result: {content}"
                    raise Exception(error_msg)
                else:
                    self.debug_summary.set_solution_result(content)
                # Add current solution to history (regardless of whether it continues)
                try:
                    success_entry = {
                        "timestamp": datetime.now().timestamp(),
                        "solution": content,
                        "analysis": getattr(self.debug_summary, 'analysis_result', {}) or {},
                        "model_code": getattr(self.debug_summary, 'final_model_code', "") or "",
                        "coordinator_mode": self.coordinator_mode,
                        "continued_after_verification": None  # Will be filled in later at verification conclusion
                    }
                    self.solution_history.append(success_entry)
                    # Sync to DebugSummary for md rendering
                    if hasattr(self.debug_summary, 'solution_history'):
                        self.debug_summary.solution_history = self.solution_history
                except Exception:
                    pass
            
            # Prepare verification data
            verification_data = self._prepare_verification_data(content)
            verification_msg = Message("coordinator", "solution_verifier", verification_data, MessageType.QUERY)
            response = self.agents["solution_verifier"].process_message(verification_msg)
            if response:
                self.pending_messages.append(response)
            
            # Record solver completion time
            duration = time.time() - start_time
            self.debug_summary.record_attempt({
                "agent": sender,
                "error_type": "success",
                "strategy": "Solving completed, forwarding to solution verification",
                "success": True,
                "duration": duration,
                "coordinator_mode": self.coordinator_mode,
                "attempt_number": 0,
                "error_details": "",
                "fix_applied": "Normal execution, starting solution verification phase"
            })
        
        elif sender == "solution_verifier":
            # Solution verification completed, handle final result
            start_time = time.time()
            print("🎯 Solution verification completed!")
            self.logger.info("Solution verification completed!")
            
            # Save verification result
            verification_result = content
            self.debug_summary.set_verification_result(verification_result)
            # Fill in "continue" flag for most recent success package
            try:
                if self.solution_history:
                    # In free mode, will decide based on should_restart, mark default value here
                    self.solution_history[-1]["continued_after_verification"] = None
                    if hasattr(self.debug_summary, 'solution_history'):
                        self.debug_summary.solution_history = self.solution_history
            except Exception:
                pass
            
            # Unified solution verification processing entry point
            self.handle_solution_verification(verification_result, content)
            
            # Record verification completion time
            duration = time.time() - start_time
            self.debug_summary.record_attempt({
                "agent": sender,
                "error_type": "success",
                "strategy": "Solution verification completed, handling final result",
                "success": True,
                "duration": duration,
                "coordinator_mode": self.coordinator_mode,
                "attempt_number": 0,
                "error_details": "",
                "fix_applied": f"Verification completed, {self.coordinator_mode} mode handling"
            })
    
    def handle_retry_message(self, message: Message) -> None:
        """Process retry message - forward to target agent"""
        target_agent = message.receiver
        
        self.logger.info(f"Processing retry message: coordinator -> {target_agent}")
        
        # Invoke target agent to process retry message
        if target_agent in self.agents:
            response = self.agents[target_agent].process_message(message)
            if response:
                self.pending_messages.append(response)
        else:
            self.logger.error(f"Unknown target agent: {target_agent}")
            error_msg = Message(
                "coordinator",
                "coordinator", 
                {
                    "error": f"Unknown target agent: {target_agent}", 
                    "error_type": "CoordinatorError", 
                    "stage": "retry_routing"
                }, 
                MessageType.ERROR
            )
            self.pending_messages.append(error_msg)
    
    def _save_to_csv(self) -> None:
        """Save to CSV summary file"""
        import csv
        import os
        
        # Create different CSV files by coordinator mode for comparison analysis
        csv_file = f"experiment_summary_{self.coordinator_mode}.csv"
        csv_data = self.debug_summary.get_csv_summary_data()
        
        # Check if file exists, if not create and write header
        file_exists = os.path.exists(csv_file)
        
        try:
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                
                # If file doesn't exist, write header
                if not file_exists:
                    writer.writeheader()
                
                # Write data
                writer.writerow(csv_data)
                
            self.logger.info(f"Experiment data saved to {csv_file}")
        except Exception as e:
            self.logger.error(f"Failed to save CSV data: {e}")
    
    def _prepare_verification_data(self, solution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare complete data to send to SolutionVerificationAgent"""
        verification_data = {
            "original_problem": self.original_problem,
            "analysis_result": self.debug_summary.analysis_result if hasattr(self.debug_summary, 'analysis_result') else {},
            "model_code": self.debug_summary.final_model_code if hasattr(self.debug_summary, 'final_model_code') else "",
            "solution_result": solution_result,
            "debug_history": self.debug_summary.debug_records if hasattr(self.debug_summary, 'debug_records') else [],
            "coordinator_mode": self.coordinator_mode
        }
        
        self.logger.info(f"Prepared verification data: original problem length={len(verification_data['original_problem'])}, "
                        f"model code length={len(verification_data['model_code'])}, "
                        f"debug records={len(verification_data['debug_history'])} entries")
        
        return verification_data
    
    def _make_restart_decision(self, verification_result: Dict[str, Any]) -> bool:
        """Unified restart decision: rule-based judgment based on quality score, verification status, and solution feasibility"""
        quality_score = verification_result.get("quality_score", 0.0)
        verification_status = verification_result.get("verification_status", "unknown")
        should_iterate_hint = verification_result.get("should_iterate", False)
        
        # Check basic feasibility of the solution
        solution_result = getattr(self.debug_summary, 'solution_result', {}) or {}
        objective_value = solution_result.get("objective_value")
        
        # Mandatory restart conditions: no solution or objective value equals zero
        must_restart_conditions = [
            not solution_result,  # No solution available
            objective_value == 0,  # Objective value equals zero
            objective_value is None,  # Objective value missing
            verification_status in ["failed", "error"],  # Verification failed
        ]
        
        if any(must_restart_conditions):
            self.logger.info(f"Mandatory restart conditions met: no_solution={not solution_result}, objective_value={objective_value}, status={verification_status}")
            return True
        
        # Quality score and status evaluation
        score_threshold = 0.8
        quality_restart_conditions = [
            quality_score < score_threshold,
            verification_status in ["needs_improvement", "failed"],
            should_iterate_hint and quality_score < 0.9
        ]
        
        should_restart = any(quality_restart_conditions)
        
        self.logger.info(f"Restart decision: quality_score={quality_score:.3f}, status={verification_status}, "
                        f"restart_recommended={should_iterate_hint}, decision={'restart' if should_restart else 'complete'}")
        
        return should_restart
    
    def _perform_final_analysis(self, verification_result: Dict[str, Any]) -> None:
        """Execute final result analysis (unified implementation, both Fixed and Free modes will go through this)"""
        try:
            # Basic final analysis
            quality_score = verification_result.get("quality_score", 0.0)
            verification_status = verification_result.get("verification_status", "unknown")
            solution_result = getattr(self.debug_summary, 'solution_result', {}) or {}
            
            # Generate final analysis report
            final_analysis = {
                "overall_assessment": self._generate_overall_assessment(quality_score, verification_status, solution_result),
                "solution_quality": f"Quality score: {quality_score:.3f}, Status: {verification_status}",
                "total_attempts": getattr(self, 'total_attempts', 0),
                "coordinator_mode": self.coordinator_mode,
                "has_valid_solution": bool(solution_result and solution_result.get("objective_value")),
                "analysis_timestamp": datetime.now().timestamp()
            }
            
            # Add final analysis to verification results
            enhanced_verification = verification_result.copy()
            enhanced_verification["final_analysis"] = final_analysis
            
            # Update verification results
            self.debug_summary.set_verification_result(enhanced_verification)
            
            self.logger.info(f"Final analysis completed: {final_analysis['overall_assessment']}")
            print(f"📊 Final analysis: {final_analysis['overall_assessment']}")
            
        except Exception as e:
            self.logger.error(f"Final analysis failed: {e}")
            # Does not affect main workflow, continue execution
    
    def _generate_overall_assessment(self, quality_score: float, verification_status: str, solution_result: Dict[str, Any]) -> str:
        """Generate overall assessment"""
        if not solution_result or not solution_result.get("objective_value"):
            return "Solving failed, no valid solution obtained"
        elif quality_score >= 0.8:
            return "Solution quality is excellent, can be used directly"
        elif quality_score >= 0.6:
            return "Solution quality is good, further verification recommended"
        elif quality_score >= 0.4:
            return "Solution quality is average, improvement needed"
        else:
            return "Solution quality is poor, recommend re-analyzing the problem"
    
    def _save_backup_results(self, verification_result: Dict[str, Any], reason: str = "Save current stage results as backup before restart") -> None:
        """Save backup results from current stage to solution_history"""
        try:
            current_solution = getattr(self.debug_summary, 'solution_result', {}) or {}
            current_analysis = getattr(self.debug_summary, 'analysis_result', {}) or {}
            current_model_code = getattr(self.debug_summary, 'final_model_code', "") or ""
            
            backup_entry = {
                "timestamp": datetime.now().timestamp(),
                "solution": current_solution,
                "analysis": current_analysis,
                "model_code": current_model_code,
                "verification": verification_result,
                "coordinator_mode": self.coordinator_mode,
                "total_attempts": getattr(self, 'total_attempts', 0),
                "is_backup": True,
                "backup_reason": reason
            }
            
            self.solution_history.append(backup_entry)
            self.logger.info(f"Backup results saved to solution_history (entry #{len(self.solution_history)})")
            
        except Exception as e:
            self.logger.error(f"Failed to save backup results: {e}")
    
    def handle_solution_verification(self, verification_result: Dict[str, Any], content: Any) -> None:
        """Unified solution verification processing entry point"""
        self.logger.info(f"{self.coordinator_mode} mode: Starting unified verification processing flow")
        
        # 1. Execute basic final analysis (shared by all modes)
        self._perform_final_analysis(verification_result)
        
        # 2. Delegate to subclass for mode-specific verification decision processing
        self._handle_verification_decision(verification_result, content)
    
    def _handle_verification_decision(self, verification_result: Dict[str, Any], content: Any) -> None:
        """Verification decision processing (subclasses override to implement different decision logic)"""
        # Base class default implementation: use unified rules, no restart, complete directly
        self.logger.info("Base class default processing: using unified restart rules for decision making")
        
        should_restart = self._make_restart_decision(verification_result)
        
        if should_restart and hasattr(self, 'total_attempts') and self.total_attempts < getattr(self, 'max_total_attempts', 120):
            self.logger.info("Base class decision: restart needed, but base class doesn't support restart, completing directly")
            self._complete_workflow(verification_result, content)
        else:
            self.logger.info("Base class decision: no restart needed, completing workflow")
            self._complete_workflow(verification_result, content)
    
    def _complete_workflow(self, verification_result: Dict[str, Any], content: Any) -> None:
        """Unified workflow completion processing"""
        # Complete workflow
        self.workflow_state = "completed"
        
        # Add final solution to history
        final_solution_entry = {
            "timestamp": datetime.now().timestamp(),
            "solution": self.debug_summary.solution_result if hasattr(self.debug_summary, 'solution_result') else {},
            "verification": verification_result,
            "coordinator_mode": self.coordinator_mode
        }
        
        # If total_attempts attribute exists, also record it in history
        if hasattr(self, 'total_attempts'):
            final_solution_entry["total_attempts_used"] = self.total_attempts
        
        self.solution_history.append(final_solution_entry)
        
        # Backfill the most recent success package with "continued: false" marker
        try:
            if len(self.solution_history) >= 2:  # Ensure at least two entries (second-to-last is success package)
                self.solution_history[-2]["continued_after_verification"] = False
                if hasattr(self.debug_summary, 'solution_history'):
                    self.debug_summary.solution_history = self.solution_history
        except Exception:
            pass
        
        print(f"✅ {self.coordinator_mode} mode: Solution verification and final evaluation completed, workflow ended")
        self.logger.info(f"{self.coordinator_mode} mode: Solution verification and final evaluation completed, preparing final report")
    
    def _handle_fixed_mode_completion(self, verification_result: Dict[str, Any], content: Any) -> None:
        """Basic Fixed mode completion processing: only used for backup verification and other special scenarios"""
        # This method is reserved for backup verification and other special scenarios, no restart decisions
        self._perform_final_analysis(verification_result)
        self._complete_workflow(verification_result, content)
    
    def _handle_free_mode_final_verification(self, verification_result: Dict[str, Any], solution_result: Dict[str, Any]) -> None:
        """Free mode verification handling after cycle ends: Base class default implementation (overridden in FreeCoordinator)"""
        # Base class provides default implementation, complete directly (no LLM decision analysis)
        self.workflow_state = "completed"
        
        # Add final solution to history
        self.solution_history.append({
            "timestamp": datetime.now().timestamp(),
            "solution": solution_result,
            "verification": verification_result,
            "coordinator_mode": self.coordinator_mode
        })

        # Fill in "continue: no" for most recent success package
        try:
            if self.solution_history:
                self.solution_history[-1]["continued_after_verification"] = False
                if hasattr(self.debug_summary, 'solution_history'):
                    self.debug_summary.solution_history = self.solution_history
        except Exception:
            pass
        
        print("✅ Base class default handling: Solution verification completed, workflow ended")

    def handle_error_message(self, message: Message) -> None:
        """Unified error handling entry: Handle different types of errors"""
        if message.sender == "solution_verifier":
            # solution_verifier error: Special handling (doesn't consume major round quota)
            self.handle_solution_verifier_error(message)
        elif self.is_solver_error(message):
            # solver error: Enter debug mechanism (implemented by subclass)
            self.handle_solver_error(message)
        else:
            # pre-solver error: Record + Direct RETRY
            self.handle_pre_solver_error(message)
    
    def handle_solution_verifier_error(self, message: Message) -> None:
        """Handle solution_verifier error: Retry without consuming major round quota"""
        error_content = message.content
        error_type = error_content.get("error_type", "VerificationError")
        retry_count = error_content.get("retry_count", 0)
        max_retries = error_content.get("max_retries", 3)
        verification_data = error_content.get("verification_data", {})
        
        self.logger.info(f"Handling solution_verifier error (Attempt {retry_count + 1}/{max_retries})")
        
        # Record error but don't count in major round statistics
        self.debug_summary.record_attempt({
            "agent": "solution_verifier",
            "error_type": error_type,
            "strategy": f"Retry verification ({retry_count + 1}/{max_retries})",
            "success": False,
            # "duration": 0.0,
            "coordinator_mode": self.coordinator_mode,
            "attempt_number": 0,  # Don't count in major rounds
            "error_details": str(error_content.get("error", "")),
            "fix_applied": "solution_verifier internal retry"
        })
        
        # Build retry message
        retry_msg = Message(
            "coordinator",
            "solution_verifier",
            {
                "strategy": "Retry solution verification",
                "verification_data": verification_data
            },
            MessageType.RETRY
        )
        
        # Send retry message
        response = self.agents["solution_verifier"].process_message(retry_msg)
        if response:
            self.pending_messages.append(response)
    
    def handle_solver_error(self, message: Message) -> None:
        """Handle solver error - Subclass implements specific debug logic"""
        raise NotImplementedError("Subclass must implement handle_solver_error method")
    
    def run(self, problem_description: str) -> Dict[str, Any]:
        """Run optimization problem solving workflow"""
        try:
            # Auto-select mode (if needed)
            if hasattr(self, 'auto_select_mode') and self.auto_select_mode:
                selected_mode = self.select_coordinator_mode(problem_description)
                if selected_mode != self.coordinator_mode:
                    self.logger.info(f"Switching coordinator mode: {self.coordinator_mode} -> {selected_mode}")
                    self.coordinator_mode = selected_mode
            else:
                # If auto_select_mode not set but mode is "auto", perform auto selection
                if self.coordinator_mode == "auto":
                    selected_mode = self.select_coordinator_mode(problem_description)
                    self.coordinator_mode = selected_mode
                    self.logger.info(f"Auto-selected coordinator mode: {selected_mode}")
            
            # Set problem basic information
            self.debug_summary.set_problem_info(problem_description, self.coordinator_mode, self.model_type)
            
            # Initialize JSON tracking
            if self.enable_json_tracking and self.json_tracking_manager:
                self.json_tracking_manager.problem_id = self.debug_summary.problem_id
                # Integrate JSON tracking with DebugSummary
                JSONTrackingIntegrator.integrate_with_debug_summary(
                    self.debug_summary, self.json_tracking_manager
                )
                # Integrate JSON tracking with Coordinator
                JSONTrackingIntegrator.integrate_with_coordinator(
                    self, self.json_tracking_manager
                )
            
            # Initialize workflow
            self.workflow_state = "analyzing"
            self.solution_history = []
            self.pending_messages = []
            self.original_problem = problem_description  # Store original problem description for retry use
            
            self.logger.info(f"Starting optimization problem solving workflow, mode: {self.coordinator_mode}")
            
            # Start problem analysis
            self.current_agent_source = "initial"  # Set as initial analysis
            analysis_msg = Message("coordinator", "analyzer", problem_description, MessageType.QUERY)
            response = self.agents["analyzer"].process_message(analysis_msg)
            if response:
                self.pending_messages.append(response)
            
            # Message processing loop
            max_iterations = 100  # Maximum iteration limit
            iteration = 0
            
            # ✅ Add loop monitoring logs
            self.logger.info(f"Starting message processing loop, maximum iterations: {max_iterations}")
            if hasattr(self, 'max_total_attempts'):
                self.logger.info(f"{self.coordinator_mode} mode maximum attempts: {self.max_total_attempts}")
            
            while self.pending_messages and iteration < max_iterations and self.workflow_state not in ["completed", "failed"]:
                iteration += 1
                self.logger.info(f"Processing iteration {iteration}, pending messages: {len(self.pending_messages)}")
                
                # Process all pending messages
                current_messages = self.pending_messages.copy()
                self.pending_messages.clear()
                
                for message in current_messages:
                    self.logger.info(f"Processing message: {message.sender} -> {message.receiver}")
                    try:
                        self.process_message(message)
                    except Exception as e:
                        # Catch and handle exceptions in message processing
                        error_msg = Message(
                            message.receiver,
                            "coordinator",
                            {
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                                "original_message": message.to_dict()
                            },
                            MessageType.ERROR
                        )
                        self.pending_messages.append(error_msg)
                
                # Brief delay to avoid too rapid processing
                time.sleep(0.1)
            
            # Backup solution verification (only called at loop end and if no verification done)
            # Note: Normally solution verification is called immediately after solver success
            # CRITICAL: Do not run backup verification if workflow completed due to max attempts reached
            has_solver_success = hasattr(self.debug_summary, 'solution_result') and self.debug_summary.solution_result
            max_attempts_reached = (hasattr(self, 'total_attempts') and hasattr(self, 'max_total_attempts') and 
                                   self.total_attempts >= self.max_total_attempts)
            
            if (self.workflow_state not in ["completed", "failed"] and 
                not max_attempts_reached and
                has_solver_success and
                (not hasattr(self.debug_summary, 'verification_result') or 
                 not self.debug_summary.verification_result)):
                
                print("🔍 Loop ended, starting backup solution verification...")
                self.logger.info("Loop ended, performing backup solution verification (safeguard mechanism when normal flow not executed)")
                
                try:
                    # Prepare verification data (even without solution result)
                    solution_result = getattr(self.debug_summary, 'solution_result', {}) or {}
                    verification_data = self._prepare_verification_data(solution_result)
                    verification_msg = Message("coordinator", "solution_verifier", verification_data, MessageType.QUERY)
                    response = self.agents["solution_verifier"].process_message(verification_msg)
                    
                    if response and response.msg_type == MessageType.RESPONSE:
                        # Process verification result directly, don't enter message loop
                        self.debug_summary.set_verification_result(response.content)
                        print("✅ Backup solution verification completed")
                        
                        # Handle based on coordinator mode
                        if self.coordinator_mode == "free":
                            # Free mode: Only perform LLM decision analysis (don't restart after loop end)
                            self._handle_free_mode_final_verification(response.content, solution_result)
                        else:
                            # Fixed mode: Backup verification (no restart)
                            self._handle_fixed_mode_completion(response.content, response.content)
                        
                    elif response and response.msg_type == MessageType.ERROR:
                        self.logger.warning(f"Solution verification failed: {response.content}")
                        # Set default verification result even if verification fails
                        fallback_verification = {
                            "verification_status": "needs_improvement",
                            "quality_score": 0.5,  # Default score for simulation result
                            "analysis": {
                                "solution_feasibility": "Verification process failed, cannot fully analyze",
                                "model_consistency": "Verification failed, model consistency not verified",
                                "result_reasonableness": "Verification failed, reasonableness needs manual check",
                                "optimization_quality": "Verification failed, optimization quality needs further verification"
                            },
                            "recommendations": ["Suggest manual check of solution quality"],
                            "confidence": 0.3,
                            "should_iterate": False,
                            "error_occurred": True
                        }
                        self.debug_summary.set_verification_result(fallback_verification)
                        self.workflow_state = "completed"
                        
                except Exception as e:
                    self.logger.error(f"Final solution verification error: {e}")
                    # Set error status verification result
                    error_verification = {
                        "verification_status": "failed",
                        "quality_score": 0.0,
                        "analysis": {
                            "solution_feasibility": f"Verification failed: {str(e)}",
                            "model_consistency": "Cannot verify",
                            "result_reasonableness": "Cannot verify", 
                            "optimization_quality": "Cannot verify"
                        },
                        "recommendations": ["Suggest rerun or manual verification"],
                        "confidence": 0.0,
                        "should_iterate": False,
                        "verification_error": str(e)
                    }
                    self.debug_summary.set_verification_result(error_verification)
                    self.workflow_state = "completed"
            
            # Save JSON tracking data
            if self.enable_json_tracking and self.json_tracking_manager:
                try:
                    self.json_tracking_manager.save_tracking_data()
                    self.json_tracking_manager.generate_analysis_markdown()
                    stats = self.json_tracking_manager.get_statistics()
                    self.logger.info(f"JSON tracking statistics: {stats}")
                except Exception as e:
                    self.logger.error(f"Failed to save JSON tracking data: {e}")
            
            # Generate debug summary and CSV record
            self.debug_summary.save_summary_to_file()
            self._save_to_csv()
            
            # Return final result
            final_result = {
                "status": self.workflow_state,
                "coordinator_mode": self.coordinator_mode,
                "solution_history": self.solution_history,
                "debug_records_count": len(self.debug_summary.debug_records),
                "iterations": iteration
            }
            
            if self.solution_history:
                final_solution = self.solution_history[-1]["solution"]
                final_result["final_solution"] = final_solution
                
                # Enhanced result display: Format solution output
                if isinstance(final_solution, dict):
                    print("\n" + "="*80)
                    print("🎯 Final Solution")
                    print("="*80)
                    
                    # Display solving status
                    status = final_solution.get("status", "unknown")
                    print(f"📊 Solving status: {status}")
                    
                    # Display objective function value
                    objective_value = final_solution.get("objective_value")
                    if objective_value is not None:
                        print(f"🎯 Objective function value: {objective_value:,.2f}")
                    
                    # Display solver used
                    solver_used = final_solution.get("solver_used", "unknown")
                    print(f"🔧 Solver: {solver_used}")
                    
                    # Display decision variable values
                    decision_vars = final_solution.get("decision_variables_values", {})
                    if decision_vars:
                        print(f"\n📋 Decision variable values ({len(decision_vars)} variables):")
                        print("-" * 60)
                        
                        # Intelligently group and sort variables
                        var_groups = {}
                        for var_name, var_value in decision_vars.items():
                            # Group by variable name base
                            base_name = var_name.split('_')[0] if '_' in var_name else var_name
                            if base_name not in var_groups:
                                var_groups[base_name] = []
                            var_groups[base_name].append((var_name, var_value))
                        
                        # Display variables by group
                        for group_name, variables in sorted(var_groups.items()):
                            if len(variables) > 1:
                                print(f"\n📊 {group_name.upper()} Series Variables:")
                                for var_name, var_value in sorted(variables):
                                    if isinstance(var_value, (int, float)) and var_value != 0:
                                        print(f"   {var_name:20s}: {var_value:>12.2f}")
                                    elif var_value == 0:
                                        print(f"   {var_name:20s}: {var_value:>12.0f}")
                                    else:
                                        print(f"   {var_name:20s}: {str(var_value):>12s}")
                            else:
                                var_name, var_value = variables[0]
                                if isinstance(var_value, (int, float)) and var_value != 0:
                                    print(f"📊 {var_name:25s}: {var_value:>12.2f}")
                                elif var_value == 0:
                                    print(f"📊 {var_name:25s}: {var_value:>12.0f}")
                                else:
                                    print(f"📊 {var_name:25s}: {str(var_value):>12s}")
                    
                    # Display termination condition
                    termination = final_solution.get("termination_condition")
                    if termination:
                        print(f"\n🏁 Termination condition: {termination}")
                    
                    print("="*80)
            
            self.logger.info(f"Workflow completed, status: {self.workflow_state}")
            
            # Save debug summary to file
            try:
                self.debug_summary.save_summary_to_file()
                final_result["debug_file"] = self.debug_summary.summary_file
                self.logger.info(f"Debug report saved: {self.debug_summary.summary_file}")
            except Exception as e:
                self.logger.error(f"Failed to save debug report: {e}")
                
            return final_result
            
        except Exception as e:
            error_trace = traceback.format_exc()
            self.logger.error(f"Workflow execution exception: {e}\n{error_trace}")
            self.handle_error(e, {"problem_description": problem_description})
            
            # Save debug summary even on exception
            try:
                self.debug_summary.save_summary_to_file()
                debug_file = self.debug_summary.summary_file
                self.logger.info(f"Debug report saved on exception: {debug_file}")
            except Exception as save_error:
                self.logger.error(f"Failed to save debug report: {save_error}")
                debug_file = None
                
            return {
                "status": "failed", 
                "error": str(e),
                "traceback": error_trace,
                "coordinator_mode": self.coordinator_mode,
                "debug_file": debug_file
            }

class FixedCoordinator(BaseCoordinator):
    """Fixed mode coordinator - rule-based"""
    
    def __init__(self, agent_id: str = "fixed_coordinator", enable_json_tracking: bool = False, model_type: str = "gemini", problem_id: str = None):
        super().__init__(agent_id, mode="fixed", enable_json_tracking=enable_json_tracking, model_type=model_type, problem_id=problem_id)
        
        # Fixed mode specific configuration
        self.max_total_attempts = 100  # Total attempt limit consistent with Free mode
        self.total_attempts = 0
        self.max_modeling_retries = 3   # Fixed 3 modeler attempts after each analyzer
        self.modeling_attempts = 0
        
        self.logger.info("FixedCoordinator initialization completed")
    
    def _execute_fixed_debug_cycle(self, error_info: Dict[str, Any], error_type: str, start_time: float):
        """Execute fixed mode debugging mechanism: infinite loop - analyzer followed by 3 modeler attempts"""
        
        self.logger.info(f"Solver ERROR received, modeling_attempts: {self.modeling_attempts}, total_attempts: {self.total_attempts}")

        if self.modeling_attempts >= self.max_modeling_retries:
            # Modeling retries reached limit, reset modeling_attempts=0, restart with analyzer
            self.modeling_attempts = 0
            self.logger.info(f"Modeling retries reached limit, reset to 0, restart with analyzer (infinite loop logic)")
            
            # Directly start new analyzer round (infinite loop)
            self._execute_analyzer_retry(error_info, error_type, start_time)
            return
        
        else:
            # Continue Modeling retry
            self.logger.info(f"Continue modeling retry, current attempts: {self.modeling_attempts}/{self.max_modeling_retries}")
            self._execute_modeling_retry(error_info, error_type, start_time)
            self.modeling_attempts += 1
            return
    
    def _execute_analyzer_retry(self, error_info: Dict[str, Any], error_type: str, start_time: float):
        """Execute Analyzer retry"""
        strategy = f"Analyzer re-analysis (current attempt {self.total_attempts})"
        self.logger.info(f"Execute analyzer retry: {strategy}")
        
        # Send retry message to AnalyzerAgent - add necessary context information
        retry_info = {
            "original_problem": self.original_problem,
            "error_info": error_info,
            "strategy": strategy,
            # ✅ Fix: add missing context information
            "previous_analysis": getattr(self.debug_summary, 'analysis_result', {}),
            "rejected_model_code": getattr(self.debug_summary, 'final_model_code', "")
        }
        retry_msg = Message("coordinator", "analyzer", retry_info, MessageType.RETRY)
        
        # Record start time, calculate duration after agent processing completes
        self.current_agent_source = "solver"  # Set source as solver error
        retry_start_time = time.time()
        response = self.agents["analyzer"].process_message(retry_msg)
        processing_duration = time.time() - retry_start_time
        
        # Record analyzer retry attempt
        attempt_info = {
            "agent": "analyzer",
            "error_type": error_type,
            "strategy": strategy,
            "success": response and response.msg_type == MessageType.RESPONSE,
            "duration": processing_duration,
            "coordinator_mode": self.coordinator_mode,
            "attempt_number": self.total_attempts,
            "error_details": str(error_info.get("error", ""))[:200],
            "fix_applied": f"solver ERROR -> analyzer re-analysis"
        }
        self.debug_summary.record_attempt(attempt_info)
        
        if response:
            self.pending_messages.append(response)
    
    def _execute_modeling_retry(self, error_info: Dict[str, Any], error_type: str, start_time: float):
        """Execute Modeling retry"""
        strategy = f"Modeling retry #{self.modeling_attempts} (current attempt {self.total_attempts})"
        self.logger.info(f"Execute modeling retry: {strategy}")
        
        # Send retry message to ModelingAgent - add necessary context information
        retry_info = {
            "error_info": error_info,
            "strategy": strategy,
            "attempt_number": self.modeling_attempts,
            # ✅ Fix: add missing context information
            "analysis_result": getattr(self.debug_summary, 'analysis_result', {}),
            "previous_model_code": getattr(self.debug_summary, 'final_model_code', ""),
            "original_problem": self.original_problem
        }
        retry_msg = Message("coordinator", "modeler", retry_info, MessageType.RETRY)
        
        # Record start time, calculate duration after agent processing completes
        retry_start_time = time.time()
        response = self.agents["modeler"].process_message(retry_msg)
        processing_duration = time.time() - retry_start_time
        
        # Record modeling retry attempt
        attempt_info = {
            "agent": "modeler",
            "error_type": error_type,
            "strategy": strategy,
            "success": response and response.msg_type == MessageType.RESPONSE,
            "duration": processing_duration,
            "coordinator_mode": self.coordinator_mode,
            "attempt_number": self.total_attempts,
            "error_details": str(error_info.get("error", ""))[:200],
            "fix_applied": f"solver ERROR -> modeling retry #{self.modeling_attempts}"
        }
        self.debug_summary.record_attempt(attempt_info)
        
        if response:
            self.pending_messages.append(response)

    def handle_solver_error(self, message: Message) -> None:
        """Fixed mode solver error handling - rule-based"""
        error_info = message.content
        start_time = time.time()
        
        # Error classification
        error_classification = self.error_classifier.classify_error(error_info)
        error_type = error_classification["error_type"]
        
        if not self.in_debug_mode:
            # First time entering debug mode
            self.in_debug_mode = True
            self.logger.info(f"Received solver error: {error_type} from {message.sender}, first time entering debug mode")
        else:
            # Solver error in debug mode
            self.logger.info(f"Received solver error in debug mode: {error_type} (total attempts: {self.total_attempts})")
        
        # Fixed mode also counts total attempts
        self.total_attempts += 1
        # self.consecutive_failures += 1
        
        # Check if should stop trying
        if self.total_attempts >= self.max_total_attempts:
            self.logger.warning("Reached maximum attempts, falling back to simulation")
            self.workflow_state = "completed"
            self.in_debug_mode = False
            return
        
        self._execute_fixed_debug_cycle(error_info, error_type, start_time)
    
    def handle_response_message(self, message: Message) -> None:
        """Override response handling to support fixed mode special logic"""
        sender = message.sender
        
        # FixedCoordinator follows infinite loop logic:
        # - After analyzer success, don't reset modeling_attempts, continue maintaining current count state
        # - Only reset counters when solver succeeds
        if sender == "solver":
            self._reset_retry_counters()
        
        # Call parent class method to handle standard sequential flow
        super().handle_response_message(message)    

    def _reset_retry_counters(self):
        """Reset all retry counters"""
        self.modeling_attempts = 0
        # Note: Don't reset total_attempts, it maintains accumulation throughout the problem-solving process
    
    def _handle_verification_decision(self, verification_result: Dict[str, Any], content: Any) -> None:
        """Fixed mode verification decision: use unified rules for restart decision"""
        self.logger.info("Fixed mode: using unified restart mechanism for decision making")
        
        # Use unified restart decision
        should_restart = self._make_restart_decision(verification_result)
        
        if should_restart and self.total_attempts < self.max_total_attempts:
            self.logger.info("Decided to restart optimization process, restart from analyzer")
            self._restart_from_analyzer(verification_result)
        else:
            # Don't restart, complete workflow
            if self.total_attempts >= self.max_total_attempts:
                self.logger.info(f"Reached maximum attempts ({self.max_total_attempts}), complete workflow")
            else:
                self.logger.info("Verification result satisfactory, complete workflow")
            
            self._complete_workflow(verification_result, content)
    
    def _restart_from_analyzer(self, verification_result: Dict[str, Any]) -> None:
        """Restart optimization process from analyzer (Fixed mode fixed from analyzer restart)"""
        # Save current stage backup results
        self._save_backup_results(verification_result, "Fixed mode restart before saving current stage results as backup")
        
        # Reset debug-related counters, restart "analyzer+3modeler" loop
        self.modeling_attempts = 0
        self.in_debug_mode = True  # Re-enter debug mode, start new loop
        
        # Reset workflow state to analysis stage, restart entire process
        self.workflow_state = "analyzing"
        
        # Record restart event
        self.debug_summary.record_attempt({
            "agent": "coordinator",
            "error_type": "verification_restart",
            "strategy": "Fixed mode restart from analyzer based on verification result",
            "success": True,
            "coordinator_mode": self.coordinator_mode,
            "attempt_number": self.total_attempts,
            "error_details": f"Quality score: {verification_result.get('quality_score', 0.0)}",
            "fix_applied": "Fixed mode restart from analyzer, re-enter analyzer+3modeler cycle"
        })
        
        # Restart from analysis stage
        self.current_agent_source = "solution verifier"  # Set source as solution verifier (Solution Verification restart)
        analysis_msg = Message("coordinator", "analyzer", self.original_problem, MessageType.QUERY)
        response = self.agents["analyzer"].process_message(analysis_msg)
        if response:
            self.pending_messages.append(response)
        
        print(f"🔄 Fixed mode: restart from analyzer based on verification result, re-enter analyzer+3modeler loop")


class FreeCoordinator(BaseCoordinator):
    """Free mode coordinator - experience-driven + LLM decision"""
    
    def __init__(self, agent_id: str = "free_coordinator", enable_json_tracking: bool = False, model_type: str = "gemini", problem_id: str = None):
        super().__init__(agent_id, mode="free", enable_json_tracking=enable_json_tracking, model_type=model_type, problem_id=problem_id)
        
        # Initialize LLM debug router
        try:
            self.debug_router = LLMDebugRouter(api_available=True, model_type=self.model_type)
        except Exception as e:
            self.logger.warning(f"LLM router initialization failed, using rule-based routing: {e}")
            self.debug_router = LLMDebugRouter(api_available=False, model_type=self.model_type)
        
        # Free mode specific configuration
        self.max_total_attempts = 100  # ✅ Increase total attempts limit for better coordination with loop limit
        self.total_attempts = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10  # ✅ Appropriately increase consecutive failures limit
        
        self.logger.info("FreeCoordinator initialization completed")
    
    def handle_solver_error(self, message: Message) -> None:
        """Free mode solver error handling - LLM intelligent decision"""
        error_info = message.content
        start_time = time.time()
        
        # Error classification
        error_classification = self.error_classifier.classify_error(error_info)
        error_type = error_classification["error_type"]
        
        if not self.in_debug_mode:
            # First time entering debug mode
            self.in_debug_mode = True
            self.logger.info(f"Received solver error: {error_type} from {message.sender}, first time entering debug mode")
        else:
            # Solver error in debug mode
            self.logger.info(f"Received solver error in debug mode: {error_type} (total attempts: {self.total_attempts})")
        
        # Fixed mode also counts total attempts
        self.total_attempts += 1
        self.consecutive_failures += 1
        
        # Check if should stop trying
        if self.total_attempts >= self.max_total_attempts:
            self.logger.warning("Reached maximum attempts, falling back to simulation")
            self.workflow_state = "completed"
            self.in_debug_mode = False
            return
        
        # Use LLM router for decision
        current_success_rates = self.get_current_success_rates()
        routing_decision = self.debug_router.route_debug_request(
            error_info,
            self.debug_summary.debug_records,
            current_success_rates
        )
        
        self.logger.info(f"LLM routing decision: {routing_decision}")
        
        # Check if should continue trying
        if not routing_decision.get("should_continue", True):
            self.logger.info("LLM suggests stopping attempts, falling back to simulation")
            self._fallback_to_simulation(error_info, start_time)
            return
        
        # Execute routing decision
        target_agent = routing_decision.get("target_agent", "modeler")
        strategy = routing_decision.get("strategy", "general_fix")
        
        self._send_intelligent_retry(target_agent, error_info, strategy, routing_decision, start_time)
    
    def _send_intelligent_retry(self, target_agent: str, error_info: Dict[str, Any], 
                               strategy: str, routing_decision: Dict[str, Any], start_time: float):
        """Send intelligent retry message"""
        
        retry_content = {
            "action": "intelligent_retry",
            "error_info": error_info,
            "strategy": strategy,
            "routing_decision": routing_decision,
            "attempt_number": self.total_attempts,
            "success_rates": self.get_current_success_rates()
        }
        
        # ✅ Fix: Add necessary context for all agents
        if target_agent == "analyzer":
            retry_content["original_problem"] = self.original_problem
            retry_content["previous_analysis"] = getattr(self.debug_summary, 'analysis_result', {})
            retry_content["rejected_model_code"] = getattr(self.debug_summary, 'final_model_code', "")
        elif target_agent == "modeler":
            retry_content["analysis_result"] = getattr(self.debug_summary, 'analysis_result', {})
            retry_content["previous_model_code"] = getattr(self.debug_summary, 'final_model_code', "")
            retry_content["original_problem"] = self.original_problem
        
        # Embed LLM's evaluation reason into error_info for downstream prompts
        try:
            if isinstance(retry_content.get("error_info"), dict):
                retry_content["error_info"]["llm_strategy"] = strategy
        except Exception:
            pass

        retry_msg = Message(
            "coordinator",
            target_agent,
            retry_content,
            MessageType.RETRY
        )
        self.pending_messages.append(retry_msg)
        
        # Record debug attempt
        self.debug_summary.record_attempt({
            "agent": target_agent,
            "error_type": self.error_classifier.classify_error(error_info)["error_type"],
            "strategy": strategy,
            "success": False,
            "duration": time.time() - start_time,
            "coordinator_mode": self.coordinator_mode,
            "attempt_number": self.total_attempts,
            "error_details": str(error_info.get("error", ""))[:200],
            "fix_applied": f"LLM routing: {strategy}"
        })
    
    def _handle_verification_decision(self, verification_result: Dict[str, Any], content: Any) -> None:
        """Free mode verification handling: make decision based on verification result"""
        self.logger.info("Free mode: Making intelligent decision based on verification result")
        
        # Check if should restart iteration
        should_restart = self._make_restart_decision(verification_result)

        if should_restart:
            should_restart = self._make_llm_iteration_decision(verification_result)
        
        if should_restart and self.total_attempts < self.max_total_attempts:
            self.logger.info("Decided to restart iteration, continuing solution optimization")
            self._restart_with_llm_agent_selection(verification_result)
        else:
            # Don't restart, but still perform LLM decision analysis
            self.logger.info("Decided not to restart, performing final decision analysis")
            self._perform_llm_final_analysis(verification_result)
            self._complete_workflow(verification_result, content)
    
    def _perform_llm_final_analysis(self, verification_result: Dict[str, Any]) -> None:
        """Perform final LLM decision analysis and update verification result"""
        try:
            decision_analysis = self._generate_final_decision_analysis(verification_result)
            
            # Add decision analysis to verification result
            enhanced_verification = verification_result.copy()
            enhanced_verification["llm_decision_analysis"] = decision_analysis
            
            # Update verification result
            self.debug_summary.set_verification_result(enhanced_verification)
            
            print("🧠 Free mode: LLM final decision analysis completed")
            self.logger.info(f"LLM final decision analysis: {decision_analysis.get('conclusion', 'No conclusion')}")
            
        except Exception as e:
            self.logger.error(f"LLM final decision analysis failed: {e}")
            # Keep original verification result unchanged
    
    def _make_llm_iteration_decision(self, verification_result: Dict[str, Any]) -> bool:
        """Use LLM to decide whether to restart iteration based on verification result"""
        
        # Basic condition check
        quality_score = verification_result.get("quality_score", 0.0)
        verification_status = verification_result.get("verification_status", "unknown")
        should_iterate_hint = verification_result.get("should_iterate", False)
        
        # Use LLM for intelligent decision
        try:
            from specialized_agents import call_llm_api
            
            # Build decision prompt
            prompt = f"""You are a decision expert for an intelligent optimization system. Based on solution verification results, determine if optimization iteration should be restarted.

**Verification Result Analysis:**
- Quality Score: {quality_score:.3f} (0-1 range)
- Verification Status: {verification_status}
- Verification Suggests Iteration: {should_iterate_hint}
- Improvement Suggestions: {verification_result.get('recommendations', [])}

**Current System Status:**
- Current Iteration: {self.total_attempts}
- Maximum Iterations: {self.max_total_attempts}
- Consecutive Failures: {self.consecutive_failures}

**Decision Criteria:**
- If quality score >= 0.8 and status is satisfactory, usually no need to restart
- If quality score < 0.5 or status is failed, usually need to restart
- If quality score is between 0.5-0.8, need to consider improvement suggestions and remaining attempts

Please make a decision based on the above information, return in JSON format:
```json
{{
    "should_restart": true/false,
    "reason": "Detailed decision reason",
    "target_agent": "Suggested agent to focus on (analyzer/modeler)",
    "strategy": "Specific optimization strategy suggestion"
}}
```

**Decision Principles:**
- Prioritize substantial improvement potential in solution quality
- Balance iteration cost and quality improvement benefits
- Avoid meaningless repeated iterations"""

            self.logger.info("Using LLM for restart decision...")
            response_text = call_llm_api(prompt, self.model_type)
            
            if response_text:
                # Parse LLM response
                decision = self._parse_llm_decision(response_text)
                should_restart = decision.get("should_restart", False)
                reason = decision.get("reason", "LLM decision")
                
                self.logger.info(f"LLM decision result: {'restart' if should_restart else 'no restart'} - {reason}")
                
                # Record decision process
                self.debug_summary.record_attempt({
                    "agent": "llm_decision",
                    "error_type": "decision_making",
                    "strategy": f"LLM intelligent decision: {reason}",
                    "success": True,
                    "coordinator_mode": self.coordinator_mode,
                    "attempt_number": self.total_attempts,
                    "error_details": f"Quality score: {quality_score}, Status: {verification_status}",
                    "fix_applied": f"Decision result: {'restart' if should_restart else 'complete'}"
                })
                
                return should_restart
            else:
                self.logger.warning("LLM decision failed, using default strategy")
                
        except Exception as e:
            self.logger.error(f"LLM decision process error: {e}")
        
        # Default strategy when LLM decision fails
        default_decision = (quality_score < 0.8 and should_iterate_hint and 
                          self.total_attempts < self.max_total_attempts * 0.8)
        
        self.logger.info(f"Using default decision strategy: {'restart' if default_decision else 'no restart'}")
        return default_decision
    
    def _parse_llm_decision(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM decision response"""
        try:
            import json
            import re
            
            # Extract JSON block
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                if json_end != -1:
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text[json_start:].strip()
            elif '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                return {"should_restart": False, "reason": "Cannot parse LLM response"}
            
            # Clean and parse JSON
            json_text = json_text.replace('```', '').strip()
            decision = json.loads(json_text)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Parse LLM decision failed: {e}")
            return {"should_restart": False, "reason": f"Parse failed: {e}"}
    
    def _restart_with_llm_agent_selection(self, verification_result: Dict[str, Any]) -> None:
        """Restart iteration process: intelligently select restart point and save current backup results"""
        self.logger.info("🔄 Restarting optimization iteration")
        
        # Step 1: Save current stage backup results to solution_history
        self._save_backup_results(verification_result)
        
        # Step 2: LLM intelligent restart point selection
        restart_agent = self._select_restart_agent(verification_result)
        
        # Step 3: Reset workflow state but keep history
        if restart_agent == "analyzer":
            self.workflow_state = "analyzing"
        elif restart_agent == "modeler":
            self.workflow_state = "modeling"
        else:
            self.workflow_state = "analyzing"  # Default to start from analysis
        
        self.consecutive_failures = 0
        
        # Record restart event
        self.debug_summary.record_attempt({
            "agent": "coordinator",
            "error_type": "restart_iteration",
            "strategy": f"Restart optimization process based on verification result, starting from {restart_agent}",
            "success": True,
            "coordinator_mode": self.coordinator_mode,
            "attempt_number": self.total_attempts,
            "error_details": f"Quality score: {verification_result.get('quality_score', 0.0)}",
            "fix_applied": f"Intelligently selected to restart from {restart_agent}"
        })
        
        # Step 4: Restart based on selected agent
        if restart_agent == "analyzer":
            # Restart from analysis stage
            self.current_agent_source = "solution verifier"  # Set source as solution verifier (Solution Verification restart)
            analysis_msg = Message("coordinator", "analyzer", self.original_problem, MessageType.QUERY)
            response = self.agents["analyzer"].process_message(analysis_msg)
            if response:
                self.pending_messages.append(response)
            print("🔄 Free mode: Restarting optimization iteration from analysis stage")
        elif restart_agent == "modeler":
            # Restart from modeling stage (needs analysis result)
            if hasattr(self.debug_summary, 'analysis_result') and self.debug_summary.analysis_result:
                modeling_msg = Message("coordinator", "modeler", self.debug_summary.analysis_result, MessageType.QUERY)
                response = self.agents["modeler"].process_message(modeling_msg)
                if response:
                    self.pending_messages.append(response)
                print("🔄 Free mode: Restarting optimization iteration from modeling stage")
            else:
                # No analysis result, fall back to start from analysis
                self.logger.warning("Missing analysis result, falling back to restart from analysis stage")
                self.workflow_state = "analyzing"
                self.current_agent_source = "solution verifier"  # Set source as solution verifier (fallback due to missing analysis)
                analysis_msg = Message("coordinator", "analyzer", self.original_problem, MessageType.QUERY)
                response = self.agents["analyzer"].process_message(analysis_msg)
                if response:
                    self.pending_messages.append(response)
                print("🔄 Free mode: Falling back to restart from analysis stage")
    
    def _save_backup_results(self, verification_result: Dict[str, Any]) -> None:
        """Save current stage backup results to solution_history"""
        try:
            # Get all results from current stage
            current_solution = getattr(self.debug_summary, 'solution_result', {}) or {}
            current_analysis = getattr(self.debug_summary, 'analysis_result', {}) or {}
            current_model_code = getattr(self.debug_summary, 'final_model_code', "") or ""
            
            # Build backup result record
            backup_entry = {
                "timestamp": datetime.now().timestamp(),
                "solution": current_solution,
                "analysis": current_analysis,
                "model_code": current_model_code,
                "verification": verification_result,
                "coordinator_mode": self.coordinator_mode,
                "total_attempts": self.total_attempts,
                "is_backup": True,  # Mark as backup result
                "backup_reason": "Free mode saving current stage results as backup before restart"
            }
            
            # Add to solution_history
            self.solution_history.append(backup_entry)
            
            self.logger.info(f"Saved current stage backup results to solution_history (Entry #{len(self.solution_history)})")
            print("💾 Free mode: Current stage results saved as backup")
            
        except Exception as e:
            self.logger.error(f"Save backup results failed: {e}")
    
    def _select_restart_agent(self, verification_result: Dict[str, Any]) -> str:
        """LLM intelligent restart point selection"""
        try:
            from specialized_agents import call_llm_api
            
            # Build selection prompt
            quality_score = verification_result.get("quality_score", 0.0)
            analysis_quality = verification_result.get("analysis", {})
            recommendations = verification_result.get("recommendations", [])
            
            prompt = f"""You are a restart strategy expert for an intelligent optimization system. Based on solution verification results, decide which stage to restart the optimization process from.

**Verification Result Analysis:**
- Quality Score: {quality_score:.3f} (0-1 range)
- Solution Feasibility: {analysis_quality.get('solution_feasibility', 'Unknown')}
- Model Consistency: {analysis_quality.get('model_consistency', 'Unknown')}
- Result Reasonableness: {analysis_quality.get('result_reasonableness', 'Unknown')}
- Optimization Quality: {analysis_quality.get('optimization_quality', 'Unknown')}
- Improvement Suggestions: {recommendations}

**Current System Status:**
- Current Iteration: {self.total_attempts}
- Maximum Iterations: {self.max_total_attempts}
- Workflow State: {self.workflow_state}

**Restart Selection Criteria:**
1. **Restart from analyzer**: If problem analysis is incorrect, constraint understanding is wrong, variable definition is incorrect
2. **Restart from modeler**: If analysis is correct but modeling implementation has issues, code logic errors, solver configuration is inappropriate

**Decision Requirements:**
Please select the most appropriate restart point based on verification results, and explain the reason.

**Output Format:**
```json
{{
    "restart_agent": "analyzer" or "modeler",
    "reason": "Detailed explanation for the selection",
    "confidence": 0.0-1.0 confidence level
}}
```"""
            
            response_text = call_llm_api(prompt, self.model_type)
            if response_text:
                decision = self._parse_restart_decision(response_text)
                restart_agent = decision.get("restart_agent", "analyzer")
                reason = decision.get("reason", "Default selection")
                confidence = decision.get("confidence", 0.5)
                
                self.logger.info(f"LLM restart point selection: {restart_agent} (confidence: {confidence:.3f})")
                self.logger.info(f"Selection reason: {reason}")
                
                return restart_agent
            else:
                self.logger.warning("LLM API unavailable, using default restart strategy")
                return self._get_default_restart_agent(verification_result)
                
        except Exception as e:
            self.logger.error(f"LLM restart selection failed: {e}")
            return self._get_default_restart_agent(verification_result)
    
    def _parse_restart_decision(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM restart decision response"""
        try:
            import json
            import re
            
            # Extract JSON block
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                if json_end != -1:
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text[json_start:].strip()
            elif '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                return {"restart_agent": "analyzer", "reason": "Cannot parse LLM response"}
            
            # Clean and parse JSON
            json_text = json_text.replace('```', '').strip()
            decision = json.loads(json_text)
            
            # Validate restart_agent value
            if decision.get("restart_agent") not in ["analyzer", "modeler"]:
                decision["restart_agent"] = "analyzer"
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Parse LLM restart decision failed: {e}")
            return {"restart_agent": "analyzer", "reason": f"Parse failed: {e}"}
    
    def _get_default_restart_agent(self, verification_result: Dict[str, Any]) -> str:
        """Default restart strategy (when LLM is unavailable)"""
        quality_score = verification_result.get("quality_score", 0.0)
        
        # Simple strategy based on quality score
        if quality_score < 0.3:
            # Very poor quality, might be analysis issue, restart from analyzer
            return "analyzer"
        elif quality_score < 0.6:
            # Medium quality, might be modeling issue, restart from modeler
            return "modeler"
        else:
            # Good quality but still needs improvement, restart from modeler (fine-tuning)
            return "modeler"
    
    def _handle_free_mode_final_verification(self, verification_result: Dict[str, Any], solution_result: Dict[str, Any]) -> None:
        """Free mode verification handling after loop ends: only perform LLM decision analysis, no restart"""
        self.logger.info("Free mode: Performing final verification decision analysis after loop ends (no restart)")
        
        # Perform LLM decision analysis (but don't restart iteration, only generate decision analysis)
        try:
            decision_analysis = self._generate_final_decision_analysis(verification_result)
            
            # Add decision analysis to verification result
            enhanced_verification = verification_result.copy()
            enhanced_verification["llm_decision_analysis"] = decision_analysis
            
            # Update verification result
            self.debug_summary.set_verification_result(enhanced_verification)
            
            print("🧠 Free mode: LLM final decision analysis completed")
            self.logger.info(f"LLM final decision analysis: {decision_analysis.get('conclusion', 'No conclusion')}")
            
        except Exception as e:
            self.logger.error(f"LLM final decision analysis failed: {e}")
            # Keep original verification result unchanged
        
        # Complete workflow
        self.workflow_state = "completed"
        
        # Add final solution to history
        self.solution_history.append({
            "timestamp": datetime.now().timestamp(),
            "solution": solution_result,
            "verification": self.debug_summary.verification_result,
            "coordinator_mode": self.coordinator_mode
        })
        
        print("✅ Free mode: Solution verification and decision analysis completed, workflow ended")
    
    def _generate_final_decision_analysis(self, verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final LLM decision analysis"""
        try:
            from specialized_agents import call_llm_api
            
            quality_score = verification_result.get("quality_score", 0.0)
            verification_status = verification_result.get("verification_status", "unknown")
            recommendations = verification_result.get("recommendations", [])
            
            prompt = f"""You are a final decision analysis expert for an intelligent optimization system. Based on solution verification results, summarize and analyze the entire optimization process.

**Verification Result Summary:**
- Quality Score: {quality_score:.3f} (0-1 range)
- Verification Status: {verification_status}
- Improvement Suggestions: {recommendations}

**System Run Status:**
- Current Iteration: {self.total_attempts}
- Maximum Iterations: {self.max_total_attempts}
- Coordinator Mode: {self.coordinator_mode}
- Has Valid Solution: {'Yes' if verification_result.get('quality_score', 0) > 0.3 else 'No'}

**Analysis Tasks:**
Please perform a comprehensive decision analysis of the entire optimization process, including:
1. Overall evaluation of current solution
2. What strategy should be taken if continuing optimization
3. Final recommendations for users

Please return analysis results in JSON format:
```json
{{
    "overall_assessment": "Overall evaluation of the optimization process",
    "solution_quality": "Evaluation of current solution quality",
    "improvement_potential": "Analysis of further improvement potential",
    "recommended_strategy": "Suggested strategy if continuing optimization",
    "user_recommendations": ["Specific recommendation 1", "Recommendation 2", "..."],
    "conclusion": "Final conclusion and recommendations"
}}
```

**Analysis Principles:**
- Objectively evaluate solution quality based on verification results
- Consider invested computational resources and time costs
- Provide practical improvement suggestions
- Give clear final conclusions"""

            response_text = call_llm_api(prompt, self.model_type)
            
            if response_text:
                # Parse LLM response
                decision_analysis = self._parse_llm_decision(response_text)
                return decision_analysis
            else:
                return self._get_default_decision_analysis(verification_result)
                
        except Exception as e:
            self.logger.error(f"Generate LLM decision analysis failed: {e}")
            return self._get_default_decision_analysis(verification_result)
    
    def _get_default_decision_analysis(self, verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get default decision analysis"""
        quality_score = verification_result.get("quality_score", 0.0)
        
        if quality_score >= 0.7:
            assessment = "Solution quality is high, acceptable"
            conclusion = "Recommend using current solution"
        elif quality_score >= 0.4:
            assessment = "Solution quality is medium, has room for improvement"
            conclusion = "Can use current solution, but recommend further optimization"
        else:
            assessment = "Solution quality is low, needs improvement"
            conclusion = "Recommend reanalyzing problem or adjusting modeling method"
        
        return {
            "overall_assessment": f"After {self.total_attempts} iterations, {assessment}",
            "solution_quality": f"Quality score {quality_score:.3f}, {assessment}",
            "improvement_potential": "Based on current results analysis, still has optimization space",
            "recommended_strategy": "Recommend rechecking problem analysis and modeling stages",
            "user_recommendations": [
                "Check if problem description is complete and accurate",
                "Verify model parameter settings",
                "Consider using different solving methods"
            ],
            "conclusion": conclusion,
            "analysis_mode": "fallback_analysis"
        }

    def handle_response_message(self, message: Message) -> None:
        """Handle response message - reset failure counter"""
        # Successful response, reset consecutive failure counter
        if message.msg_type == MessageType.RESPONSE:
            self.consecutive_failures = 0
        
        # Call parent class method
        super().handle_response_message(message)

# Factory function
def create_coordinator(mode: str = "auto", problem_input: str = "", enable_json_tracking: bool = False, model_type: str = "gemini", problem_id: str = None) -> BaseCoordinator:
    """Create coordinator instance based on mode"""
    if mode == "auto":
        # Auto mode selection based on problem input
        coordinator = BaseCoordinator()
        mode = coordinator.select_coordinator_mode(problem_input)
    
    if mode == "free":
        return FreeCoordinator(enable_json_tracking=enable_json_tracking, model_type=model_type, problem_id=problem_id)
    else:  # Default to fixed mode
        return FixedCoordinator(enable_json_tracking=enable_json_tracking, model_type=model_type, problem_id=problem_id)

def solve_optimization_problem(problem_description: str, coordinator_mode: str = "auto", problem_id: str = None, enable_json_tracking: bool = False, model_type: str = "gemini") -> Dict[str, Any]:
    """Main entry point for optimization problem solving"""
    # Create coordinator instance
    coordinator = create_coordinator(coordinator_mode, problem_description, enable_json_tracking, model_type, problem_id)
    
    # Run optimization process
    result = coordinator.run(problem_description)
    
    return result 