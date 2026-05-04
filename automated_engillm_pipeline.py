#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EngiLLM Fully Automated Pipeline
CSV Problem Loading → EngiLLM Report Generation → GPT Four-Dimension Evaluation
"""

import pandas as pd
import json
import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import subprocess
import time

# Import existing modules
from enhanced_coordinator import solve_optimization_problem
from EngiLLM_modeling_evaluator import EngiLLMReportEvaluator

# Import EngiLLM components
from enhanced_coordinator import solve_optimization_problem
from EngiLLM_modeling_evaluator import EngiLLMReportEvaluator
from EngiLLM_numerical_analyzer import EngiLLMNumericalAnalyzer
from specialized_agents import reset_token_usage, get_token_usage


class EngiLLMFileManager:
    """EngiLLM File Path Manager - Unified management of all output file paths"""
    
    def __init__(self, base_output_dir: str = "engillm_outputs"):
        self.base_output_dir = Path(base_output_dir)
        self.file_registry = {}  # Record all generated file paths
        
        # Create unified output directory structure
        self.setup_output_directories()
        
    def setup_output_directories(self):
        """Create unified output directory structure"""
        directories = [
            self.base_output_dir,
            self.base_output_dir / "reports",           # EngiLLM generated reports
            self.base_output_dir / "json_tracking",     # JSON tracking files
            self.base_output_dir / "evaluations",       # Evaluation results
            self.base_output_dir / "numerical_analysis", # Numerical analysis results
            self.base_output_dir / "visualizations",    # Visualization charts
            self.base_output_dir / "summaries"          # Summary reports
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Created unified output directory: {self.base_output_dir}")
        
    def get_file_path(self, problem_id: str, file_type: str, timestamp: str = None) -> str:
        """Generate standardized file paths"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        file_patterns = {
            'report': f"engillm_report_{problem_id}_{timestamp}.md",
            'json_tracking': f"json_tracking_{problem_id}_{timestamp}.json", 
            'evaluation': f"evaluation_{problem_id}_{timestamp}.json",
            'numerical_analysis': f"numerical_analysis_{problem_id}_{timestamp}.json",
            'visualization': f"quality_curve_{problem_id}_{timestamp}.png",
            'summary': f"summary_{problem_id}_{timestamp}.txt"
        }
        
        if file_type not in file_patterns:
            raise ValueError(f"Unknown file type: {file_type}")
            
        subdir_map = {
            'report': 'reports',
            'json_tracking': 'json_tracking', 
            'evaluation': 'evaluations',
            'numerical_analysis': 'numerical_analysis',
            'visualization': 'visualizations',
            'summary': 'summaries'
        }
        
        file_path = self.base_output_dir / subdir_map[file_type] / file_patterns[file_type]
        return str(file_path)
        
    def record_problem_files(self, problem_id: str, files: Dict[str, str]):
        """Record all file paths related to a problem"""
        if problem_id not in self.file_registry:
            self.file_registry[problem_id] = {}
        self.file_registry[problem_id].update(files)
        
    def get_problem_files(self, problem_id: str) -> Dict[str, str]:
        """Get all file paths related to a problem"""
        return self.file_registry.get(problem_id, {})
        
    def find_json_tracking_file(self, problem_id: str) -> Optional[str]:
        """Find JSON tracking file"""
        # Search from registry first
        files = self.get_problem_files(problem_id)
        if 'json_tracking' in files and os.path.exists(files['json_tracking']):
            return files['json_tracking']
            
        # Search in unified directory
        tracking_dir = self.base_output_dir / "json_tracking"
        if tracking_dir.exists():
            import glob
            import os
            
            # Search for files containing problem ID (support timestamp format)
            all_files = glob.glob(str(tracking_dir / "*.json"))
            
            # Sort by modification time in descending order, get the latest file
            all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Find files containing problem ID
            for file_path in all_files:
                filename = os.path.basename(file_path)
                # Check if filename corresponds to current problem ID
                if problem_id in filename or f"problem_{problem_id}" in filename.lower():
                    return file_path
            
            # Compatible search for original format
            patterns = [
                f"json_tracking_{problem_id}.json",  # Original format (in root directory)
                f"json_tracking_problem_*{problem_id}*.json"  # Another format
            ]
            
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    return matches[0]
                    
        return None
        
    def find_report_file(self, problem_id: str) -> Optional[str]:
        """Find report file"""
        # Search from registry first
        files = self.get_problem_files(problem_id)
        if 'report' in files and os.path.exists(files['report']):
            return files['report']
            
        # Search in unified directory and other common directories
        search_dirs = [
            str(self.base_output_dir / "reports"),
            "automated_results/reports",
            "experiments/outputs/debug_reports"
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                import glob
                patterns = [
                    f"{search_dir}/*{problem_id}*.md",
                    f"{search_dir}/engillm_report_{problem_id}_*.md",
                    f"{search_dir}/debug_{problem_id}_*.md"
                ]
                for pattern in patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        return matches[0]
                        
        return None


# Global file manager instance
file_manager = EngiLLMFileManager()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutomatedEngiLLMPipeline:
    """EngiLLM Fully Automated Pipeline"""
    
    def __init__(self, csv_file: str = "EngiLLM Dataset-Sheet1.xlsx", 
                 output_dir: str = "automated_results", 
                 coordinator_mode: str = "free",
                 model_type: str = "gpt4o"):
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.coordinator_mode = coordinator_mode
        self.model_type = model_type
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "evaluations").mkdir(exist_ok=True)
        
        # Initialize evaluator with gpt-4.1-nano for evaluation
        self.evaluator = EngiLLMReportEvaluator(evaluation_model="gpt-4.1-nano")
        
        logger.info(f"Automated pipeline initialization completed")
        logger.info(f"CSV file: {self.csv_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Coordinator mode: {self.coordinator_mode}")
        logger.info(f"Model type: {self.model_type}")
    
    def _find_latest_json_tracking_file(self) -> Optional[str]:
        """Find the latest generated JSON tracking file"""
        import glob
        import os
        
        # Search in unified directory
        tracking_dir = file_manager.base_output_dir / "json_tracking"
        if tracking_dir.exists():
            json_files = glob.glob(str(tracking_dir / "*.json"))
            if json_files:
                # Sort by modification time in descending order, return the latest
                json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return json_files[0]
        
        # Compatible original directory search
        json_files = glob.glob("json_tracking_*.json")
        if json_files:
            json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return json_files[0]
            
        return None

    def load_problems_from_csv(self) -> List[Dict[str, Any]]:
        """Load all problems from CSV file"""
        try:
            # ✅ According to file extension, select the correct read method
            if self.csv_file.endswith('.xlsx'):
                df = pd.read_excel(self.csv_file, engine='openpyxl')
            elif self.csv_file.endswith('.csv'):
                df = pd.read_csv(self.csv_file)
            else:
                # Try automatic detection
                try:
                    df = pd.read_excel(self.csv_file, engine='openpyxl')
                except:
                    df = pd.read_csv(self.csv_file)
            
            problems = []
            
            for index, row in df.iterrows():
                # Check Problem column instead of ID column
                if pd.notna(row['Problem']):
                    # Create problem ID: use numeric index to create P{index}
                    numeric_index = len(problems) + 1  # Use processed problem count + 1
                    problem_id = f"P{numeric_index}"
                    
                    # Get problem text from Problem column
                    problem_text = str(row['Problem']).strip()
                    
                    problem = {
                        'id': problem_id,
                        'problem_text': problem_text,
                        'field': str(row['Field']) if pd.notna(row['Field']) else "",
                        # Set reference same as problem_text to ensure consistency
                        'reference': problem_text,
                        'information_extraction': str(row['Information Extraction']) if pd.notna(row['Information Extraction']) else "",
                        'domain_specific_reasoning': str(row['Domain-Specific Reasoning']) if pd.notna(row['Domain-Specific Reasoning']) else "",
                        'multi_objective_decision': str(row['Multi-Objective Decision-Making']) if pd.notna(row['Multi-Objective Decision-Making']) else "",
                        'uncertainty_handling': str(row['Uncertainty Handling']) if pd.notna(row['Uncertainty Handling']) else "",
                        'feasibility': str(row['Feasibility']) if pd.notna(row['Feasibility']) else ""
                    }
                    problems.append(problem)
            
            logger.info(f"Successfully loaded {len(problems)} problems")
            return problems
            
        except Exception as e:
            logger.error(f"Failed to load CSV file: {str(e)}")
            raise
    
    def run_engillm_for_problem(self, problem: Dict[str, Any]) -> Optional[str]:
        """Run EngiLLM to generate report for a single problem"""
        problem_id = problem['id']
        problem_text = problem['problem_text']
        
        logger.info(f"🚀 Starting EngiLLM for problem {problem_id}")
        
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Reset token statistics and record start time
            reset_token_usage()
            start_time = datetime.now()
            
            # Run solve_optimization_problem
            result = solve_optimization_problem(
                problem_description=problem_text,
                coordinator_mode=self.coordinator_mode,
                problem_id=problem_id,
                enable_json_tracking=True,
                model_type=self.model_type
            )
            
            # Calculate total duration and get token usage
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            token_usage = get_token_usage()
            
            # Record statistics
            logger.info(f"📊 EngiLLM statistics - Total duration: {total_duration:.2f} seconds")
            logger.info(f"📊 Token usage - Input: {token_usage['input_tokens']}, Output: {token_usage['output_tokens']}, Total: {token_usage['total_tokens']}")
            
            # Store statistics in results
            if result:
                result['engillm_stats'] = {
                    'total_duration_seconds': total_duration,
                    'input_tokens': token_usage['input_tokens'],
                    'output_tokens': token_usage['output_tokens'],
                    'total_tokens': token_usage['total_tokens'],
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat()
                }
            
            if result and 'debug_file' in result:
                # Move generated report to our directory
                original_file = result['debug_file']
                if os.path.exists(original_file):
                    # Ensure filename contains problem number
                    report_filename = file_manager.get_file_path(problem_id, 'report', f"{problem_id}_{timestamp}")
                    report_path = Path(report_filename)
                    
                    # Ensure directory exists
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    import shutil
                    shutil.copy2(original_file, report_path)
                    
                    # Record file paths
                    generated_files = {'report': str(report_path)}
                    
                    # Find JSON tracking file (find latest generated file)
                    json_tracking_file = self._find_latest_json_tracking_file()
                    if json_tracking_file:
                        generated_files['json_tracking'] = json_tracking_file
                        logger.info(f"📊 Found JSON tracking file: {json_tracking_file}")
                    else:
                        logger.warning(f"⚠️ No JSON tracking file found for {problem_id}")
                    
                    # Record to file manager
                    file_manager.record_problem_files(problem_id, generated_files)
                    
                    logger.info(f"✅ Problem {problem_id} report generated successfully: {report_path}")
                    
                    # Return dictionary with statistics
                    return {
                        'report_path': str(report_path),
                        'engillm_stats': result.get('engillm_stats', {})
                    }
                else:
                    logger.warning(f"⚠️ Debug file for problem {problem_id} does not exist: {original_file}")
                    return None
            else:
                logger.warning(f"⚠️ EngiLLM run failed for problem {problem_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error running EngiLLM for problem {problem_id}: {str(e)}")
            return None
    
    def evaluate_report(self, report_path: str, problem_id: str, problem_data: Dict[str, Any], engillm_stats: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Evaluate the generated report using CSV-based evaluation criteria"""
        logger.info(f"📊 Starting evaluation for problem {problem_id}")
        
        try:
            # ✅ Construct evaluation criteria dictionary from CSV data
            criteria = {
                'problem_id': problem_data['id'],
                'problem': problem_data['problem_text'],
                'information_extraction': problem_data['information_extraction'],
                'domain_specific_reasoning': problem_data['domain_specific_reasoning'],
                'multi_objective_decision': problem_data['multi_objective_decision'],
                'uncertainty_handling': problem_data['uncertainty_handling'],
                'feasibility': problem_data['feasibility'],
                'numerical_solution': '',  # Numerical solution uses default standard
                'field': problem_data['field'],
                'reference': problem_data['reference']  # ✅ Use reference field from CSV
            }
            
            logger.info(f"✅ Using CSV-based evaluation criteria for {problem_id}")
            logger.info(f"Reference: {criteria['reference'][:100]}..." if len(criteria['reference']) > 100 else f"Reference: {criteria['reference']}")
            
            # ✅ Use modified evaluation method, directly pass criteria
            results = self.evaluator.evaluate_engillm_report_with_criteria(
                report_file_path=report_path,
                criteria=criteria,
                file_type="markdown",
                engillm_stats=engillm_stats
            )
            
            # Save evaluation results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            eval_filename = file_manager.get_file_path(problem_id, 'evaluation', f"{problem_id}_{timestamp}")
            eval_path = Path(eval_filename)
            
            # Ensure directory exists
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.evaluator.save_evaluation_results(results, str(eval_path))
            
            logger.info(f"✅ Problem {problem_id} evaluation completed: {eval_path}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error evaluating problem {problem_id}: {str(e)}")
            return None
    
    def run_single_problem(self, problem_id: str) -> Dict[str, Any]:
        """Run complete pipeline for a single problem"""
        logger.info(f"🎯 Starting processing for problem {problem_id}")
        
        # Load problems
        problems = self.load_problems_from_csv()
        problem = next((p for p in problems if p['id'] == problem_id), None)
        
        if not problem:
            raise ValueError(f"Problem {problem_id} not found in CSV file")
        
        result = {
            'problem_id': problem_id,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'report_path': None,
            'evaluation_results': None,
            'error': None
        }
        
        try:
            # Step 1: Run EngiLLM to generate reports
            engillm_result = self.run_engillm_for_problem(problem)
            
            if isinstance(engillm_result, dict):
                result['report_path'] = engillm_result.get('report_path')
                result['engillm_stats'] = engillm_result.get('engillm_stats', {})
            else:
                result['report_path'] = engillm_result
                result['engillm_stats'] = {}
            
            if not result['report_path']:
                result['status'] = 'failed'
                result['error'] = 'EngiLLM report generation failed'
                return result
            
            # Step 2: Evaluate reports
            evaluation_results = self.evaluate_report(result['report_path'], problem_id, problem, result['engillm_stats'])
            result['evaluation_results'] = evaluation_results
            
            if not evaluation_results:
                result['status'] = 'partial'
                result['error'] = 'Report evaluation failed'
                return result
            
            result['status'] = 'completed'
            result['end_time'] = datetime.now().isoformat()
            
            # Print summaries
            self._print_problem_summary(problem_id, evaluation_results)
            
            return result
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['end_time'] = datetime.now().isoformat()
            logger.error(f"❌ Problem {problem_id} processing failed: {str(e)}")
            return result
    
    def run_batch_problems(self, problem_ids: List[str] = None) -> Dict[str, Any]:
        """Run multiple problems in batch"""
        if problem_ids is None:
            # Run all problems in batch
            problems = self.load_problems_from_csv()
            problem_ids = [p['id'] for p in problems]
        
        logger.info(f"🚀 Starting batch processing of {len(problem_ids)} problems")
        
        batch_results = {
            'start_time': datetime.now().isoformat(),
            'problem_ids': problem_ids,
            'results': {},
            'summary': {}
        }
        
        successful_count = 0
        failed_count = 0
        
        for i, problem_id in enumerate(problem_ids, 1):
            logger.info(f"📋 Progress: {i}/{len(problem_ids)} - Problem {problem_id}")
            
            try:
                result = self.run_single_problem(problem_id)
                batch_results['results'][problem_id] = result
                
                if result['status'] == 'completed':
                    successful_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                batch_results['results'][problem_id] = {
                    'problem_id': problem_id,
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"❌ Problem {problem_id} processing error: {str(e)}")
        
        batch_results['end_time'] = datetime.now().isoformat()
        batch_results['summary'] = {
            'total_problems': len(problem_ids),
            'successful': successful_count,
            'failed': failed_count,
            'success_rate': successful_count / len(problem_ids) if problem_ids else 0
        }
        
        # Save batch results to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_file = self.output_dir / f"batch_results_{timestamp}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        # Print batch summaries
        self._print_batch_summary(batch_results)
        
        logger.info(f"✅ Batch processing completed, results saved to: {batch_file}")
        
        return batch_results
    
    def _print_problem_summary(self, problem_id: str, evaluation_results: Dict[str, Any]):
        """Print evaluation summary for a single problem"""
        print(f"\n{'='*60}")
        print(f"📊 Problem {problem_id} Evaluation Results")
        print(f"{'='*60}")
        
        overall = evaluation_results.get('overall_assessment', {})
        print(f"📈 Average Score: {overall.get('average_score', 'N/A')}/10")
        print(f"🏆 Performance Level: {overall.get('overall_performance', 'N/A')}")
        print(f"📏 Dimensions Evaluated: {overall.get('dimensions_evaluated', 'N/A')}")
        
        print(f"\n📋 Dimension Scores:")
        dimension_scores = evaluation_results.get('dimension_scores', {})
        for dimension, score_data in dimension_scores.items():
            if isinstance(score_data, dict) and 'score' in score_data:
                print(f"  {dimension}: {score_data['score']}/10")
            else:
                print(f"  {dimension}: Error")
        
        print(f"{'='*60}")
    
    def _print_batch_summary(self, batch_results: Dict[str, Any]):
        """Print batch processing summary"""
        summary = batch_results['summary']
        
        print(f"\n{'='*80}")
        print(f"🎯 Batch Processing Summary")
        print(f"{'='*80}")
        print(f"📊 Total Problems: {summary['total_problems']}")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        
        # Display evaluation scores for each problem in batch
        print(f"\n📋 Problem Evaluation Scores:")
        for problem_id, result in batch_results['results'].items():
            if result['status'] == 'completed' and result['evaluation_results']:
                overall = result['evaluation_results'].get('overall_assessment', {})
                score = overall.get('average_score', 'N/A')
                print(f"  {problem_id}: {score}/10")
            else:
                print(f"  {problem_id}: {result['status']}")
        
        print(f"{'='*80}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='EngiLLM Fully Automated Pipeline')
    parser.add_argument('--csv', default='EngiLLM Dataset-Sheet1.xlsx', 
                       help='Evaluation standard CSV file path')
    parser.add_argument('--output', default='automated_results', 
                       help='Output directory')
    parser.add_argument('--mode', choices=['fixed', 'free'], default='free',
                       help='Coordinator mode')
    parser.add_argument('--model', choices=['gemini', 'gpt4', 'gpt4o', 'gpt5', 'gpt4.1nano', 'deepseek'], default='gpt4o',
                       help='LLM model type')
    
    # Run mode selection
    parser.add_argument('--problem', help='Run single problem ID (e.g., P1)')
    parser.add_argument('--index', type=int, help='Run single problem by index (0-based, e.g., 0 means P1)')
    parser.add_argument('--batch', nargs='*', help='Run multiple specified problem IDs (e.g., P1 P2 P3)')
    parser.add_argument('--all', action='store_true', help='Run all problems')
    parser.add_argument('--list', action='store_true', help='List all available problems')
    
    args = parser.parse_args()
    
    try:
        # Create pipeline instance
        pipeline = AutomatedEngiLLMPipeline(
            csv_file=args.csv,
            output_dir=args.output,
            coordinator_mode=args.mode,
            model_type=args.model
        )
        
        # List available problems
        if args.list:
            problems = pipeline.load_problems_from_csv()
            print("\n" + "=" * 60)
            print("📋 Available Problems List:")
            print("=" * 60)
            for problem in problems:
                # If field is empty or meaningless, show first 50 chars of problem text
                if problem['field'] and not problem['field'].startswith('0–') and not problem['field'].startswith('•'):
                    field = problem['field'][:40] + "..." if len(problem['field']) > 40 else problem['field']
                else:
                    field = problem['problem_text'][:50] + "..." if len(problem['problem_text']) > 50 else problem['problem_text']
                print(f"  {problem['id']}: {field}")
            print("=" * 60)
            print(f"Total: {len(problems)} problems")
            return 0
        
        # Run single problem (by problem ID)
        if args.problem:
            result = pipeline.run_single_problem(args.problem)
            if result['status'] == 'completed':
                print(f"\n✅ Problem {args.problem} processing completed!")
                return 0
            else:
                print(f"\n❌ Problem {args.problem} processing failed: {result.get('error', 'Unknown error')}")
                return 1
        
        # Run single problem (by index)
        if args.index is not None:
            problems = pipeline.load_problems_from_csv()
            if args.index < 0 or args.index >= len(problems):
                print(f"❌ Error: Index {args.index} out of range (0-{len(problems)-1})")
                return 1
            
            problem_id = problems[args.index]['id']
            print(f"🎯 Running problem index {args.index} -> {problem_id}")
            
            result = pipeline.run_single_problem(problem_id)
            if result['status'] == 'completed':
                print(f"\n✅ Problem {problem_id} (index {args.index}) processing completed!")
                return 0
            else:
                print(f"\n❌ Problem {problem_id} (index {args.index}) processing failed: {result.get('error', 'Unknown error')}")
                return 1
        
        # Run batch of specified problems
        if args.batch is not None:
            if len(args.batch) == 0:
                parser.error("Please specify problem IDs to run")
            result = pipeline.run_batch_problems(args.batch)
            return 0 if result['summary']['failed'] == 0 else 1
        
        # Run all problems
        if args.all:
            result = pipeline.run_batch_problems()
            return 0 if result['summary']['failed'] == 0 else 1
        
        # Default: show help
        parser.print_help()
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 