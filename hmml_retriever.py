"""
HMML retrieval module - based on Engi-HMML-v2.md for engineering problem domain matching and method retrieval
Based on MM-Agent's HMML retrieval mechanism implementation
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

@dataclass
class HMMLMethod:
    """HMML method data structure"""
    domain: str
    subdomain: str
    method_name: str
    modeling_method: str
    core_idea: str
    mathematical_form: str
    solution_methods: str
    engineering_applications: str
    advantages: str
    limitations: str
    
class HMMLRetriever:
    """HMML retriever - responsible for parsing and retrieving Engi-HMML library"""
    
    def __init__(self, hmml_path: str = "Engi-HMML-v2.md"):
        """Initialize HMML retriever"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hmml_path = hmml_path
        self.methods_db = {}
        self.domain_structure = {}
        self.similarity_threshold = 0.05  # Optimization threshold, improve matching accuracy
        self.power_domain_boost = 1.15  # Power domain specific weighting
        
        # Load HMML library
        self._load_hmml_library()
    
    def _load_hmml_library(self):
        """Load and parse Engi-HMML-v2.md file"""
        try:
            if not os.path.exists(self.hmml_path):
                self.logger.warning(f"HMML file does not exist: {self.hmml_path}")
                return
            
            with open(self.hmml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self._parse_hmml_content(content)
            self.logger.info(f"Successfully loaded HMML library, containing {len(self.methods_db)} methods")
            
        except Exception as e:
            self.logger.error(f"Failed to load HMML library: {e}")
    
    def _parse_hmml_content(self, content: str):
        """Parse HMML content, build three-level structure"""
        lines = content.split('\n')
        current_domain = ""
        current_subdomain = ""
        current_method = {}
        
        for line in lines:
            line = line.strip()
            
            # Identify domain (first level title)
            if line.startswith('## ') and not line.startswith('### '):
                current_domain = line[3:].strip()
                if current_domain not in self.domain_structure:
                    self.domain_structure[current_domain] = {}
                continue
            
            # Identify subdomain (second level title)
            if line.startswith('### '):
                current_subdomain = line[4:].strip()
                if current_domain and current_subdomain not in self.domain_structure[current_domain]:
                    self.domain_structure[current_domain][current_subdomain] = []
                continue
            
            # Identify method (fourth level title or list item)
            if line.startswith('#### ') or line.startswith('- '):
                # Save previous method
                if current_method and current_domain and current_subdomain:
                    self._save_method(current_domain, current_subdomain, current_method)
                
                # Start new method
                if line.startswith('#### '):
                    method_name = line[5:].strip()
                elif line.startswith('- '):
                    method_name = line[2:].strip()
                    if ':' in method_name:
                        method_name = method_name.split(':')[0].strip()
                
                current_method = {
                    'method_name': method_name,
                    'modeling_method': '',
                    'core_idea': '',
                    'mathematical_form': '',
                    'solution_methods': '',
                    'engineering_applications': '',
                    'advantages': '',
                    'limitations': ''
                }
                continue
            
            # Parse method detailed information
            if current_method:
                self._parse_method_details(line, current_method)
        
        # Save last method
        if current_method and current_domain and current_subdomain:
            self._save_method(current_domain, current_subdomain, current_method)
    
    def _parse_method_details(self, line: str, method: Dict):
        """Parse method detailed information"""
        if '<modeling_method>' in line:
            method['modeling_method'] = self._extract_field_content(line, 'modeling_method')
        elif '<core_idea>' in line:
            method['core_idea'] = self._extract_field_content(line, 'core_idea')
        elif '<mathematical_form>' in line:
            method['mathematical_form'] = self._extract_field_content(line, 'mathematical_form')
        elif '<solution_methods>' in line:
            method['solution_methods'] = self._extract_field_content(line, 'solution_methods')
        elif '<engineering_applications>' in line:
            method['engineering_applications'] = self._extract_field_content(line, 'engineering_applications')
        elif '<advantages>' in line:
            method['advantages'] = self._extract_field_content(line, 'advantages')
        elif '<limitations>' in line:
            method['limitations'] = self._extract_field_content(line, 'limitations')
    
    def _extract_field_content(self, line: str, field_name: str) -> str:
        """Extract field content from line"""
        pattern = f'<{field_name}>:(.*?)(?:<|$)'
        match = re.search(pattern, line)
        if match:
            return match.group(1).strip()
        return ""
    
    def _save_method(self, domain: str, subdomain: str, method: Dict):
        """Save method to database"""
        method_obj = HMMLMethod(
            domain=domain,
            subdomain=subdomain,
            method_name=method['method_name'],
            modeling_method=method['modeling_method'],
            core_idea=method['core_idea'],
            mathematical_form=method['mathematical_form'],
            solution_methods=method['solution_methods'],
            engineering_applications=method['engineering_applications'],
            advantages=method['advantages'],
            limitations=method['limitations']
        )
        
        # Use unique key to save
        key = f"{domain}::{subdomain}::{method['method_name']}"
        self.methods_db[key] = method_obj
        
        # Add to domain structure
        if domain in self.domain_structure and subdomain in self.domain_structure[domain]:
            self.domain_structure[domain][subdomain].append(method['method_name'])

class DomainMatcher:
    """Domain matcher - responsible for problem feature extraction and domain matching"""
    
    def __init__(self, hmml_retriever: HMMLRetriever):
        self.hmml_retriever = hmml_retriever
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Enhanced domain keyword mapping
        self.domain_keywords = {
            "1. Power and Energy Systems": [
                "power", "energy", "generation", "storage", "grid", "unit", "load", "dispatch",
                "electricity", "generator", "battery", "solar", "wind", "hydro", "thermal",
                "transmission", "distribution", "substation", "microgrid", "smart", "renewable",
                "commitment", "economic", "optimal", "flow", "market", "demand", "response"
            ],
            "2. Manufacturing and Industrial Systems": [
                "manufacturing", "production", "factory", "scheduling", "inventory", "quality",
                "shop", "floor", "process", "assembly", "capacity", "efficiency", "batch",
                "order", "machine", "equipment", "tool", "automation", "lean", "sigma"
            ],
            "3. Transportation and Logistics": [
                "transportation", "logistics", "routing", "vehicle", "supply", "traffic",
                "freight", "cargo", "delivery", "warehouse", "distribution", "shipping",
                "truck", "driver", "route", "fleet", "inventory", "chain", "last", "mile"
            ],
            "4. Financial Engineering": [
                "financial", "investment", "risk", "portfolio", "pricing", "option", "bond",
                "stock", "return", "banking", "insurance", "fund", "security", "derivative",
                "hedge", "arbitrage", "credit", "interest", "rate", "volatility", "capital"
            ],
            "5. Supply Chain Management": [
                "supply", "chain", "procurement", "supplier", "demand", "forecast", "planning",
                "inventory", "order", "delivery", "logistics", "warehouse", "distribution",
                "coordination", "collaboration", "bullwhip", "sourcing"
            ],
            "6. Environmental Engineering": [
                "environmental", "pollution", "emission", "waste", "water", "air", "carbon",
                "clean", "green", "sustainable", "recycling", "treatment", "ecology"
            ],
            "7. Chemical and Process Engineering": [
                "chemical", "reactor", "separation", "catalysis", "process", "reaction",
                "distillation", "extraction", "crystallization", "drying", "mixing", "fluid"
            ],
            "8. Telecommunications and Networks": [
                "telecommunications", "network", "signal", "spectrum", "transmission", "bandwidth",
                "wireless", "mobile", "fiber", "data", "traffic", "protocol", "communication"
            ],
            "9. Healthcare Systems": [
                "healthcare", "medical", "diagnosis", "treatment", "drug", "patient", "hospital",
                "nurse", "doctor", "surgery", "ward", "appointment", "emergency", "clinic"
            ],
            "10. Construction and Infrastructure": [
                "construction", "infrastructure", "building", "bridge", "foundation", "concrete",
                "steel", "design", "load", "safety", "project", "schedule", "cost", "material"
            ]
        }
        # Add 11th class: Smart Manufacturing and Industry 4.0
        self.domain_keywords["11. Smart Manufacturing and Industry 4.0"] = [
            "industry 4.0", "smart manufacturing", "digital twin", "iot", "iiot", "sensor", "predictive maintenance",
            "edge", "cybersecurity", "interoperability", "real-time", "quality control", "computer vision",
            "mes", "scada", "cps", "latency", "uptime", "downtime"
        ]
        
        # Enhanced method type keywords (add weight identifier)
        self.method_keywords = {
            "optimization": [
                # High weight core vocabulary
                ("minimize", 3.0), ("maximize", 3.0), ("optimal", 3.0),
                # Medium weight method vocabulary
                ("linear", 2.0), ("integer", 2.0), ("programming", 2.0), ("convex", 2.0),
                # Low weight related vocabulary
                ("cost", 1.0), ("resource", 1.0),
                ("allocation", 1.0), ("production", 1.0)
            ],
            "multi_objective": [
                ("multi-objective", 3.0), ("weight", 2.0), ("ε-constraint", 2.5), ("pareto", 3.0), ("multi objective", 3.0),
                ("NSGA", 2.5), ("frontier", 2.0), ("trade-off", 2.0), ("tradeoff", 2.0)
            ],
            "network_flow": [
                ("network", 2.0), ("flow", 2.0), ("minimum cost", 3.0), ("shortest path", 2.5), ("maximum flow", 3.0),
                ("simplex", 2.0), ("network simplex", 2.5), ("residual", 2.0), ("capacity", 1.5)
            ],
            "scheduling_planning": [
                ("scheduling", 3.0), ("scheduling", 3.0), ("job shop", 3.0), ("operation", 2.0), ("machine", 2.0),
                ("job", 2.0), ("makespan", 2.5), ("deadline", 2.5), ("planning", 2.0), ("production scheduling", 2.5),
                ("RCPSP", 2.5), ("resource constraint", 2.5), ("CPM", 2.0), ("PERT", 2.0)
            ],
            "game_theory": [
                # High weight core vocabulary
                ("game", 3.0), ("strategy", 3.0), ("equilibrium", 3.0), ("nash", 3.0),
                ("bidding", 3.5), ("two players", 2.5), ("multiple players", 2.5), 
                ("participate", 2.5), ("bid", 3.0), ("quote", 3.0),
                # Power market specific high weight vocabulary
                ("electricity market", 3.5), ("unit", 3.0), ("generator", 3.0),
                ("marginal cost", 3.0), ("clearing price", 3.0),
                # Medium weight method vocabulary
                ("cooperation", 2.0), ("non-cooperation", 2.0), ("auction", 2.5), ("mechanism design", 2.0),
                ("auction", 2.5), ("mechanism", 2.0), ("competition", 2.0),
                ("stackelberg", 2.0), ("leader", 2.0), ("follower", 2.0),
                # Low weight related vocabulary
                ("participant", 1.5), ("player", 1.5), ("payoff", 1.5), ("profit", 1.5),
                ("market", 1.5), ("capacity", 1.0),
                ("clearing", 2.0), ("price mechanism", 2.0), ("pricing", 2.0),
                ("supply", 1.5), ("demand", 1.5), ("market power", 2.0)
            ],
            "data_driven": [
                # Explicit data driven method (high weight)
                ("data-driven", 3.0), ("machine learning", 3.0), ("deep learning", 3.0), ("neural network", 3.0), ("artificial intelligence", 3.0),
                # ML specific method (medium weight)
                ("regression", 2.0), ("classification", 2.0), ("clustering", 2.0), ("reinforcement learning", 2.5),
                # General vocabulary (low weight, avoid mis-matching engineering optimization)
                ("data", 1.0), ("learning", 0.8), ("prediction", 1.0), ("pattern", 1.0), ("training", 1.0), ("model", 0.5)
            ],
            "stochastic": [
                # High weight core vocabulary
                ("stochastic", 3.0), ("uncertain", 3.0), ("probability", 3.0), ("risk", 3.0), ("robust", 3.0),
                # Medium weight method vocabulary
                ("scenario", 2.0), ("monte", 2.0),
                # Low weight related vocabulary
                ("uncertainty", 1.0)
            ],
            "heuristic": [
                # High weight core vocabulary
                ("heuristic", 3.0), ("genetic", 3.0), ("particle", 3.0), ("annealing", 3.0),
                # Medium weight method vocabulary
                ("tabu", 2.0), ("ant", 2.0), ("metaheuristic", 2.0),
                # Low weight related vocabulary
                ("algorithm", 1.0)
            ]
        }
    
    def extract_problem_features(self, problem_text: str) -> Dict[str, Any]:
        """Extract features from problem description"""
        features = {
            'domain_scores': {},
            'method_scores': {},
            'keywords': [],
            'problem_type': 'unknown'
        }
        
        # Convert to lowercase for matching
        text_lower = problem_text.lower()
        
        # Calculate domain matching score
        for domain, keywords in self.domain_keywords.items():
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                features['domain_scores'][domain] = score / len(keywords)
                features['keywords'].extend(matched_keywords)
        
        # Calculate method type score
        for method_type, keywords in self.method_keywords.items():
            score = 0.0
            total_weight = 0.0
            for keyword, weight in keywords:
                total_weight += weight
                if keyword.lower() in text_lower:
                    score += weight
            
            if score > 0:
                features['method_scores'][method_type] = score / total_weight
        
        return features
    
    def calculate_similarity(self, problem_text: str, method: HMMLMethod) -> float:
        """Calculate similarity between problem and method - based on domain keyword matching"""
        try:
            # 1. Match based on domain keywords
            domain_score = self._calculate_domain_similarity(problem_text, method.domain)
            
            # 2. Match based on method type
            method_score = self._calculate_method_type_similarity(problem_text, method)
            
            # 3. Match based on application scenario
            application_score = self._calculate_application_similarity(problem_text, method.engineering_applications)
            
            # Weighted comprehensive similarity (optimize weight allocation, improve method type matching weight)
            total_score = 0.2 * domain_score + 0.7 * method_score + 0.1 * application_score
            
            # Power domain specific weighting
            if "Power and Energy Systems" in method.domain:
                power_keywords = ["power", "energy", "generation", "grid", "unit", "dispatch", "electricity", "generator", "battery", "solar", "wind", "hydro", "thermal", "transmission", "distribution", "substation", "microgrid", "smart", "renewable", "commitment", "economic", "optimal", "flow", "market", "demand", "response"]
                if any(keyword.lower() in problem_text.lower() for keyword in power_keywords):
                    total_score *= self.hmml_retriever.power_domain_boost
            
            return min(total_score, 1.0)  # Ensure not exceed 1.0
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_domain_similarity(self, problem_text: str, method_domain: str) -> float:
        """Calculate domain similarity"""
        text_lower = problem_text.lower()
        
        # Find matching domain keywords
        if method_domain in self.domain_keywords:
            keywords = self.domain_keywords[method_domain]
            matches = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matches += 1
            
            return matches / total_keywords if total_keywords > 0 else 0.0
        
        return 0.0
    
    def _calculate_method_type_similarity(self, problem_text: str, method: HMMLMethod) -> float:
        """Calculate method type similarity (use weight)"""
        text_lower = problem_text.lower()
        max_score = 0.0
        
        for method_type, keywords in self.method_keywords.items():
            # Calculate weighted matching in problem text
            problem_score = 0.0
            total_weight = 0.0
            for keyword, weight in keywords:
                total_weight += weight
                if keyword.lower() in text_lower:
                    problem_score += weight
            
            # Check if method name and description match this type
            method_text = f"{method.method_name} {method.modeling_method}".lower()
            method_score = 0.0
            for keyword, weight in keywords:
                if keyword.lower() in method_text:
                    method_score += weight
            
            if problem_score > 0 and method_score > 0:
                # Normalize score
                problem_normalized = problem_score / total_weight
                method_normalized = method_score / total_weight
                type_score = problem_normalized * method_normalized
                max_score = max(max_score, type_score)
        
        return max_score
    
    def _calculate_application_similarity(self, problem_text: str, applications: str) -> float:
        """Calculate application scenario similarity"""
        if not applications:
            return 0.0
        
        text_lower = problem_text.lower()
        app_lower = applications.lower()
        
        # Simple keyword matching
        common_terms = [
            "optimization", "scheduling", "planning", "control", "management",
            "allocation", "decision"
        ]
        
        matches = 0
        for term in common_terms:
            if term in text_lower and term in app_lower:
                matches += 1
        
        return matches / len(common_terms) if common_terms else 0.0
    
    def rank_domains(self, features: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Rank domains based on features"""
        domain_scores = features.get('domain_scores', {})
        
        # Sort by score
        ranked = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked

class MethodExtractor:
    """Method extractor - responsible for generating modeling elements and solution templates"""
    
    def __init__(self, hmml_retriever: HMMLRetriever):
        self.hmml_retriever = hmml_retriever
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_modeling_elements(self, method: HMMLMethod, problem_text: str) -> Dict[str, Any]:
        """Extract modeling elements based on selected method"""
        elements = {
            "recommended_domain": method.domain,
            "recommended_subdomain": method.subdomain,
            "recommended_method": method.method_name,
            "modeling_approach": method.modeling_method,
            "core_methodology": method.core_idea,
            "mathematical_framework": method.mathematical_form,
            "solution_strategy": method.solution_methods,
            "application_context": method.engineering_applications,
            "method_advantages": method.advantages,
            "method_limitations": method.limitations,
            "confidence_score": 0.8  # Default confidence score
        }
        
        return elements
    
    def generate_solution_template(self, method: HMMLMethod) -> Dict[str, str]:
        """Generate solution template"""
        template = {
            "modeling_template": self._get_modeling_template(method),
            "solution_template": self._get_solution_template(method)
        }
        
        return template
    
    def _get_modeling_template(self, method: HMMLMethod) -> str:
        """Get modeling template"""
        # Based on method type and domain to generate specific modeling guidance
        if "optimization" in method.method_name.lower() or "programming" in method.method_name.lower():
            if "power" in method.domain.lower() or "energy" in method.domain.lower():
                return """Modeling template based on power system optimization:
1. Define time set and device set
2. Declare decision variables (electricity power, energy storage state, etc.)
3. Build objective function (minimize electricity cost or maximize profit)
4. Set power balance constraint
5. Add device capacity constraint and technical constraint
6. Determine parameter value (demand, cost, capacity, etc.)"""
            elif "manufacturing" in method.domain.lower():
                return """Modeling template based on manufacturing system optimization:
1. Define time period and product set
2. Declare decision variables (production quantity, inventory quantity, device status, etc.)
3. Build objective function (minimize production cost or maximize profit)
4. Set demand satisfaction constraint
5. Add production capacity constraint and inventory constraint
6. Determine parameter value (demand, cost, capacity, etc.)"""
            elif "financial" in method.domain.lower():
                return """Modeling template based on financial engineering optimization:
1. Define asset set and time period
2. Declare decision variables (investment weight, transaction quantity, etc.)
3. Build objective function (maximize profit or minimize risk)
4. Set budget constraint and risk limit
5. Add regulatory requirement constraint
6. Determine parameter value (yield, covariance matrix, etc.)"""
            else:
                return """Generic optimization modeling template:
1. Define decision variable and index set
2. Build objective function (optimization direction)
3. Set constraint condition (equation and inequality)
4. Determine parameter value and boundary condition
5. Verify the feasibility and rationality of the model"""
        elif "game" in method.method_name.lower():
            return """Game theory modeling template:
1. Identify participants and decision sequence
2. Define strategy space for each participant
3. Build payoff function or utility function
4. Determine information structure (complete/incomplete information)
5. Analyze existence and uniqueness of equilibrium solution.
6. Solve equilibrium strategy"""
        elif "learning" in method.method_name.lower() or "data" in method.method_name.lower():
            return """Data driven modeling template:
1. Data collection and preprocessing
2. Feature engineering and selection
3. Model selection and architecture design
4. Training/validation/test set division
5. Model training and hyperparameter tuning
6. Model evaluation and validation"""
        else:
            return """Generic engineering modeling template:
1. Problem abstraction and mathematical representation
2. Determine key variables and parameters
3. Build mathematical model or algorithm framework
4. Set constraints and boundaries
5. Select solving method and tool
6. Model validation and result analysis"""
    
    def _get_solution_template(self, method: HMMLMethod) -> str:
        """Get solution template"""
        return method.solution_methods if method.solution_methods else "Use appropriate numerical methods to solve"
    
    def format_json_output(self, elements: Dict[str, Any], template: Dict[str, str]) -> Dict[str, Any]:
        """Format as standard JSON output"""
        formatted_output = {
            "hmml_analysis": {
                "domain": elements["recommended_domain"],
                "subdomain": elements["recommended_subdomain"],  
                "method": elements["recommended_method"],
                "confidence": elements["confidence_score"]
            },
            "modeling_guidance": {
                "approach": elements["modeling_approach"],
                "methodology": elements["core_methodology"],
                "framework": elements["mathematical_framework"],
                "solution_strategy": elements["solution_strategy"]
            },
            "application_context": {
                "applications": elements["application_context"],
                "advantages": elements["method_advantages"],
                "limitations": elements["method_limitations"]
            },
            "templates": template
        }
        
        return formatted_output

class HMMLManager:
    """HMML manager - unified interface"""
    
    def __init__(self, hmml_path: str = "Engi-HMML-v2.md"):
        """Initialize HMML manager"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            self.retriever = HMMLRetriever(hmml_path)
            self.matcher = DomainMatcher(self.retriever)
            self.extractor = MethodExtractor(self.retriever)
            self.available = len(self.retriever.methods_db) > 0
            
            if self.available:
                self.logger.info("HMML manager initialized successfully")
            else:
                self.logger.warning("HMML library is empty, functionality limited")
                
        except Exception as e:
            self.logger.error(f"HMML manager initialization failed: {e}")
            self.available = False
    
    def retrieve_methods(self, problem_text: str, max_results: int = 3) -> Dict[str, Any]:
        """Retrieve related methods"""
        if not self.available:
            return self._get_fallback_result(problem_text)
        
        try:
            # Extract problem features
            features = self.matcher.extract_problem_features(problem_text)
            
            # Calculate similarity of all methods
            method_scores = []
            for key, method in self.retriever.methods_db.items():
                similarity = self.matcher.calculate_similarity(problem_text, method)
                method_scores.append((method, similarity))
            
            # Sort by similarity
            method_scores.sort(key=lambda x: x[1], reverse=True)

            # Enhanced re-ranking: more accurately distinguish engineering optimization problems from machine learning methods
            text_l = problem_text.lower()
            
            # Engineering optimization signal words
            optimization_signals = any(k in text_l for k in [
                "minimize", "maximize", "optimal", "cost",
                "game", "equilibrium", "bilevel", "stackelberg", "nash", "mpec", "kkt",
                "scheduling", "allocation", "planning", "programming",
                "constraint", "objective", "decision variable"
            ])
            
            # Clear machine learning signal words (exclude generic words)
            ml_signals = any(k in text_l for k in [
                "machine learning", "deep learning", 
                "neural network", "artificial intelligence", "training data",
                "feature engineering", "model training", "prediction model"
            ])
            
            # Generic prediction words (may mislead)
            generic_prediction = any(k in text_l for k in ["project", "prediction", "forecast"]) and not ml_signals
            
            if optimization_signals or ml_signals or generic_prediction:
                reranked = []
                for method, score in method_scores:
                    boost = 1.0
                    mt = f"{method.method_name} {method.modeling_method}".lower()
                    
                    # Boost engineering optimization methods
                    if optimization_signals:
                        if any(k in mt for k in ["optimization", "programming", "bilevel", "stackelberg", "game", "equilibrium", "mpec", "kkt"]):
                            boost *= 1.3
                        elif any(k in mt for k in ["linear", "integer", "nonlinear", "convex", "robust", "stochastic"]):
                            boost *= 1.2
                    
                    # Clear ML problems boost ML methods
                    if ml_signals and any(k in mt for k in ["learning", "neural", "deep", "regression", "classification"]):
                        boost *= 1.2
                    
                    # Suppress mismatched methods
                    if optimization_signals and any(k in mt for k in ["deep", "neural", "cnn", "rnn", "transformer", "reinforcement"]):
                        boost *= 0.5  # Stronger suppression
                    elif generic_prediction and not ml_signals and any(k in mt for k in ["deep", "neural", "learning"]):
                        boost *= 0.7  # Medium suppression
                    
                    reranked.append((method, min(score * boost, 1.0)))
                method_scores = sorted(reranked, key=lambda x: x[1], reverse=True)
            
            # Filter out methods with low similarity (using dynamic threshold)
            dynamic_threshold = max(self.retriever.similarity_threshold, 
                                  max(score for _, score in method_scores[:5]) * 0.7 if method_scores else 0.0)
            
            filtered_methods = [(method, score) for method, score in method_scores 
                              if score >= dynamic_threshold]
            
            # If there are matching methods, take the top max_results
            if filtered_methods:
                top_methods = filtered_methods[:max_results]
                self.logger.info(f"Using dynamic threshold {dynamic_threshold:.3f} to filter, get {len(top_methods)} matching methods")
            else:
                # If there are no high similarity matches, use domain ranking strategy
                self.logger.info("No high similarity matches found, using domain ranking strategy")
                ranked_domains = self.matcher.rank_domains(features)
                if ranked_domains:
                    # Select methods from the highest ranked domain
                    top_domain = ranked_domains[0][0]
                    domain_methods = [
                        method for method in self.retriever.methods_db.values()
                        if method.domain == top_domain
                    ]
                    if domain_methods:
                        # Select the method with the highest similarity in this domain
                        domain_scores = [(method, self.matcher.calculate_similarity(problem_text, method)) 
                                       for method in domain_methods]
                        domain_scores.sort(key=lambda x: x[1], reverse=True)
                        best_domain_method = domain_scores[0]
                        top_methods = [best_domain_method]
                    else:
                        # If there are no methods in the domain, select the method with the highest global similarity
                        if method_scores:
                            top_methods = [method_scores[0]]
                        else:
                            top_methods = []
                else:
                    # If there are no domain matches, select the method with the highest global similarity
                    if method_scores:
                        top_methods = [method_scores[0]]
                    else:
                        top_methods = []
            
            if top_methods:
                # Use the best matching method to generate results
                best_method = top_methods[0][0]
                elements = self.extractor.extract_modeling_elements(best_method, problem_text)
                template = self.extractor.generate_solution_template(best_method)
                
                # Update confidence score
                elements["confidence_score"] = top_methods[0][1]
                
                result = self.extractor.format_json_output(elements, template)
                # Return multiple candidates, for Analyzer reference selection
                result["retrieved_methods"] = [
                    {
                        "method": method.method_name,
                        "domain": method.domain,
                        "subdomain": method.subdomain,
                        "similarity": score,
                        "description": method.modeling_method
                    }
                    for method, score in top_methods
                ]
                
                return result
            else:
                return self._get_fallback_result(problem_text)
                
        except Exception as e:
            self.logger.error(f"Method retrieval failed: {e}")
            return self._get_fallback_result(problem_text)
    
    def _get_fallback_result(self, problem_text: str = "") -> Dict[str, Any]:
        """Get fallback result - based on problem features for intelligent degradation"""
        # Intelligent detection of problem type
        problem_lower = problem_text.lower()
        
        # Power domain specific degradation
        if any(keyword in problem_lower for keyword in ["power", "energy", "generation", "dispatch", "unit", "generator", "battery", "solar", "wind", "hydro", "thermal", "transmission", "distribution", "substation", "microgrid", "smart", "renewable", "commitment", "economic", "optimal", "flow", "market", "demand", "response"]):
            return {
                "hmml_analysis": {
                    "domain": "Power and Energy Systems",
                    "subdomain": "Optimization Problems", 
                    "method": "Mixed-Integer Linear Programming",
                    "confidence": 0.4
                },
                "modeling_guidance": {
                    "approach": "Power system optimization modeling method",
                    "methodology": "Construct economic dispatch or unit commitment model",
                    "framework": "Mixed integer linear programming framework",
                    "solution_strategy": "Use commercial solver (Gurobi/CPLEX) or open source solver (GLPK/CBC)"
                },
                "application_context": {
                    "applications": "Power system economic dispatch, unit commitment, energy storage optimization",
                    "advantages": "Designed for power system characteristics, considering physical constraints",
                    "limitations": "Need precise power system parameters"
                },
                "templates": {
                    "modeling_template": "Power system optimization modeling template",
                    "solution_template": "MILP solution method"
                },
                "retrieved_methods": []
            }
        
        # Default generic degradation
        return {
            "hmml_analysis": {
                "domain": "General Engineering",
                "subdomain": "Optimization",
                "method": "Mathematical Programming",
                "confidence": 0.2
            },
            "modeling_guidance": {
                "approach": "General modeling method based on mathematical optimization",
                "methodology": "Construct objective function and constraints, find optimal solution",
                "framework": "linear or nonlinear programming framework",
                "solution_strategy": "Use appropriate optimization algorithms to solve"
            },
            "application_context": {
                "applications": "Applicable to various engineering optimization problems",
                "advantages": "Method mature, theoretical foundation solid",
                "limitations": "Need clear mathematical expressions"
            },
            "templates": {
                "modeling_template": "General optimization modeling method",
                "solution_template": "Numerical optimization solution"
            },
            "retrieved_methods": []
        }

if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)
    
    hmml_manager = HMMLManager("Engi-HMML-v2.md")
    
    # Test case 1: Power system optimization problem
    test_problem_1 = """
    A power company needs to optimize the operation scheduling of its power generation units. The company has three power generation units, each with different power generation costs and capacity limits. It is necessary to meet the 24-hour power demand while minimizing the total power generation cost.
    """
    
    # Test case 2: Power market bidding problem
    test_problem_2 = """
    In the electricity market, multiple power generators need to participate in bidding. Each generator has its own marginal cost and needs to formulate the optimal bidding strategy to maximize profits. The market clearing price is determined by the quotations of all participants.
    """
    
    print("="*60)
    print("Test case 1: Power system optimization problem")
    print("="*60)
    result1 = hmml_manager.retrieve_methods(test_problem_1)
    print(json.dumps(result1, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("Test case 2: Power market bidding problem")
    print("="*60)
    result2 = hmml_manager.retrieve_methods(test_problem_2)
    print(json.dumps(result2, indent=2, ensure_ascii=False))
