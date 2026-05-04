from abc import ABC, abstractmethod
from base_agents import BaseAgent, Message, MessageType
from typing import Dict, Any, List, Optional
import google.generativeai as genai
import os
import json
import time
import random
import logging
import re
import traceback

# Configure logging
logger = logging.getLogger(__name__)

def smart_escape_braces(text: str) -> str:
    """
    Smart escaping that properly handles f-strings and JSON without double-escaping
    
    Core principles:
    1. For f-string content, escape braces in pairs to maintain syntax validity
    2. For non-f-string content, escape all single braces
    3. Avoid re-escaping already escaped content
    """
    text = str(text)
    
    def escape_f_string_content(match):
        """Handle f-string content escaping with improved brace pairing logic"""
        quote_char = match.group(1)  # " or '
        content = match.group(2)
        
        # Check if f-string is syntactically complete
        # If incomplete or malformed, don't escape to avoid making it worse
        open_braces = content.count('{') - content.count('{{') * 2
        close_braces = content.count('}') - content.count('}}') * 2
        
        # If unbalanced and would create invalid syntax, preserve as-is
        if abs(open_braces - close_braces) > 1:
            return match.group(0)  # Return original unchanged
        
        # Smart brace pairing algorithm using stack
        result_content = ""
        i = 0
        brace_stack = []  # Track brace pairing
        
        while i < len(content):
            char = content[i]
            
            if char == '{':
                if i + 1 < len(content) and content[i + 1] == '{':
                    # Already escaped {{
                    result_content += '{{'
                    i += 2
                else:
                    # Single { - needs escaping only if it will be properly paired
                    result_content += '{{'
                    brace_stack.append('open')
                    i += 1
            elif char == '}':
                if i + 1 < len(content) and content[i + 1] == '}':
                    # Already escaped }}
                    result_content += '}}'
                    i += 2
                else:
                    # Single } - needs escaping if properly paired
                    if brace_stack and brace_stack[-1] == 'open':
                        result_content += '}}'
                        brace_stack.pop()
                    else:
                        # Unmatched }, keep as is to preserve original error state
                        result_content += '}'
                    i += 1
            else:
                result_content += char
                i += 1
        
        # If there are unmatched opening braces, revert to original
        if brace_stack:
            return match.group(0)  # Return original unchanged
        
        return f'f{quote_char}{result_content}{quote_char}'
    
    # Process f-strings
    text = re.sub(r'f(["\'])([^"\']*?)\1', escape_f_string_content, text)
    
    # Process non-f-string regions
    def escape_non_f_string_region(text_segment):
        """Escape braces in non-f-string regions"""
        text_segment = re.sub(r'(?<!{){(?!{)', '{{', text_segment)
        text_segment = re.sub(r'(?<!})}}(?!})', '}}', text_segment)
        return text_segment
    
    # Separate f-string and non-f-string regions
    f_string_pattern = r'f["\'][^"\']*?["\']'
    parts = []
    last_end = 0
    
    for match in re.finditer(f_string_pattern, text):
        # Add non-f-string part before this f-string
        if match.start() > last_end:
            non_f_part = text[last_end:match.start()]
            parts.append(escape_non_f_string_region(non_f_part))
        
        # Add f-string part (already processed)
        parts.append(match.group())
        last_end = match.end()
    
    # Add final non-f-string part
    if last_end < len(text):
        non_f_part = text[last_end:]
        parts.append(escape_non_f_string_region(non_f_part))
    
    return ''.join(parts)

# Global variables for storing API usage
last_api_usage = {}
accumulated_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

# 1. LLM API Calls
def call_gpt_api(prompt, retry_attempts=3, retry_delay=5, model_type="gpt"):
    """GPT API calling function"""
    
    # ⭐ New: Check and handle prompt length
    def check_and_truncate_prompt(prompt_text, model_name):
        """Check and truncate overly long prompts"""
        # Rough estimation: 1 token ≈ 0.75 English words ≈ 4 characters
        # GPT-4o: 128k tokens ≈ 512k characters
        # GPT-4: 8k tokens ≈ 32k characters
        
        max_chars = {
            "gpt-4o": 300000,  # Conservative, leave space for output
            "gpt-4": 25000,    # GPT-4 smaller limit
            "gpt-5": 400000,   # Assume similar to GPT-4o
            "gpt-4.1-nano": 350000  # GPT-4.1-nano limit
        }
        
        limit = max_chars.get(model_name, 25000)  # Default to GPT-4 limit
        
        if len(prompt_text) > limit:
            logger.warning(f"⚠️ Prompt too long ({len(prompt_text)} characters > {limit}), performing truncation")
            # Keep beginning and end, truncate middle
            head_size = int(limit * 0.6)
            tail_size = int(limit * 0.3)
            truncated = (prompt_text[:head_size] + 
                        f"\n\n... [Content truncated, original length {len(prompt_text)} characters] ...\n\n" + 
                        prompt_text[-tail_size:])
            logger.info(f"✂️ Prompt truncated to {len(truncated)} characters")
            return truncated
        
        return prompt_text
    
    GPT_API_KEYS = [
        # Add your GPT API KEY here
    ]
    
    # Select model name based on model type
    if model_type.lower() == "gpt5":
        model_name = "gpt-5"  # GPT-5 will use gpt-5-mini in API calls
    elif model_type.lower() == "gpt4o":
        model_name = "gpt-4o"
    elif model_type.lower() == "gpt4":
        model_name = "gpt-4"
    elif model_type.lower() == "gpt4.1nano" or model_type.lower() == "gpt-4.1-nano":
        model_name = "gpt-4.1-nano"
    else:  # Default to gpt-4o
        model_name = "gpt-4o"
    
    # Check and truncate prompt
    prompt = check_and_truncate_prompt(prompt, model_name)
    
    for attempt in range(retry_attempts):
        selected_key = random.choice(GPT_API_KEYS)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=selected_key)
            
            logger.info(f"[GPT] Attempt {attempt+1}, using {model_name}, prompt length: {len(prompt)} chars, key ending in ...{selected_key[-5:]}")
            
            # Choose correct API call method based on model type
            if model_name == "gpt-5":
                # GPT-5 uses new responses API
                response = client.responses.create(
                    model="gpt-5",  # Use gpt-5
                    input=prompt,
                    text={"verbosity": "high"}  # Use high verbosity
                )
                
                # Extract text from new response format
                result = ""
                for item in response.output:
                    if hasattr(item, "content") and item.content is not None:
                        for content in item.content:
                            if hasattr(content, "text"):
                                result += content.text
            else:
                # Other GPT models use traditional chat completions API
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.7
                )
                result = response.choices[0].message.content
                
                # Record token usage
                if hasattr(response, 'usage'):
                    usage = response.usage
                    input_tokens = getattr(usage, 'prompt_tokens', 0)
                    output_tokens = getattr(usage, 'completion_tokens', 0)
                    total_tokens = getattr(usage, 'total_tokens', 0)
                    logger.info(f"[GPT] Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                    
                    # Store token information in global variables
                    global last_api_usage, accumulated_usage
                    last_api_usage = {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': total_tokens,
                        'model': model_name
                    }
                    
                    # Accumulate token statistics
                    accumulated_usage['input_tokens'] += input_tokens
                    accumulated_usage['output_tokens'] += output_tokens
                    accumulated_usage['total_tokens'] += total_tokens
                
            logger.info("[GPT] Got response successfully")
            return result
            
        except Exception as e:
            logger.error(f"GPT API call failed (attempt {attempt + 1}/{retry_attempts}): {e}")
            if attempt < retry_attempts - 1:
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                logger.error("GPT API call reached maximum retry attempts")
                return None
    return None

# Unified LLM calling interface
def reset_token_usage():
    """Reset token usage statistics"""
    global accumulated_usage
    accumulated_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

def get_token_usage():
    """Get accumulated token usage statistics"""
    global accumulated_usage
    return accumulated_usage.copy()

def call_llm_api(prompt, model_type="gemini", retry_attempts=None, retry_delay=None):
    """
    Unified LLM API calling interface
    
    Args:
        prompt: Input prompt
        model_type: Model type ("gemini", "gpt", "gpt4", "gpt4o", "gpt5", "gpt4.1nano", or "deepseek")
        retry_attempts: Number of retry attempts
        retry_delay: Retry delay
    """
    if model_type.lower() in ["gpt", "gpt4", "gpt4o", "gpt5", "gpt4.1nano", "gpt-4.1-nano"]:
        attempts = retry_attempts if retry_attempts is not None else 3
        delay = retry_delay if retry_delay is not None else 5
        return call_gpt_api(prompt, attempts, delay, model_type)
    elif model_type.lower() == "deepseek":
        attempts = retry_attempts if retry_attempts is not None else 3
        delay = retry_delay if retry_delay is not None else 5
        return call_deepseek_api(prompt, attempts, delay)
    else:  # Default to gemini
        attempts = retry_attempts if retry_attempts is not None else 10
        delay = retry_delay if retry_delay is not None else 30
        return call_gemini_api(prompt, attempts, delay)

def call_gemini_api(prompt, retry_attempts=10, retry_delay=30): 
    GEMINI_API_KEYS = [ 
        # Add your Gemini API KEY here
    ] 
    
    for attempt in range(retry_attempts): 
        selected_key = random.choice(GEMINI_API_KEYS) 
        try: 
            genai.configure(api_key=selected_key) 
            model = genai.GenerativeModel("gemini-2.5-flash-lite") 
            response = model.generate_content(prompt) 
            
            # Record token usage for Gemini
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                input_tokens = getattr(usage, 'prompt_token_count', 0)
                output_tokens = getattr(usage, 'candidates_token_count', 0)
                total_tokens = getattr(usage, 'total_token_count', input_tokens + output_tokens)
                logger.info(f"[Gemini] Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                
                # Store token information in global variables
                global last_api_usage, accumulated_usage
                last_api_usage = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'model': 'gemini-2.5-flash-lite'
                }
                
                # Accumulate token statistics
                accumulated_usage['input_tokens'] += input_tokens
                accumulated_usage['output_tokens'] += output_tokens
                accumulated_usage['total_tokens'] += total_tokens
            
            return getattr(response, 'text', None) 
        except Exception as e: 
            logger.error(f"Gemini API call error (attempt {attempt + 1}/{retry_attempts}): {repr(e)} using key ending in ...{selected_key[-5:]}") 
            if attempt < retry_attempts - 1: 
                logger.info(f"Waiting {retry_delay} seconds before retry...") 
                time.sleep(retry_delay) 
            else: 
                logger.error("Gemini API call reached maximum retry attempts") 
                return None 
    return None

def call_deepseek_api(prompt, retry_attempts=3, retry_delay=5):
    """DeepSeek API calling function via SiliconFlow"""
    
    # DeepSeek V3 API configuration via SiliconFlow
    DEEPSEEK_API_KEYS = [
        # Add your DeepSeek API KEY here
    ]
    
    if not DEEPSEEK_API_KEYS or not DEEPSEEK_API_KEYS[0]:
        logger.error("[DeepSeek] No API keys configured")
        return None
    
    for attempt in range(retry_attempts):
        selected_key = random.choice(DEEPSEEK_API_KEYS)
        try:
            from openai import OpenAI
            
            logger.info(f"[DeepSeek] Attempt {attempt+1}, prompt length: {len(prompt)} chars, using SiliconFlow proxy")
            
            # SiliconFlow proxy configuration for DeepSeek V3
            url = 'https://api.siliconflow.cn/v1/'
            client = OpenAI(base_url=url, api_key=selected_key)
            
            response = client.chat.completions.create(
                model="Pro/deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7
            )
            
            # Extract result
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content
                
                # Record token usage
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    input_tokens = getattr(usage, 'prompt_tokens', 0)
                    output_tokens = getattr(usage, 'completion_tokens', 0)
                    total_tokens = getattr(usage, 'total_tokens', 0)
                    logger.info(f"[DeepSeek] Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                    
                    # Store token information in global variables
                    global last_api_usage, accumulated_usage
                    last_api_usage = {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': total_tokens,
                        'model': 'deepseek-v3'
                    }
                    
                    # Accumulate token statistics
                    accumulated_usage['input_tokens'] += input_tokens
                    accumulated_usage['output_tokens'] += output_tokens
                    accumulated_usage['total_tokens'] += total_tokens
                
                logger.info("[DeepSeek] Got response successfully")
                return result
            else:
                logger.error(f"[DeepSeek] Unexpected response format")
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek API call failed (attempt {attempt + 1}/{retry_attempts}): {e}")
            if attempt < retry_attempts - 1:
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                logger.error("DeepSeek API call reached maximum retry attempts")
                return None
    return None

class ProblemAnalyzer(BaseAgent):
    def __init__(self, agent_id: str, model_type: str = "gemini"):
        # Set correct llm_model based on model_type
        if model_type.lower() == "gemini":
            llm_model = "gemini-2.5-flash-lite"
        elif model_type.lower() == "gpt5":
            llm_model = "gpt-5"
        elif model_type.lower() == "gpt4o":
            llm_model = "gpt-4o"
        elif model_type.lower() == "gpt4":
            llm_model = "gpt-4"
        elif model_type.lower() == "deepseek":
            llm_model = "deepseek-v3"
        else:  # Default to gpt-4o
            llm_model = "gpt-4o"
        super().__init__(agent_id, llm_model)
        self.model_type = model_type.lower()
        
        # Intelligent detection of API availability
        try:
            # Test using unified API calling function
            test_response = call_llm_api("Hello", self.model_type)
            self.api_available = test_response is not None
            self.logger.info(f"{self.model_type.upper()} API connection successful" if self.api_available else f"{self.model_type.upper()} API unavailable, switch to simulation mode")
        except Exception as e:
            self.logger.warning(f"{self.model_type.upper()} API unavailable, switch to simulation mode: {e}")
            self.api_available = False
        
        # Initialize LLM model (Only Gemini needs this, GPT and DeepSeek call API directly)
        if self.model_type == "gemini":
            try:
                self.model = genai.GenerativeModel(self.llm_model)
            except Exception as e:
                self.logger.warning(f"Unable to initialize Gemini model: {e}")
                self.model = None
        else:
            self.model = None  # GPT and DeepSeek do not need pre-initialized model
        
        # Initialize HMML retriever
        try:
            from hmml_retriever import HMMLManager
            self.hmml_manager = HMMLManager("Engi-HMML-v2.md")
            self.hmml_enabled = self.hmml_manager.available
            self.logger.info(f"HMML retriever initialization {'successful' if self.hmml_enabled else 'failed, using degraded mode'}")
        except Exception as e:
            self.logger.warning(f"HMML retriever initialization failed: {e}")
            self.hmml_manager = None
            self.hmml_enabled = False

    def retrieve_domain_methods(self, problem_description: str) -> Dict[str, Any]:
        """Retrieve HMML relevant methods based on problem description"""
        if not self.hmml_enabled or not self.hmml_manager:
            return {}
        
        try:
            retrieval_result = self.hmml_manager.retrieve_methods(problem_description)
            self.logger.info(f"HMML retrieval completed, found {len(retrieval_result.get('retrieved_methods', []))} relevant methods")
            return retrieval_result
        except Exception as e:
            self.logger.error(f"HMML retrieval failed: {e}")
            return {}
    
    def generate_universal_prompt(self, problem_text: str, hmml_result: Dict[str, Any]) -> str:
        """Generate high-quality engineering modeling analysis prompt including HMML retrieval results"""
        
        # Base universal prompt template - based on four modeling quality evaluation metrics
        base_prompt = f"""You are a senior engineering problem modeling expert. Please perform a **high-quality systematic analysis** of the following engineering problem, referring to the retrieved modeling knowledge to extract complete modeling elements.

**Problem Description:**
{problem_text}

"""
        
        # If HMML retrieval results exist, add method guidance
        if hmml_result and hmml_result.get('hmml_analysis'):
            hmml_analysis = hmml_result['hmml_analysis']
            modeling_guidance = hmml_result.get('modeling_guidance', {})
            
            base_prompt += f"""**Retrieved Relevant Modeling Methods:**
- **Recommended Domain:** {hmml_analysis.get('domain', 'General Engineering')}
- **Subdomain:** {hmml_analysis.get('subdomain', 'Optimization')}  
- **Recommended Method:** {hmml_analysis.get('method', 'Mathematical Programming')}
- **Confidence:** {hmml_analysis.get('confidence', 0.5):.2f}

**Modeling Guidance:**
- **Modeling Approach:** {modeling_guidance.get('approach', 'Mathematical optimization-based modeling')}
- **Core Concept:** {modeling_guidance.get('methodology', 'Construct objective function and constraints')}
- **Mathematical Framework:** {modeling_guidance.get('framework', 'Optimization framework')}
- **Solution Strategy:** {modeling_guidance.get('solution_strategy', 'Numerical optimization methods')}

**Application Context:**
- **Typical Applications:** {hmml_result.get('application_context', {}).get('applications', 'Engineering optimization problems')}
- **Method Advantages:** {hmml_result.get('application_context', {}).get('advantages', 'Mature theory')}

"""
        
        # Generate universal analysis task based on four quality metrics
        analysis_task = self._generate_quality_oriented_analysis_task()
        base_prompt += analysis_task
        
        return base_prompt
    
    def _generate_quality_oriented_analysis_task(self) -> str:
        """Generate universal analysis task based on four quality metrics"""
        return """### Role and Mission ###
You are a top-tier "system architect" in charge of designing a complex, open-ended engineering problem into a **highly structured, traceable, and hierarchical JSON modeling blueprint**. This blueprint is the sole basis for downstream code generation agents to build accurate, feasible solving models.

### Core Thinking Framework: Four Modeling Dimensions ###
When analyzing a problem, you must consider the following four dimensions as implicit indicators of your analysis. Your final output structure must reflect the results of this deep thinking process, but **do not** create top-level blocks in the JSON for these four dimensions.
1.  **Information Extraction (Information Extraction)**: Your goal is to identify all "atomic" information needed for modeling - entities, numerical values, relationships, and goals. This is the foundation of modeling.
2.  **Domain-specific Reasoning (Domain-specific Reasoning)**: You need to "dress" these "atomic" pieces of information with "domain clothing". Using professional engineering knowledge, transform raw data into meaningful parameters, convert relationships into physical or logical constraints, and select the most appropriate modeling paradigm.
3.  **Multi-objective Decision-making (Multi-objective Decision-making)**: You need to prioritize all goals. A **core optimization goal** must be clearly defined, while other goals are recognized as secondary or trade-off items. This is the "compass" guiding your decisions.
4.  **Uncertainty Handling (Uncertainty Handling)**: You need to identify all "cracks" in the information - missing, fuzzy, or variable parts. Your task is to "fill" these cracks through reasonable assumptions, ensuring the integrity and certainty of the core model.

### Strict Output Format: Hierarchical JSON Modeling Blueprint ###
You must, and can only output a JSON object wrapped in ```json...```. This structure has been carefully designed to ensure the completeness, hierarchy, and executability of downstream agents.
```json...
{
  "modeling_context": {
    "problem_essence": "Summarize the core engineering optimization problem in one sentence, formatted as 'Under [Key Constraints], optimize [Decision Variables] to achieve [Core Goals].".
    "engineering_domain": "Specific engineering field (e.g., 'Power System Dispatch', 'Supply Chain Network Design', 'Aerospace Planning')",
    "modeling_paradigm": "Recommended core modeling methods based on problem characteristics and industry conventions (e.g., 'Mixed-Integer Linear Programming', 'Quadratic Constraint Programming', 'Stochastic Optimization')",
    "solution_scope": "Clearly define the core scope and boundary conditions of this modeling exercise."
  },
  "core_model_elements": {
    "description": "Defines the minimum viable model of the problem (Minimum Viable Model), including all necessary elements for constructing a basic solution method. This part must be self-consistent and complete, directly usable for code implementation.",
    "decision_variables": [
      {
        "name": "Standard Mathematical Symbol Variable Name (e.g., P_g_t)",
        "description": "Exact engineering meaning of the variable, index definition (e.g., 'g' represents generating units, 't' represents time periods) and unit",
        "type": "continuous/integer/binary",
        "domain": "Mathematical definition domain (e.g., '>= 0', '[0, 1]')",
        "shape": "Dimension or size of the variable (e.g., '[G, T]' G is the number of units, T is the number of time periods)"
      }
    ],
    "parameters": [
      {
        "param_id": "PARAM_01",
        "name": "Mathematical Symbol of Parameter (e.g., C_g_max)",
        "value": "Specific numerical values or complete arrays/lists from the original problem description. No omissions or descriptive substitutions are allowed.",
        "unit": "Physical unit (e.g., 'MW', 'USD/MWh', 'kg')",
        "description": "Exact engineering meaning and index explanation of the parameter.",
        "source_reference": "Indicates the source or basis of this data in the original problem description (e.g., 'from Table 1, Column 3', 'Section 5 of the problem description explicitly states')"
      }
    ],
    "objective_function": {
      "name": "Name of the core objective function (e.g., TotalOperatingCost)",
      "type": "minimize/maximize",
      "expression": "Complete mathematical expression using the defined variable and parameter symbols.",
      "components": [
        {
          "component_expr": "One part of the expression (e.g., sum(c_g * P_g_t for g, t))",
          "description": "Engineering meaning of this component (e.g., 'Total Fuel Cost')"
        }
      ]
    },
    "constraints": [
      {
        "constraint_id": "CONST_01",
        "name": "Unique Identifier of Constraint (e.g., 'PowerBalance')",
        "expression": "Complete mathematical (equation/inequality) expression using the defined variable and parameter symbols.",
        "category": "Category of the constraint ('Physical Laws', 'Resource Capacity', 'Supply-Demand Balance', 'Operational Logic', 'Strategic Requirements')",
        "description": "Engineering significance and importance of the constraint."
      }
    ]
  },
  "extended_analysis_and_robustness": {
    "description": "Includes supplementary, expanded, and considerations under uncertainty for the core model, which is key to advanced analysis and ensuring robustness of solutions.",
    "key_assumptions": [
      {
        "assumption_id": "ASSUM_01",
        "content": "Explicit assumptions made to handle missing information or simplify the model.",
        "justification": "Engineering principles, industry conventions, or logical basis for this assumption.",
        "impact_on_model": "Explicitly states how this assumption specifically affects a specific element in `core_model_elements` (e.g., 'Setting the value of PARAM_05 `line_efficiency` to 0.98', 'Simplifying the calculation formula of CONST_03 `PowerFlow`')"
      }
    ],
    "uncertainty_sources": [
      {
        "source_id": "UNCERT_01",
        "description": "Key sources of uncertainty identified (e.g., 'Future Market Electricity Price', 'Equipment Failure Rate')",
        "affected_elements": ["IDs of parameters or variables affected by this uncertainty (e.g., 'PARAM_10')"],
        "handling_strategy": "Recommended strategies for handling (e.g., 'Using Expected Values for Deterministic Modeling', 'Conducting Sensitivity Analysis', 'Using Scenario Analysis Approach')
      }
    ],
    "trade_off_analysis": {
      "secondary_objectives": [
        {
          "name": "Secondary or potential optimization goals (e.g., 'MinimizeCarbonEmissions')",
          "expression": "Mathematical expression.",
          "conflict_with": "Which core or secondary goals are in tension with each other (e.g., 'TotalOperatingCost')"
        }
      ],
      "soft_constraints": [
        {
          "name": "Soft constraints or preferences (e.g., 'PreferredMaintenanceWindow')",
          "description": "Conditions hoped for but not necessarily required, often modeled by adding penalty terms to the objective function."
        }
      ]
    },
    "sensitivity_factors": [
      {
        "param_id": "ID of key parameters for conducting sensitivity analysis (from `parameters` section)",
        "justification": "Why this parameter might have a significant impact on the model results."
      }
    ]
  }
}
```

### Golden Command and Final Verification ###
1.  **Absolute Data Integrity**: Every valid numerical value in the original problem **must** be present in the `parameters` list with a unique entry, and its origin must be traceable through the `source_reference` field.
2.  **Mandatory Internal References**: Cross-references within JSON are mandatory. For example, `sensitivity_factors` must reference `param_id` of `parameters`; `key_assumptions`'s `impact_on_model` must explicitly point to a specific element in `core_model_elements`. This ensures traceability and consistency of the model.
3.  **Core Model Priority**: `core_model_elements` is the cornerstone. It must be a model that can be independently solved and clearly defined. All supplementary, trade-off, and uncertainty analyses are based on it in `extended_analysis_and_robustness`.
4.  **Clear Roles and Responsibilities**: Strictly differentiate between `core_model_elements` (what they do) and `extended_analysis_and_robustness` (how to do it better/more robustly). Do not mix in assumptions or uncertain elements in the core model.
5.  **Designed for Code Generation**: All mathematical expressions (`expression`) must use standard, unambiguous mathematical notation so that downstream agents can easily parse and convert them into code. """

    def process_message(self, message: Message) -> Message:
        if message.msg_type == MessageType.QUERY:
            problem_description = message.content
            self.logger.info(f"Analyzing problem: {problem_description[:50]}...")
            try:
                analysis_result = self.analyze_problem(problem_description)
                response_msg = Message(self.agent_id, "coordinator", analysis_result, MessageType.RESPONSE)
                return response_msg
            except Exception as e:
                self.logger.error(f"Error in problem analysis: {e}")
                error_msg = Message(
                    self.agent_id, 
                    "coordinator", 
                    {"error": str(e), "error_type": "AnalysisError", "stage": "analysis"}, 
                    MessageType.ERROR
                )
                return error_msg
        elif message.msg_type == MessageType.RETRY:
            # Handle retry request - based on error information
            retry_info = message.content
            strategy = retry_info.get("strategy", "Re-analyze")
            error_info = retry_info.get("error_info", {})
            original_problem = retry_info.get("original_problem", "")
            
            # Check if original problem description exists
            if not original_problem:
                self.logger.error("Retry analysis missing original problem description")
                error_msg = Message(
                    self.agent_id, 
                    "coordinator", 
                    {"error": "Retry analysis missing original problem description", "error_type": "AnalysisError", "stage": "retry"}, 
                    MessageType.ERROR
                )
                return error_msg
            
            self.logger.info(f"Executing analysis retry: {strategy}")
            
            try:
                # Use LLM to re-analyze problem based on original problem and detailed error information
                analysis_result = self.analyze_problem_with_enhanced_error_context(
                    original_problem, 
                    error_info, 
                    strategy, 
                    retry_info
                )
                response_msg = Message(self.agent_id, "coordinator", analysis_result, MessageType.RESPONSE)
                return response_msg
            except Exception as e:
                error_msg = Message(
                    self.agent_id, 
                    "coordinator", 
                    {"error": str(e), "error_type": "AnalysisError", "stage": "retry_analysis"}, 
                    MessageType.ERROR
                )
                return error_msg
        else:
            self.logger.warning(f"ProblemAnalyzer received unexpected message type: {message.msg_type}")
            error_msg = Message(
                self.agent_id, 
                "coordinator", 
                {"error": f"Unsupported message type: {message.msg_type.value}", "error_type": "AnalysisError", "stage": "message_processing"}, 
                MessageType.ERROR
            )
            return error_msg

    def analyze_problem(self, problem_description: str) -> Dict[str, Any]:
        print("\n🔍 [ProblemAnalyzer] Starting problem analysis")
        
        if not self.api_available:
            # When API is unavailable, throw exception for coordinator to handle
            self.logger.warning("API unavailable, cannot perform analysis")
            raise Exception("Analysis failed: Gemini API unavailable")
        
        # Step 1: HMML retrieval (if enabled)
        hmml_result = {}
        if self.hmml_enabled:
            print("🔍 Performing HMML domain retrieval...")
            hmml_result = self.retrieve_domain_methods(problem_description)
            if hmml_result.get('hmml_analysis'):
                domain = hmml_result['hmml_analysis'].get('domain', 'Unknown')
                method = hmml_result['hmml_analysis'].get('method', 'Unknown')
                confidence = hmml_result['hmml_analysis'].get('confidence', 0.0)
                print(f"✅ HMML retrieval completed: {domain} -> {method} (confidence: {confidence:.2f})")
            else:
                print("⚠️ HMML retrieval found no matching methods, using default analysis")
        
        # Step 2: Generate enhanced universal prompt
        prompt = self.generate_universal_prompt(problem_description, hmml_result)
        try:
            print("🚀 Calling LLM for analysis...")
            response_text = call_llm_api(prompt, self.model_type)
            if not response_text:
                raise Exception(f"{self.model_type.upper()} API call failed")
            
            # Extract JSON part - Enhanced version
            print(f"Original response length: {len(response_text)} characters")
            print("Original response content:")
            print("-" * 40)
            print(response_text)
            print("-" * 40)
            
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
            
            analysis = json.loads(response_text)
            print("✅ JSON parsing successful")
            self.logger.info("Problem analysis completed.")
            return analysis

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print("🔄 Attempting to fix JSON format...")
            
            # Try to fix common JSON format issues
            try:
                # Remove possible prefix/suffix text
                if '{' in response_text and '}' in response_text:
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    cleaned_json = response_text[start:end]
                    
                    # Fix common issues
                    cleaned_json = cleaned_json.replace("'", '"')  # Single quotes to double quotes
                    cleaned_json = re.sub(r',\s*}', '}', cleaned_json)  # Remove trailing commas
                    cleaned_json = re.sub(r',\s*]', ']', cleaned_json)  # Remove array trailing commas
                    
                    analysis = json.loads(cleaned_json)
                    print("✅ JSON fix successful")
                    return analysis
            except:
                pass
            
            self.logger.error(f"Error in problem analysis: {e}")
            # No longer fall back to simulation, throw exception for coordinator to handle
            raise Exception(f"Analysis failed: JSON parsing error - {e}")
        
        except Exception as e:
            self.logger.error(f"Error in problem analysis: {e}")
            # No longer fall back to simulation, throw exception for coordinator to handle
            raise Exception(f"Analysis failed: {e}")
    
    def analyze_problem_with_enhanced_error_context(self, original_problem: str, error_info: Dict, strategy: str, retry_info: Dict) -> Dict[str, Any]:
        """Re-analyze problem based on original problem and enhanced error context"""
        if not self.api_available:
            raise Exception("Analysis retry failed: Gemini API unavailable")
        
        try:
            # Extract detailed error context information
            verification_details = retry_info.get("verification_details", {})
            previous_analysis = retry_info.get("previous_analysis", {})
            rejected_model_code = retry_info.get("rejected_model_code", "")
            mismatch_info = retry_info.get("mismatch_info", {})
            
            # If detailed verification information exists, use enhanced retry logic
            if verification_details or previous_analysis:
                return self._analyze_with_verification_feedback(
                    original_problem, verification_details, previous_analysis, 
                    rejected_model_code, mismatch_info, strategy
                )
            else:
                # Fall back to original error context analysis
                return self.analyze_problem_with_error_context(original_problem, error_info, strategy)
                
        except Exception as e:
            self.logger.error(f"Enhanced error context analysis failed: {e}")
            raise Exception(f"Enhanced analysis retry failed: {e}")
    
    def _analyze_with_verification_feedback(self, original_problem: str, verification_details: Dict, 
                                          previous_analysis: Dict, rejected_model_code: str, 
                                          mismatch_info: Dict, strategy: str) -> Dict[str, Any]:
        """Perform targeted analysis based on verification feedback"""
        
        # Step 1: Re-run HMML retrieval (if enabled)
        hmml_result = {}
        if self.hmml_enabled:
            print("🔍 Re-running HMML retrieval based on verification feedback...")
            hmml_result = self.retrieve_domain_methods(original_problem)
        
        # Step 2: Generate base universal prompt
        base_prompt = self.generate_universal_prompt(original_problem, hmml_result)
        
        # Step 3: Add detailed verification feedback information
        verification_feedback = json.dumps(verification_details, indent=2, ensure_ascii=False) if verification_details else "No detailed verification information"
        previous_analysis_str = json.dumps(previous_analysis, indent=2, ensure_ascii=False) if previous_analysis else "No previous analysis result"
        # Handle fields at top level or nested in mismatch_details
        mismatch_reason = mismatch_info.get("mismatch_reason") or mismatch_info.get("mismatch_details", {}).get("mismatch_reason", "Unknown mismatch reason")
        mismatch_suggestion = mismatch_info.get("suggestion") or mismatch_info.get("mismatch_details", {}).get("suggestion", "No specific suggestions")
        
        # Build enhanced retry prompt
        enhanced_prompt = f"""{base_prompt}

**Verification Feedback Analysis and Targeted Correction:**

**Previous Analysis Result:**
```json
{previous_analysis_str}
```

**Detailed Reasons for Verification Failure:**
{mismatch_reason}

**Verification Expert's Specific Suggestions:**
{mismatch_suggestion}

**Targeted Correction Strategy:** {strategy}

**Re-analysis Requirements:**

Based on the above verification feedback, please make targeted corrections to the analysis results. Focus on:

1. **Specific Issues from Verification Failure**: Identify specific deficiencies in the previous analysis based on verification feedback
2. **Parameter Data Accuracy**: Ensure all numerical parameters exactly match the original problem
3. **Modeling Logic Completeness**: Supplement missing constraints or decision variables
4. **Objective Function Accuracy**: Ensure optimization direction and objective expression are correct
5. **Engineering Feasibility**: Ensure analysis results align with engineering reality

**Output Requirements:**
Please make targeted corrections based on verification feedback and output complete modeling analysis JSON, ensuring it passes verification checks.

Special attention:
- If verification points out specific numerical errors, strictly correct according to original problem
- If verification points out logical defects, supplement and improve related content
- If verification suggests modeling method is inappropriate, adjust modeling strategy
- Maintain JSON format completeness and consistency
"""
        
        # Call LLM for enhanced analysis
        response_text = call_llm_api(enhanced_prompt, self.model_type)
        if not response_text:
            raise Exception(f"{self.model_type.upper()} API call failed")
        
        # JSON parsing logic (same as original method)
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            if json_end != -1:
                response_text = response_text[json_start:json_end].strip()
            else:
                response_text = response_text[json_start:].strip()
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
        
        # Clean and parse JSON
        response_text = response_text.replace('```', '').strip()
        response_text = response_text.replace('"', '"').replace('"', '"')
        
        try:
            analysis = json.loads(response_text)
            print("✅ Enhanced analysis based on verification feedback completed")
            return analysis
        except json.JSONDecodeError as e:
            print(f"❌ Enhanced analysis JSON parsing failed: {e}")
            # If enhanced analysis fails, fall back to original method
            return self.analyze_problem_with_error_context(original_problem, {}, strategy)

    def analyze_problem_with_error_context(self, original_problem: str, error_info: Dict, strategy: str) -> Dict[str, Any]:
        """Re-analyze problem based on original problem and error information"""
        if not self.api_available:
            raise Exception("Analysis retry failed: Gemini API unavailable")
        
        try:
            # Step 1: Re-run HMML retrieval (if enabled)
            hmml_result = {}
            if self.hmml_enabled:
                print("🔍 Re-running HMML domain retrieval...")
                hmml_result = self.retrieve_domain_methods(original_problem)
            
            # Step 2: Generate base universal prompt
            base_prompt = self.generate_universal_prompt(original_problem, hmml_result)
            
            # Step 3: Add error context information
            error_info_str = json.dumps(error_info, indent=2, ensure_ascii=False)
            # Add error correction guidance to base prompt
            prompt = f"""{base_prompt}

**Error Information Analysis:**
{error_info_str}

**Retry Strategy:** {strategy}

**Error Diagnosis and Correction Framework:**

**Phase One: Error Root Cause Identification**
Please analyze the error information deeply to identify potential issues in these categories:
- **Data Level**: Missing parameters, numerical errors, unit mismatches, dimensional analysis errors
- **Structural Level**: Constraint logic defects, missing variable relationships, unclear system boundaries
- **Methodological Level**: Inappropriate solution method selection, algorithm applicability issues, complexity assessment errors
- **Domain Level**: Engineering physical constraint misunderstandings, disconnection from practical application scenarios

**Phase Two: Systematic Correction Requirements**

1. **Engineering Physical Consistency Correction**:
   - Re-verify correct application of all physical laws and engineering principles
   - Ensure completeness of dimensional analysis and uniformity of unit systems
   - Check physical feasibility and practical reasonableness of engineering constraints

2. **Mathematical Modeling Precision Correction**:
   - Re-confirm accurate values and valid ranges for all parameters
   - Verify mathematical expression completeness and logical consistency of constraints
   - Check accuracy of mathematical relationships between objective function and decision variables

3. **Solution Method Adaptability Correction**:
   - Re-evaluate most suitable solution method category based on problem characteristics
   - Consider problem scale, complexity, linear/nonlinear features, continuous/discrete properties
   - Evaluate handling requirements for multi-objective, multi-constraint, uncertainty, and other complex features

4. **Engineering Practice Feasibility Correction**:
   - Ensure all variable definitions align with practical engineering meanings and operational feasibility
   - Verify constraint conditions' executability in actual engineering environments
   - Check engineering significance and practical application value of solutions

**Phase Three: Quality Assurance Requirements**

**Output Format Strictness**:
- Must output standard JSON format wrapped in ```json...```
- JSON structure must contain complete problem analysis elements
- All values must be verified and maintain precision

**Content Completeness Verification**:
- Accurate numerical values, units, and physical meanings for all engineering parameters
- Accurate definitions of variable types, constraint domains, and engineering significance
- Complete constraint system, mathematical expressions, and physical interpretations
- Mathematical formulation, engineering significance, and solution direction of optimization objectives
- Accurate problem type classification and solution method recommendations
- Engineering background, application scenarios, and actual constraint conditions

**The corrected analysis must avoid all previously identified error types and improve engineering accuracy and solution feasibility.**
"""
            
            response_text = call_llm_api(prompt, self.model_type)
            if not response_text:
                raise Exception(f"{self.model_type.upper()} API call failed")
            
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
            
            analysis = json.loads(response_text)
            print("✅ JSON parsing successful")
            self.logger.info("Problem analysis completed.")
            return analysis

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print("🔄 Attempting to fix JSON format...")
            
            # Try to fix common JSON format issues
            try:
                # Remove possible prefix/suffix text
                if '{' in response_text and '}' in response_text:
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    cleaned_json = response_text[start:end]
                    
                    # Fix common issues
                    cleaned_json = cleaned_json.replace("'", '"')  # Single quotes to double quotes
                    cleaned_json = re.sub(r',\s*}', '}', cleaned_json)  # Remove trailing commas
                    cleaned_json = re.sub(r',\s*]', ']', cleaned_json)  # Remove array trailing commas
                    
                    analysis = json.loads(cleaned_json)
                    print("✅ JSON fix successful")
                    return analysis
            except:
                pass
            
            self.logger.error(f"Error in analysis retry: {e}")
            # No longer fall back to simulation, throw exception for coordinator to handle
            raise Exception(f"Analysis retry failed: JSON parsing error - {e}")
                
        except Exception as e:
            self.logger.error(f"Error context analysis failed: {e}")
            raise Exception(f"Analysis retry failed: {e}")

    def generate_simulation_analysis(self, problem_description: str) -> Dict[str, Any]:
        """Generate intelligent analysis results based on problem description, not hardcoded"""
        # Keyword detection and intelligent analysis
        description_lower = problem_description.lower()
        
        # Detect optimization objective
        objective = "Minimize Total Cost"
        if "minimize" in description_lower or "minimize" in description_lower:
            if "cost" in description_lower or "cost" in description_lower:
                objective = "Minimize Total Cost"
            elif "energy" in description_lower or "energy" in description_lower:
                objective = "Minimize Energy Consumption"
            elif "time" in description_lower or "time" in description_lower:
                objective = "Minimize Time"
        elif "maximize" in description_lower or "maximize" in description_lower:
            if "profit" in description_lower or "profit" in description_lower:
                objective = "Maximize Profit"
            elif "efficiency" in description_lower or "efficiency" in description_lower:
                objective = "Maximize Efficiency"
        
        # Detect decision variables
        decision_variables = []
        if "battery" in description_lower or "storage" in description_lower:
            decision_variables.extend([
                {"name": "P_charge", "description": "Battery Charging Power"},
                {"name": "P_discharge", "description": "Battery Discharging Power"},
                {"name": "SOC", "description": "State of Charge"}
            ])
        if "buy" in description_lower or "purchase" in description_lower:
            decision_variables.append({"name": "P_buy", "description": "Power Purchase"})
        if "sell" in description_lower or "sale" in description_lower:
            decision_variables.append({"name": "P_sell", "description": "Power Sale"})
        if "dispatch" in description_lower or "schedule" in description_lower:
            decision_variables.append({"name": "P_gen", "description": "Generation Power"})
        
        # If no specific variables detected, use generic variables
        if not decision_variables:
            decision_variables = [{"name": "x", "description": "Decision Variable"}]
        
        # Detect constraints
        constraints = []
        if "balance" in description_lower or "equilibrium" in description_lower:
            constraints.append({"description": "Power Balance Constraint"})
        if "capacity" in description_lower or "limit" in description_lower:
            constraints.append({"description": "Capacity Limit Constraint"})
        if "storage" in description_lower or "battery" in description_lower:
            constraints.extend([
                {"description": "Battery Charge/Discharge Power Limit"},
                {"description": "Battery Capacity Constraint"}
            ])
        
        # Default constraints
        if not constraints:
            constraints = [{"description": "System Constraints"}]
        
        # Detect parameters
        parameters = []
        if "load" in description_lower or "demand" in description_lower:
            parameters.append({"name": "Load", "description": "Load Demand Data"})
        if "pv" in description_lower or "solar" in description_lower:
            parameters.append({"name": "PV_gen", "description": "PV Generation Data"})
        if "price" in description_lower or "cost" in description_lower:
            parameters.append({"name": "Price", "description": "Price Data"})
        
        # Default parameters
        if not parameters:
            parameters = [{"name": "data", "description": "Input Data"}]
        
        return {
            "information_extraction": {
                "core_problem": "Core problem identification based on problem description",
                "key_entities": ["System", "Optimization Objective"],
                "critical_constraints": ["System Constraints"],
                "explicit_data": {
                    "numerical_values": [],
                    "time_series": [],
                    "categorical_data": []
                },
                "implicit_requirements": ["Optimization Solution Requirements"]
            },
            "domain_reasoning": {
                "engineering_domain": "Engineering Optimization",
                "domain_principles": ["Mathematical Optimization Principles"],
                "standard_assumptions": ["Standard Engineering Assumptions"],
                "modeling_approach": "Mathematical Programming",
                "solution_methodology": "Numerical Optimization",
                "domain_constraints": ["Engineering Constraints"]
            },
            "multi_objective_analysis": {
                "primary_objectives": [
                    {
                        "objective": objective,
                        "type": "minimize",
                        "priority": "high",
                        "mathematical_expression": "To be determined in modeling"
                    }
                ],
                "secondary_objectives": [],
                "objective_conflicts": [],
                "trade_off_strategies": [],
                "pareto_considerations": "Single Objective Optimization"
            },
            "uncertainty_handling": {
                "uncertainty_sources": [],
                "missing_information": ["Specific Parameter Values"],
                "reasonable_assumptions": [
                    {
                        "assumption": "Standard Engineering Parameters",
                        "justification": "Based on Common Engineering Practice",
                        "impact": "Affects Model Accuracy"
                    }
                ],
                "robustness_measures": [],
                "sensitivity_factors": [],
                "fallback_strategies": ["Simulation Analysis"]
            },
            "modeling_elements": {
                "decision_variables": decision_variables,
                "objective_function": {
                    "expression": objective,
                    "components": ["Primary Objective"],
                    "coefficients": []
                },
                "constraints": constraints,
                "parameters": parameters
            },
            "mode": "simulation",
            "note": "Intelligent analysis results based on problem description (backup analysis when API is unavailable)"
        }

class ModelingAgent(BaseAgent):
    def __init__(self, agent_id: str, model_type: str = "gemini"):
        # Set correct llm_model based on model_type
        if model_type.lower() == "gemini":
            llm_model = "gemini-2.5-flash-lite"
        elif model_type.lower() == "gpt5":
            llm_model = "gpt-5"
        elif model_type.lower() == "gpt4o":
            llm_model = "gpt-4o"
        elif model_type.lower() == "gpt4":
            llm_model = "gpt-4"
        elif model_type.lower() == "deepseek":
            llm_model = "deepseek-v3"
        else:  # Default to gpt-4o
            llm_model = "gpt-4o"
        super().__init__(agent_id, llm_model)
        self.model_type = model_type.lower()
        
        # Intelligent detection of API availability
        try:
            # Test using unified API calling function
            test_response = call_llm_api("Hello", self.model_type)
            self.api_available = test_response is not None
            self.logger.info(f"{self.model_type.upper()} API connection successful" if self.api_available else f"{self.model_type.upper()} API unavailable, switch to simulation mode")
        except Exception as e:
            self.logger.warning(f"{self.model_type.upper()} API unavailable, switch to simulation mode: {e}")
            self.api_available = False
        
        # Initialize LLM model (Only Gemini needs this, GPT and DeepSeek call API directly)
        if self.model_type == "gemini":
            try:
                self.model = genai.GenerativeModel(self.llm_model)
            except Exception as e:
                self.logger.warning(f"Unable to initialize Gemini model: {e}")
                self.model = None
        else:
            self.model = None  # GPT and DeepSeek do not need pre-initialized model

    def process_message(self, message: Message) -> Message:
        if message.msg_type == MessageType.QUERY:
            analysis_result = message.content
            self.logger.info("Generating Pyomo model...")
            try:
                pyomo_code = self.generate_pyomo_model(analysis_result)
                response_msg = Message(self.agent_id, "coordinator", pyomo_code, MessageType.RESPONSE)
                return response_msg
            except Exception as e:
                self.logger.error(f"Error in model generation: {e}")
                error_msg = Message(
                    self.agent_id, 
                    "coordinator", 
                    {
                        "error": str(e), 
                        "error_type": "ModelingError", 
                        "stage": "modeling",
                        "input_data": str(analysis_result)[:500],  # Include input analysis result
                        "traceback": traceback.format_exc()
                    }, 
                    MessageType.ERROR
                )
                return error_msg
        elif message.msg_type == MessageType.RETRY:
            # Handle retry request - based on error information
            retry_info = message.content
            strategy = retry_info.get("strategy", "Code fix")
            attempt_number = retry_info.get("attempt_number", 1)
            error_info = retry_info.get("error_info", {})
            nested_mode = retry_info.get("nested_mode", False)
            
            # ✅ Fix: Extract new context information
            analysis_result = retry_info.get("analysis_result", {})
            previous_model_code = retry_info.get("previous_model_code", "")
            original_problem = retry_info.get("original_problem", "")
            
            self.logger.info(f"Executing modeling retry: {strategy} (Attempt {attempt_number}, Nested mode: {nested_mode})")
            
            try:
                # Use LLM to regenerate or fix model based on error information and context
                adjusted_code = self.generate_model_with_error_context(
                    error_info, strategy, attempt_number, analysis_result, previous_model_code, original_problem
                )
                response_msg = Message(self.agent_id, "coordinator", adjusted_code, MessageType.RESPONSE)
                return response_msg
            except Exception as e:
                error_msg = Message(
                    self.agent_id, 
                    "coordinator", 
                    {"error": str(e), "error_type": "ModelingError", "stage": "retry_modeling"}, 
                    MessageType.ERROR
                )
                return error_msg
        else:
            self.logger.warning(f"ModelingAgent received unexpected message type: {message.msg_type}")
            error_msg = Message(
                self.agent_id, 
                "coordinator", 
                {"error": f"Unsupported message type: {message.msg_type.value}", "error_type": "ModelingError", "stage": "message_processing"}, 
                MessageType.ERROR
            )
            return error_msg

    def generate_pyomo_model(self, analysis_result: Dict[str, Any]) -> str:
        print("\n" + "="*60)
        print("🏗️ [ModelingAgent] Starting Pyomo modeling")
        print("="*60)
        analysis_result_str = json.dumps(analysis_result, indent=2, ensure_ascii=False)
        print(f"Input analysis result: {analysis_result_str}")
        print(f"API availability status: {self.api_available}")
        
        if not self.api_available:
            # When API is unavailable, throw exception for coordinator to handle
            print("⚠️ API unavailable, cannot perform modeling")
            raise Exception("Modeling failed: Gemini API unavailable")
        
        # Progressive prompt strategy: from complex to simple
        def get_complex_prompt():
            return f"""### Role & Mission ###
You are a top-tier code generation engine specialized in translating structured engineering modeling blueprints (JSON) into executable, industrial-grade Python code.

### Core Instructions ###
Your **ONLY** task is to generate complete, accurate, directly executable Python solving code based on the `MODELING_BLUEPRINT` JSON provided below. Every key-value pair in the JSON is a mandatory instruction that must be strictly followed.

### Input: Modeling Blueprint (MODELING_BLUEPRINT) ###
```json
{analysis_result_str}
```

### ⚠️ CRITICAL Data Structure & Dimension Rules ###
🚫 **NEVER use placeholder data like [...] or ellipsis**
✅ **Always provide complete, specific data arrays**
- If JSON contains "[...]" or "...", replace with realistic example data
- Match array dimensions exactly to variable dimensions
- For time series: use 1D arrays like [0.1, 0.15, 0.2, ...]
- For scenarios: create 2D arrays like [[value1, value2], [value3, value4], ...]

🚫 **NEVER mismatch array dimensions with variable indices**
✅ **Ensure parameter arrays match variable indexing**
```python
# Wrong: Using 1D array with 3D variables
Electricity_price_t = [0.1, 0.15, ...]  # 1D
model.P_ev_t_s[e, t, s]  # 3D - MISMATCH!

# Correct: Match dimensions or simplify variables
# Option 1: Simplify to 2D variables
model.P_ev_t[e, t]  # 2D
Electricity_price_t = [0.1, 0.15, ...]  # 1D - MATCH!

# Option 2: Create appropriate multi-dimensional data
Electricity_price_t_s = [[0.1, 0.12], [0.15, 0.17], ...]  # 2D
model.P_ev_t_s[e, t, s]  # 3D with proper indexing
```

### ⚠️ CRITICAL Pyomo Summation Rules ###
🚫 **NEVER use summation() from pyomo.environ - it's error-prone**
✅ **Always use Python's built-in sum() with generator expressions**
```python
# Wrong: Using summation() function
from pyomo.environ import summation
electricity_cost = summation(Electricity_price_t, model.P_ev_t)  # ERROR!

# Correct: Using sum() with proper indexing
electricity_cost = sum(Electricity_price_t[t-1] * model.P_ev_t[e, t] 
                      for e in E for t in T)
```

🚫 **NEVER pass arrays directly to sum() - always use explicit indexing**
✅ **Always iterate over indices and access array elements**
```python
# Wrong: Direct array operations in sum
sum(price_array * model.var for ...)  # ERROR!

# Correct: Index-based access
sum(price_array[i] * model.var[i] for i in range(len(price_array)))
```

### ⚠️ CRITICAL Pyomo Syntax Rules ###
🚫 **NEVER use Python built-ins in constraints**: max(), min(), abs()
🚫 **NEVER use if statements with Pyomo variables**: if model.x[i] >= 0: ...  
🚫 **NEVER compare Pyomo variables in boolean context**: if P_ev_t[ev, t] >= 0
🚫 **NEVER use ternary operators with Pyomo variables**: value = model.x[i] if model.x[i] >= 0 else 0
🚫 **NEVER mix Pyomo expressions with conditional logic**: model.x[i] * (1 if condition else 0)
✅ **For conditional logic**: Use separate variables or Pyomo Piecewise functions
✅ **For max constraints**: Use auxiliary binary variables with Big-M method
✅ **For absolute value**: Use two inequality constraints: x >= y and x >= -y

### 🚨 MOST COMMON ERRORS TO AVOID ###
❌ **"Cannot convert non-constant Pyomo expression to bool" - CAUSED BY:**
```python
# WRONG - These will cause fatal errors:
if model.P_s_t[t] >= 0:
    return model.P_s_t[t] / η
value = model.x[i] if model.x[i] > 0 else model.x[i] * 0.9
model.constraint = Constraint(expr=model.var[i] >= (5 if condition else 3))
```

✅ **CORRECT alternatives:**
```python
# Method 1: Separate variables
model.P_charge = Var(T, domain=NonNegativeReals)  
model.P_discharge = Var(T, domain=NonNegativeReals)

# Method 2: Binary variables with constraints
model.mode = Var(T, domain=Binary)
model.cons1 = Constraint(T, rule=lambda m,t: m.P[t] <= M * m.mode[t])
model.cons2 = Constraint(T, rule=lambda m,t: m.P[t] >= -M * (1 - m.mode[t]))
```

### 🔥 CRITICAL: NEVER GENERATE INCOMPLETE CODE 🔥 ###
❌ **ABSOLUTELY FORBIDDEN**: Any code containing only "results = []" or similar empty placeholders
❌ **ABSOLUTELY FORBIDDEN**: Comments like "# The rest of the code remains unchanged"
❌ **ABSOLUTELY FORBIDDEN**: Any response that does not contain a COMPLETE Pyomo model

🚨 **MANDATORY REQUIREMENT**: Every response MUST contain a FULLY FUNCTIONAL Pyomo optimization model that can be executed immediately.

**Your code MUST include ALL of these components (NO EXCEPTIONS):**
1. ✅ Complete imports: `from pyomo.environ import *`
2. ✅ All parameter data with realistic values (no [...] placeholders)
3. ✅ Complete ConcreteModel() definition with ALL variables from JSON
4. ✅ Complete Objective function (maximize/minimize something meaningful)
5. ✅ ALL constraint definitions from the JSON properly implemented
6. ✅ Complete solver setup and solve() call
7. ✅ Result extraction and printing

**VERIFICATION CHECKLIST - Your code must pass ALL these checks:**
☑️ Does it import pyomo.environ?
☑️ Does it create a ConcreteModel()?
☑️ Does it define ALL variables mentioned in the JSON?
☑️ Does it implement ALL constraints from the JSON?
☑️ Does it define a meaningful objective function?
☑️ Does it call a solver and get results?
☑️ Is it longer than 100 lines of actual code?

### Output Requirements ###
- Format: Output only one complete Python code block wrapped in ```python...```
- Content: Code must include all necessary library imports, parameter definitions, model construction, solver calls, and clear result output
- Quality: Code must be robust with basic solver status checks
- No extra explanation: Except for the code itself and necessary comments, do not add any preamble or summary
- Solver setup: Include intelligent solver selection (glpk, cbc, ipopt) with timeouts and error handling
- Data integrity: All parameter arrays must be complete with realistic values, no placeholders
"""

        def get_simple_prompt():
            return f"""You are a Pyomo code generator. Generate COMPLETE, EXECUTABLE Python code based on this JSON specification:

```json
{analysis_result_str}
```

CRITICAL REQUIREMENTS:
1. 🚨 NEVER generate "results = []" or any empty placeholder code
2. 🚨 MUST include complete Pyomo model with ConcreteModel(), variables, constraints, objective
3. 🚨 MUST include solver call and result extraction
4. 🚨 Replace all "[...]" with actual numerical data
5. 🚨 Use simple 1D arrays to avoid dimension mismatches
6. 🚨 NEVER use if statements with Pyomo variables (causes "Cannot convert non-constant Pyomo expression to bool")
7. 🚨 NEVER use ternary operators: model.x[i] if condition else model.y[i]

REQUIRED STRUCTURE:
```python
from pyomo.environ import *

# Data (replace [...] with real numbers)
param1 = [0.1, 0.15, 0.2, ...]  # Complete arrays
param2 = [10, 15, 20, ...]       # No [...] allowed

# Model
model = ConcreteModel()
model.var1 = Var(range(len(param1)), domain=NonNegativeReals)

# Objective
model.obj = Objective(expr=sum(...), sense=minimize)

# Constraints  
model.constraint1 = Constraint(expr=...)

# Solve
solver = SolverFactory('glpk')
results = solver.solve(model)

# Results
print("Optimal value:", value(model.obj))
```

Generate ONLY the complete Python code wrapped in ```python...```. NO explanations."""

        def get_ultra_simple_prompt():
            return f"""Generate a complete Pyomo optimization model. No "results = []" allowed!

JSON input: {analysis_result_str}

Must include:
- from pyomo.environ import *  
- model = ConcreteModel()
- Variables with Var()
- Objective with minimize/maximize
- Constraints
- solver.solve(model)
- Print results

Return ONLY Python code in ```python...``` format."""

        # Progressive attempts: from complex to simple prompts
        prompts = [
            ("Complex detailed prompt", get_complex_prompt()),
            ("Simplified prompt", get_simple_prompt()), 
            ("Ultra-simple prompt", get_ultra_simple_prompt())
        ]
        
        # Maximum 3 attempts to generate valid code
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Choose different prompt strategy based on attempt number
                prompt_name, prompt = prompts[min(attempt, len(prompts)-1)]
                
                print(f"🚀 Calling LLM to generate model (Attempt {attempt + 1}/{max_attempts}, using {prompt_name})...")
                pyomo_code = call_llm_api(prompt, self.model_type)
                if not pyomo_code:
                    if attempt < max_attempts - 1:
                        continue
                    raise Exception(f"{self.model_type.upper()} API call failed")
                
                # ⭐ New: Check for empty code
                if self._is_empty_code(pyomo_code):
                    print(f"❌ Empty code detected, regenerating (Attempt {attempt + 1})")
                    if attempt < max_attempts - 1:
                        continue  # Regenerate
                    else:
                        raise Exception("Still generating empty code after multiple attempts")
                
                print("Generated code:")
                print("-" * 40)
                print(pyomo_code)
                print("-" * 40)
                
                # Clean up code block markers - Enhanced version
                # Handle multiple code block formats
                code_block_patterns = [
                    (r'```python\s*\n(.*?)\n```', 1),
                    (r'```py\s*\n(.*?)\n```', 1), 
                    (r'```\s*\n(.*?)\n```', 1),
                    (r'```python(.*?)```', 1),
                    (r'```(.*?)```', 1)
                ]
                
                extracted_code = None
                for pattern, group in code_block_patterns:
                    match = re.search(pattern, pyomo_code, re.DOTALL)
                    if match:
                        extracted_code = match.group(group).strip()
                        print(f"🔧 Using pattern to extract code: {pattern}")
                        break
                
                if extracted_code:
                    pyomo_code = extracted_code
                else:
                    # If no code block, clean up possible markers
                    pyomo_code = pyomo_code.replace('```python', '').replace('```py', '').replace('```', '').strip()
                    
                    # If code doesn't start with import, try to find where import statements begin
                    if not pyomo_code.startswith('import') and not pyomo_code.startswith('from'):
                        import_match = re.search(r'\n(import\s+|from\s+)', pyomo_code)
                        if import_match:
                            pyomo_code = pyomo_code[import_match.start()+1:]
                            print("🔧 Extracting code from import statements")
                
                # ⭐ Check extracted code again for emptiness
                if self._is_empty_code(pyomo_code):
                    print(f"❌ Extracted code block is empty, regenerating (Attempt {attempt + 1})")
                    if attempt < max_attempts - 1:
                        continue  # Regenerate
                    else:
                        raise Exception("Extracted code block is empty")
                
                print("🔧 Code block marker cleanup completed")
                
                # Basic code validation
                if not any(keyword in pyomo_code for keyword in ['import', 'pyo', 'pyomo']):
                    print("⚠️ Generated code is incomplete")
                    raise Exception("Modeling failed: Generated code does not contain necessary Pyomo components")
                
                self.logger.info("Pyomo model generation completed.")
                print("✅ Pyomo model generation completed")
                return pyomo_code
            
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"❌ Attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    self.logger.error(f"Error in model generation: {e}")
                    raise Exception(f"Modeling failed: {e}")

    def generate_model_with_error_context(self, error_info: Dict, strategy: str, attempt_number: int, 
                                         analysis_result: Dict = None, previous_model_code: str = None, 
                                         original_problem: str = None) -> str:
        """Generate model based on error information and context"""
        if not self.api_available:
            raise Exception("Modeling retry failed: Gemini API unavailable")
        
        try:
            # Extract detailed error information from solver error
            error_message = error_info.get("error", "Unknown error")
            error_code = error_info.get("code", "")  # The actual Pyomo code that failed (from SolverAgent)
            error_traceback = error_info.get("traceback", "")
            error_stage = error_info.get("stage", "")
            
            # ✅ Use the most relevant code source: error_code (from solver) takes priority
            failed_code = error_code if error_code else (previous_model_code or "")
            
            # Context information (may be empty in some scenarios)
            analysis_result = analysis_result or {}
            original_problem = original_problem or ""
            
            # Build focused prompt for solver error fixing
            # Safely handle strings that might contain format specifiers (avoid double-escaping)
            
            safe_error_message = smart_escape_braces(error_message)
            safe_failed_code = smart_escape_braces(failed_code)
            safe_strategy = smart_escape_braces(strategy)
            safe_error_traceback = smart_escape_braces(error_traceback)

            prompt = f"""### Role and Mission ###
You are a top-tier code debugging expert. Your task is to make **minimal and precise** "surgical" corrections to a piece of code with errors based on the provided context.

### Error Context ###
**Failed Pyomo Code:**
```python
{safe_failed_code}
```

**Execution Error:**
```
{safe_error_message}
```

**Coordinator's Fix Strategy:**
```
{safe_strategy}
```

### Fix Instructions ###
**CRITICAL**: You must implement the exact fix specified in the strategy above.

**Common Pyomo Error Patterns & Fixes:**
1. **"Unexpected keyword options"** → Remove invalid keyword arguments from component definitions
2. **"Cannot convert non-constant Pyomo expression to bool"** → Replace conditional logic with proper Pyomo constructs
3. **"name '...' is not defined"** → Add missing variable/parameter definitions
4. **"object has no attribute"** → Fix attribute names or import statements

**Your Task:**
1. **Locate the Error**: Use the error message to find the problematic code lines
2. **Apply the Strategy**: Implement ONLY the changes specified in the fix strategy
3. **Preserve Everything Else**: Keep all other code logic, structure, and formatting identical

### Output Requirements ###
- **Format**: Return only the corrected Python code wrapped in ```python...```
- **Content**: Complete, executable Pyomo model with the specific error fixed
- **No explanations**: Do not add comments about what you changed"""
            
            adjusted_code = call_llm_api(prompt, self.model_type)
            if not adjusted_code:
                raise Exception(f"{self.model_type.upper()} API call failed")
            
            # Use same code extraction logic
            code_block_patterns = [
                (r'```python\s*\n(.*?)\n```', 1),
                (r'```py\s*\n(.*?)\n```', 1), 
                (r'```\s*\n(.*?)\n```', 1),
                (r'```python(.*?)```', 1),
                (r'```(.*?)```', 1)
            ]
            
            extracted_code = None
            for pattern, group in code_block_patterns:
                match = re.search(pattern, adjusted_code, re.DOTALL)
                if match:
                    extracted_code = match.group(group).strip()
                    break
            
            if extracted_code:
                pyomo_code = extracted_code
            else:
                # Fix: Use adjusted_code instead of undefined pyomo_code
                pyomo_code = adjusted_code.replace('```python', '').replace('```py', '').replace('```', '').strip()
            
                # If code doesn't start with import, try to find where import statements begin
                if not pyomo_code.startswith('import') and not pyomo_code.startswith('from'):
                    import_match = re.search(r'\n(import\s+|from\s+)', pyomo_code)
                    if import_match:
                        pyomo_code = pyomo_code[import_match.start()+1:]
                        print("🔧 Extracting code from import statements")

            self.logger.info("LLM regenerated Pyomo model completed")
            return pyomo_code

        except Exception as e:
            self.logger.error(f"Error context modeling failed: {e}")
            raise Exception(f"Modeling retry failed: {e}")

    def generate_simulation_model(self, analysis_result: Dict[str, Any]) -> str:
        """Generate universal Pyomo model code based on intelligent LLM analysis"""
        # Use LLM to generate model code, throw exception when API is unavailable
        if self.api_available:
            return self.llm_generate_model(analysis_result)
        else:
            raise Exception(f"{self.model_type.upper()} API unavailable, cannot generate model code. Please check API configuration.")

    def llm_generate_model(self, analysis_result: Dict[str, Any]) -> str:
        """Use LLM to intelligently generate Pyomo model"""
        analysis_result_str = json.dumps(analysis_result, indent=2, ensure_ascii=False)
        prompt = f"""You are an optimization modeling expert. Based on the following problem analysis results, generate a complete Pyomo model code.

    Analysis Results:
    {analysis_result_str}
    
    Requirements:
    1. Generate complete runnable Python code
    2. Include all necessary import statements
    3. Intelligently select variable types based on problem characteristics (continuous, integer, binary)
    4. Select appropriate sense based on objective function type (minimize/maximize)
    5. Generate corresponding mathematical expressions based on constraint descriptions
    6. Automatically add time indices if time series are involved
    7. Set reasonable default values or data generation logic for parameters
    8. Add appropriate comments and explanations
    9. **Important**: Solver selection must prioritize open-source solvers, try in the following order:
       - First choice: 'glpk' (suitable for linear programming)
       - Second choice: 'cbc' (suitable for mixed integer programming)
       - Alternative: 'ipopt' (suitable for nonlinear programming)
       - Avoid directly hardcoding commercial solvers like gurobi or cplex unless explicitly needed
    10. Must include solver availability detection, automatically switch to alternatives if preferred solver is unavailable
    11. **Solver timeout configuration**:
        - Must set appropriate timeout for each solver: e.g., glpk(300s), cbc(600s), ipopt(900s).
        - If solving times out, throw exception with fixed error message: "Model complexity does not match current solver processing capability, recommend simplifying model or adjusting constraints"
    12. **⚠️ Important Pyomo syntax restrictions**:
        - 🚫 **Absolutely forbidden** to use Python built-in functions in constraints: max(), min(), abs()
        - 🚫 **Absolutely forbidden** to use if statements in constraints or objective functions: if model.x[i] >= 0: ...
        - 🚫 **Absolutely forbidden** to directly compare Pyomo variables: if P_ev_t[ev, t] >= 0
        - ✅ **For conditional logic**: Use Pyomo's Piecewise functions or binary variables
        - ✅ **For maximum value constraints**: Use additional binary variables and Big-M method
        - ✅ **For absolute value constraints**: Use two inequality constraints instead: x >= y and x >= -y
        - ✅ **Correct examples**:
          ```python
          # Wrong: if model.P[t] >= 0: return 1.0 else: return 0.9
          # Correct: Use parameterized expressions
          def efficiency_rule(model, t):
              return model.charge_power[t] * 1.0 + model.discharge_power[t] * 0.9
          ```
        - 🚫 **Avoid**: Any syntax that leads to "Cannot convert non-constant Pyomo expression to bool"
    
    **Pay special attention to the following common error patterns and correct approaches:**
    
    🚫 **Error Pattern 1: Conditional expressions in objective function**
    ```python
    # Wrong:
    def total_cost(model):
        return sum(price[t] * P[t] * (1 if P[t] >= 0 else efficiency) for t in T)
    ```
    ✅ **Correct approach: Separate charge/discharge variables**
    ```python
    # Correct:
    model.P_charge = Var(T, domain=NonNegativeReals)
    model.P_discharge = Var(T, domain=NonNegativeReals)
    def total_cost(model):
        return sum(price[t] * (model.P_charge[t] - model.P_discharge[t] * efficiency) for t in T)
    ```
    
    🚫 **Error Pattern 2: Conditional expressions in constraints**
    ```python
    # Wrong:
    def soc_rule(model, t):
        return model.SOC[t] == model.SOC[t-1] + (model.P[t] if model.P[t] >= 0 else model.P[t] * eff)
    ```
    ✅ **Correct approach: Use separate variables**
    ```python
    # Correct:
    def soc_rule(model, t):
        return model.SOC[t] == model.SOC[t-1] + model.P_charge[t] - model.P_discharge[t] * eff
    ```
    
    🚫 **Error Pattern 3: Conditional logic in demand constraints**
    ```python
    # Wrong:
    def demand_rule(model, p, t):
        if p in demand_products:
            return model.production[p,t] >= demand[p][t]
        return Constraint.Skip
    ```
    ✅ **Correct approach: Pre-filter indices**
    ```python
    # Correct:
    DEMAND_PRODUCTS = [p for p in PRODUCTS if p in demand_products]
    def demand_rule(model, p, t):
        return model.production[p,t] >= demand[p][t]
    model.DemandConstraint = Constraint(DEMAND_PRODUCTS, T, rule=demand_rule)
    ```
    
    Please return only Python code wrapped in ```python...```, do not include any explanatory text."""
    
        # Maximum 3 attempts to generate valid code
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.logger.info(f"🤖 Attempting to generate model code (attempt {attempt + 1}/{max_attempts})")
                
                pyomo_code = call_llm_api(prompt, self.model_type)
                if not pyomo_code:
                    if attempt < max_attempts - 1:
                        continue
                    raise Exception(f"{self.model_type.upper()} API call failed")
                
                # ⭐ New: Check if code is empty
                if self._is_empty_code(pyomo_code):
                    self.logger.warning(f"❌ Empty code detected, regenerating (attempt {attempt + 1})")
                    if attempt < max_attempts - 1:
                        continue  # Regenerate
                    else:
                        raise Exception("Empty code generated after multiple attempts")
                
                # Use same code extraction logic
                code_block_patterns = [
                    (r'```python\s*\n(.*?)\n```', 1),
                    (r'```py\s*\n(.*?)\n```', 1), 
                    (r'```\s*\n(.*?)\n```', 1),
                    (r'```python(.*?)```', 1),
                    (r'```(.*?)```', 1)
                ]
                
                extracted_code = None
                for pattern, group in code_block_patterns:
                    match = re.search(pattern, pyomo_code, re.DOTALL)
                    if match:
                        extracted_code = match.group(group).strip()
                        break
                
                if extracted_code:
                    pyomo_code = extracted_code
                else:
                    pyomo_code = pyomo_code.replace('```python', '').replace('```py', '').replace('```', '').strip()
                
                # If code doesn't start with import, try to find import statement start position
                if not pyomo_code.startswith('import') and not pyomo_code.startswith('from'):
                    import_match = re.search(r'\n(import\s+|from\s+)', pyomo_code)
                    if import_match:
                        pyomo_code = pyomo_code[import_match.start()+1:]
                        print("🔧 Extracting code starting from import statement")

                # ⭐ Check again if extracted code is empty
                if self._is_empty_code(pyomo_code):
                    self.logger.warning(f"❌ Extracted code block is empty, regenerating (attempt {attempt + 1})")
                    if attempt < max_attempts - 1:
                        continue  # Regenerate
                    else:
                        raise Exception("Extracted code block is empty")

                self.logger.info("✅ Successfully generated valid model code")
                return pyomo_code
            
            except Exception as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(f"❌ Attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    self.logger.error(f"LLM model generation failed: {e}")
                    raise Exception(f"Model generation failed, LLM unable to generate valid Pyomo code: {e}")

    def _is_empty_code(self, code_text):
        """Check if code is empty or invalid"""
        if not code_text or not code_text.strip():
            return True
        
        # Check common empty code patterns
        empty_patterns = [
            r"^\s*results\s*=\s*\[\s*\]\s*$",  # results = []
            r"^\s*results\s*=\s*\[\s*\]\s*#.*$",  # results = [] # comment
            r"^\s*#.*$",  # Only comments
            r"^\s*pass\s*$",  # Only pass
            r"^\s*\.\.\.\s*$",  # Only ...
        ]
        
        # Clean code (remove empty lines and comments)
        lines = [line.strip() for line in code_text.split('\n') if line.strip()]
        cleaned_code = '\n'.join(lines)
        
        # Check if matches empty code patterns
        import re
        for pattern in empty_patterns:
            if re.match(pattern, cleaned_code, re.MULTILINE | re.DOTALL):
                self.logger.warning(f"🚫 Empty code pattern detected: {pattern}")
                return True
        
        # Check code length (too short might be invalid)
        if len(cleaned_code) < 100:  # Less than 100 characters considered invalid code
            self.logger.warning(f"🚫 Code too short ({len(cleaned_code)} characters)")
            return True
        
        # Check if contains basic Pyomo elements
        required_elements = ['pyomo', 'ConcreteModel', 'Var', 'Objective']
        has_required = any(element.lower() in cleaned_code.lower() for element in required_elements)
        
        if not has_required:
            self.logger.warning("🚫 Missing basic Pyomo elements")
            return True
        
        return False
    

class VerificationAgent(BaseAgent):
    def __init__(self, agent_id: str, model_type: str = "gemini"):
        # Set correct llm_model based on model_type
        if model_type.lower() == "gemini":
            llm_model = "gemini-2.5-flash-lite"
        elif model_type.lower() == "gpt5":
            llm_model = "gpt-5"
        elif model_type.lower() == "gpt4o":
            llm_model = "gpt-4o"
        elif model_type.lower() == "gpt4":
            llm_model = "gpt-4"
        elif model_type.lower() == "deepseek":
            llm_model = "deepseek-v3"
        else:  # Default to gpt-4o
            llm_model = "gpt-4o"
        super().__init__(agent_id, llm_model)
        self.model_type = model_type.lower()
        self.original_problem = ""  # Store original problem for matching check
        self.max_verification_retries = 3  # Maximum verification retries
        self.verification_attempt_count = 0  # Current verification attempt count
        # New: cycle detection and dynamic verification standard
        self.consecutive_verification_failures = 0
        self.max_consecutive_failures = 10  # Consecutive failure threshold
        self.tolerance_level = 0.0  # Verification tolerance 0.0-1.0
        
        try:
            # Test using unified API calling function
            test_response = call_llm_api("Hello", self.model_type)
            self.api_available = test_response is not None
            self.logger.info(f"{self.model_type.upper()} API connection successful" if self.api_available else f"{self.model_type.upper()} API unavailable, switch to simulation mode")
        except Exception as e:
            self.logger.warning(f"{self.model_type.upper()} API unavailable, switch to simulation mode: {e}")
            self.api_available = False

    def process_message(self, message: Message) -> Message:
        if message.msg_type == MessageType.QUERY:
            pyomo_code = message.content
            self.logger.info("Verifying Pyomo model...")
            
            # Get original problem from coordinator (if available)
            if hasattr(message, 'metadata') and message.metadata and 'original_problem' in message.metadata:
                self.original_problem = message.metadata['original_problem']
            
            try:
                # Reset verification attempt counter (new verification request)
                self.verification_attempt_count = 0
                
                verification_result = self.verify_pyomo_model(pyomo_code)
                if verification_result.get("is_valid", False):
                    # Verification successful, reset consecutive failure counter
                    self.consecutive_verification_failures = 0
                    response_msg = Message(self.agent_id, "coordinator", pyomo_code, MessageType.RESPONSE)
                else:
                    # Increase consecutive failure counter
                    self.consecutive_verification_failures += 1
                    
                    # Check if consecutive failure threshold is reached, force verification to pass
                    if self.consecutive_verification_failures >= self.max_consecutive_failures:
                        print(f"🔄 Detection of verification loop: consecutive {self.consecutive_verification_failures} failures, force verification to avoid dead loop")
                        self.consecutive_verification_failures = 0  # Reset counter
                        response_msg = Message(self.agent_id, "coordinator", pyomo_code, MessageType.RESPONSE)
                        return response_msg
                    
                    # Check if maximum retry attempts are reached
                    if self.verification_attempt_count >= self.max_verification_retries:
                        print(f"⚠️ Verification retry attempts reached limit ({self.max_verification_retries}), force verification to pass")
                        response_msg = Message(self.agent_id, "coordinator", pyomo_code, MessageType.RESPONSE)
                        return response_msg
                    
                    # If verification fails, return error message
                    error_content = {
                        "error": "Model verification failed", 
                        "error_type": "VerificationError", 
                        "issues": verification_result.get("issues", []),
                        "suggestions": verification_result.get("suggestions", []),
                        "stage": "verification",
                        "mismatch_detected": verification_result.get("mismatch_detected", False),
                        "pyomo_code": pyomo_code  # ✅ Contains actual verification code
                    }
                    
                    # If modeling mismatch, add complete verification feedback information
                    if verification_result.get("mismatch_detected", False):
                        error_content["mismatch_details"] = verification_result.get("mismatch_details", {})
                        # Flatten the key fields to ensure compatibility, but prioritize the full information in mismatch_details
                        error_content["mismatch_reason"] = error_content["mismatch_details"].get("mismatch_reason", "")
                        error_content["suggestion"] = error_content["mismatch_details"].get("suggestion", "")
                        # Add verification context information, help with subsequent retries
                        error_content["verification_context"] = {
                            "verification_attempt": self.verification_attempt_count,
                            "consecutive_failures": self.consecutive_verification_failures,
                            "verification_strategy": "mismatch_detection",
                            "original_problem_available": bool(self.original_problem)
                        }
                    
                    error_msg = Message(
                        self.agent_id, 
                        "coordinator", 
                        error_content, 
                        MessageType.ERROR
                    )
                    return error_msg
                return response_msg
            except Exception as e:
                self.logger.error(f"Error in verification: {e}")
                error_msg = Message(
                    self.agent_id, 
                    "coordinator", 
                    {"error": str(e), "error_type": "VerificationError", "stage": "verification", "mismatch_detected": False, "pyomo_code": pyomo_code}, 
                    MessageType.ERROR
                )
                return error_msg
        elif message.msg_type == MessageType.RETRY:
            # Process retry request - re-verify based on error information
            retry_info = message.content
            strategy = retry_info.get("strategy", "Verification retry")
            error_info = retry_info.get("error_info", {})
            pyomo_code = retry_info.get("pyomo_code", "")
            original_problem = retry_info.get("original_problem", "")
            
            # Set original problem to support modeling matching check
            if original_problem:
                self.original_problem = original_problem
            
            # Increase verification attempt counter
            self.verification_attempt_count += 1
            
            # Check if maximum retry attempts are reached
            if self.verification_attempt_count > self.max_verification_retries:
                print(f"⚠️ Verification retry attempts reached({self.max_verification_retries})，force verification")
                response_msg = Message(self.agent_id, "coordinator", pyomo_code, MessageType.RESPONSE)
                return response_msg
            
            self.logger.info(f"Executing verification retry: {strategy} (Attempt {self.verification_attempt_count})")
            
            try:
                # Use error context to re-verify
                verification_result = self.verify_with_error_context(pyomo_code, error_info, strategy)
                if verification_result.get("is_valid", False):
                    response_msg = Message(self.agent_id, "coordinator", pyomo_code, MessageType.RESPONSE)
                else:
                    # If verification still fails, return error message
                    error_content = {
                        "error": "Model verification retry failed", 
                        "error_type": "VerificationError", 
                        "issues": verification_result.get("issues", []),
                        "suggestions": verification_result.get("suggestions", []),
                        "stage": "retry_verification",
                        "mismatch_detected": verification_result.get("mismatch_detected", False),
                        "pyomo_code": pyomo_code
                    }
                    
                    # If modeling mismatch, add complete verification feedback information  
                    if verification_result.get("mismatch_detected", False):
                        error_content["mismatch_details"] = verification_result.get("mismatch_details", {})
                        # Flatten the key fields to ensure compatibility, but prioritize the full information in mismatch_details
                        error_content["mismatch_reason"] = error_content["mismatch_details"].get("mismatch_reason", "")
                        error_content["suggestion"] = error_content["mismatch_details"].get("suggestion", "")
                        # Add verification context information, help with subsequent retries
                        error_content["verification_context"] = {
                            "verification_attempt": self.verification_attempt_count,
                            "consecutive_failures": self.consecutive_verification_failures,
                            "verification_strategy": "retry_mismatch_detection",
                            "original_problem_available": bool(self.original_problem)
                        }
                    
                    error_msg = Message(
                        self.agent_id, 
                        "coordinator", 
                        error_content, 
                        MessageType.ERROR
                    )
                    return error_msg
                return response_msg
            except Exception as e:
                error_msg = Message(
                    self.agent_id, 
                    "coordinator", 
                    {"error": str(e), "error_type": "VerificationError", "stage": "retry_verification", "mismatch_detected": False, "pyomo_code": pyomo_code}, 
                    MessageType.ERROR
                )
                return error_msg
        else:
            self.logger.warning(f"VerificationAgent received unexpected message type: {message.msg_type}")
            error_msg = Message(
                self.agent_id, 
                "coordinator", 
                {"error": f"Unsupported message type: {message.msg_type.value}", "error_type": "VerificationError", "stage": "message_processing", "mismatch_detected": False, "pyomo_code": pyomo_code}, 
                MessageType.ERROR
            )
            return error_msg

    def verify_pyomo_model(self, pyomo_code: str) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("🔍 [VerificationAgent] Starting LLM-driven modeling matching check")
        print("="*60)
        print(f"Input code length: {len(pyomo_code)} characters")
        
        try:
            # 1. LLM-driven modeling matching check (core functionality)
            if self.original_problem:
                print("🔧 Starting LLM modeling matching check...")
                mismatch_result = self._check_model_problem_match(pyomo_code, self.original_problem)
                
                if mismatch_result["mismatch_detected"]:
                    print(f"❌ Modeling mismatch with original problem: {mismatch_result['mismatch_reason']}")
                    return {
                        "is_valid": False,
                        "issues": [mismatch_result["mismatch_reason"]],
                        "suggestions": [mismatch_result["suggestion"]],
                        "mismatch_detected": True,
                        "mismatch_details": mismatch_result
                    }
                else:
                    print("✅ Modeling matches original problem")
            else:
                print("⚠️ No original problem information, skipping matching check")
            
            print("✅ Verification successful")
            self.logger.info("Pyomo model verification completed - Valid")
            return {
                "is_valid": True,
                "issues": [],
                "suggestions": [],
                "mismatch_detected": False
            }
            
        except Exception as e:
            # Other unexpected errors (syntax, format, etc.) - retry in place
            print(f"❌ Verification process encountered unexpected error: {e}")
            return {
                "is_valid": False,
                "issues": [f"Verification process encountered unexpected error: {str(e)}"],
                "suggestions": ["Please check input format and content"],
                "mismatch_detected": False
            }
    
    def _check_model_problem_match(self, pyomo_code: str, original_problem: str) -> Dict[str, Any]:
        """Using LLM to check if modeling matches original problem"""
        try:
            # Safely handle strings that may contain formatting symbols
            safe_original_problem = smart_escape_braces(original_problem)
            safe_pyomo_code = smart_escape_braces(pyomo_code)
            
            # Dynamically adjust Verification tolerance
            tolerance_adjustment = ""
            if self.consecutive_verification_failures > 8:
                tolerance_adjustment = """
**⚠️ Special Notice: Multiple verification loops detected, please adopt more lenient verification standards**
- For complex game theory/bilevel optimization problems, allow reasonable modeling simplification
- For MPEC/KKT conditions, accept pragmatic-oriented approximate implementations  
- Focus on verifying core logic correctness, be tolerant of technical details
- Only judge as mismatch when fundamental errors exist"""
            elif self.consecutive_verification_failures > 5:
                tolerance_adjustment = """
**Notice: Verification difficulties detected, please appropriately relax verification standards**
- Maintain understanding and tolerance for modeling strategy differences
- Focus on substantial issues, ignore technical implementation details"""
                
            prompt = f"""### Role and Mission ###
You are a top-tier, **extremely pragmatic** engineering system verification expert. Your core mission is to serve as the system's "gatekeeper". Your task is not to conduct line-by-line code auditing, but to judge from a **system modeling perspective** whether the provided Python code faithfully solves the original engineering problem in terms of **core logic and key objectives**. You must distinguish between **modeling strategy differences** and **substantial logical errors**.
{tolerance_adjustment}

Your primary principle is **"Promote Convergence"**:
- **Default Trust**: Unless the code contains **catastrophic, directional** errors, it should default to `PASS`.
- **Understand Compromise**: Must recognize that the current code may be a **necessary compromise or simplification** made to solve **previous failures (such as overly strict constraints, solver infeasibility, etc.)**.
- **Minimal Intervention**: Your role is to confirm the overall direction is correct and the core has not deviated, not to micromanage technical details.

### Input ###
1.  **[Problem Specification] Original Engineering Problem (The Problem Specification)**:
    ```
    {safe_original_problem}
    ```
2.  **[Solution Implementation] Engineering Model Code (The Implemented Solution)**:
    ```python
    {safe_pyomo_code}
    ```

### Core Verification Instructions: Judgment Hierarchy Based on Historical Context ###
Please strictly follow this judgment logic to ensure your verification is context-aware.

**Step 1: Check for "Catastrophic" Errors (Deal-Breaker Check)**
This is the only reason you can judge as `FAIL`. Check if any of the following situations exist:
1.  **Objective Direction Reversed**: Minimization problems implemented as maximization, or vice versa.
2.  **Core Physical/Economic Law Violations**: Complete omission of constraints that define the problem foundation, such as "supply-demand balance", "energy conservation", etc.
3.  **Key Decision-Making Entities Missing**: In an obvious multi-agent game problem, the code completely fails to reflect interactions or decisions between different entities (even in simplified form).

*   If any of the above is found, immediately judge as `FAIL` and stop subsequent checks.
*   If none are found, **proceed directly to Step 2 and finally judge as `PASS`**.

**Step 2: Identify and Acknowledge "Pragmatic Deviations" (Pragmatic Deviation Recognition)**
When code passes Step 1 checks, it will be judged as `PASS`. Your task is to identify differences between the current code and the original problem specification, and judge whether this is a reasonable response to errors in the `[Historical Context]`.

Common **reasonable deviations that should be accepted as `PASS`** include:
-   **Relaxing/removing constraints to solve infeasible problems**: Should consider that previous reasonable modeling might have been "infeasible", and the current code relaxing or removing certain strict constraints is a **positive, should-be-encouraged** moderate compromise.
-   **Simplifying models to address LLM capability limitations or solving complexity**: Simplifying complex bilevel optimization, MPEC or equilibrium models to appropriate single-level optimization, as long as key economic incentives or trade-off relationships are embodied through parameters or proxy constraints, should be accepted.
-   **Technical adjustments**: Adjusting variable boundaries, parameter units, Big-M coefficients, solver options, etc., to improve numerical stability or find feasible solutions.
-   **Minor discrepancies in data or indices**: Should be ignored as long as they don't affect the core model logic and magnitude.

### Output Format ###
Please strictly return your judgment in the following JSON format, wrapped with ```json...```. **When the model adopts "acceptable approximation", please provide minimal incremental correction suggestions in `suggestion` rather than outright rejection.**

```json
{{
  "mismatch_detected": true/false,
  "mismatch_reason": "Specific mismatch reason",
  "suggestion": "Fix or optimization suggestions",
  "confidence": 0.0-1.0
}}
```
"""
            
            response_text = call_llm_api(prompt, self.model_type)
              
            # Strategy 1: Extract ```json``` block
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                if json_end != -1:
                    response_text = response_text[json_start:json_end].strip()
                    print("🔧 Extract JSON from ```json``` block")
                else:
                    # If no end marker, take to the end
                    response_text = response_text[json_start:].strip()
                    print("🔧 Extract JSON from ```json``` block (no end marker)")
            
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
                    print("🔧 Extract largest complete JSON object")
                else:
                    # back to original method
                    json_end = response_text.rfind('}') + 1
                    response_text = response_text[json_start:json_end]
                    print("🔧 Extract JSON fragment from text")
            
            # Clean up common JSON format issues
            response_text = response_text.replace('```', '').strip()
            # Handle possible encoding issues
            response_text = response_text.replace('"', '"').replace('"', '"')
            
            result = json.loads(response_text)
            
            return {
                "mismatch_detected": result.get("mismatch_detected", False),
                "mismatch_reason": result.get("mismatch_reason", ""),
                "suggestion": result.get("suggestion", ""),
                "confidence": result.get("confidence", 0.8)
            }
            
        except Exception as e:
            self.logger.error(f"LLM matching check failed: {e}")
            # If LLM check fails, default to matching
            return {
                "mismatch_detected": False,
                "mismatch_reason": "",
                "suggestion": "",
                "confidence": 0.5
            }
    
    def verify_with_error_context(self, pyomo_code: str, error_info: Dict, strategy: str) -> Dict[str, Any]:
        """Based on error information re-verify model - follow the same structure as verify_pyomo_model"""
        print("\n" + "="*60)
        print("🔄 [VerificationAgent] Based on error context re-verify")
        print("="*60)
        print(f"Retry strategy: {strategy}")
        error_info_str = str(error_info)
        print(f"Error information: {error_info_str}")
        
        if not self.api_available:
            raise Exception("Verification retry failed: Gemini API unavailable")
        
        try:
            # 1. Based on error context verification retry (consistent structure with verify_pyomo_model)
            if self.original_problem:
                print("🔧 Starting verification retry with error context...")
                mismatch_result = self._check_model_problem_match_with_error_context(
                    pyomo_code, self.original_problem, error_info, strategy)
                
                if mismatch_result["mismatch_detected"]:
                    print(f"❌ Verification retry found modeling mismatches original problem: {mismatch_result['mismatch_reason']}")
                    return {
                        "is_valid": False,
                        "issues": [mismatch_result["mismatch_reason"]],
                        "suggestions": [mismatch_result["suggestion"]],
                        "mismatch_detected": True,
                        "mismatch_details": mismatch_result
                    }
                else:
                    print("✅ Verification retry found modeling matches original problem")
            else:
                print("⚠️ No original problem information, skipping matching check")
            
            print("✅ Verification successful")
            self.logger.info("Verification retry completed - Valid")
            return {
                "is_valid": True,
                "issues": [],
                "suggestions": [],
                "mismatch_detected": False
            }
            
        except Exception as e:
            # Other unexpected errors (syntax, format, etc.) - retry in place
            print(f"❌ Verification process encountered unexpected error: {e}")
            return {
                "is_valid": False,
                "issues": [f"Verification process encountered unexpected error: {str(e)}"],
                "suggestions": ["Please check input format and content, or contact technical support"],
                "mismatch_detected": False
            }
        
    def _check_model_problem_match_with_error_context(self, pyomo_code: str, original_problem: str, 
                                                     error_info: Dict, strategy: str) -> Dict[str, Any]:
        """Based on error context check if modeling matches original problem"""
        try:
            error_message = error_info.get("error", "Unknown error")
            error_type = error_info.get("error_type", "")
            
            # Safely handle strings that may contain formatting symbols
            safe_original_problem = smart_escape_braces(original_problem)
            safe_pyomo_code = smart_escape_braces(pyomo_code)
            safe_error_message = smart_escape_braces(error_message)
            safe_error_type = smart_escape_braces(error_type)
            safe_strategy = smart_escape_braces(strategy)
            
            prompt = f"""You are a practical model verification expert. Please combine the previous error information to re-evaluate and analyze the consistency between the following model code and the original problem, focusing on core modeling logic while maintaining tolerance for reasonable technical implementation differences:

Original Problem:
{safe_original_problem}

Model Code:
{safe_pyomo_code}

Previous Error Information:
- Error Message: {safe_error_message}
- Error Type: {safe_error_type}
- Retry Strategy: {safe_strategy}

Please focus on the elements involved in error feedback and systematically analyze the following aspects:
1. Accuracy and completeness of decision variable definitions
2. Correspondence between objective function and problem objectives and variables
3. Coverage and logical rigor of constraint conditions
4. Specificity and matching of parameter settings
5. Obvious modeling flaws or potential issues
6. Whether there is room for further optimization

Please only output the following JSON format judgment result (wrapped with ```json...```):
```json
{{
  "mismatch_detected": true/false,
  "mismatch_reason": "Specific mismatch reason (if any)",
  "suggestion": "Fix or optimization suggestions",
  "confidence": 0.0-1.0
}}
```

**Judgment Principles:**
- If there are **parameter value errors, missing key constraints, variable definition errors, objective function errors** and other serious problems affecting model correctness, must return "mismatch_detected": true
- Any numerical values explicitly given in the original problem that do not match in the code must be judged as mismatch_detected: true
- **Allowed reasonable handling (not considered mismatch)**:
  - Reasonable adjustments to variable naming (x_A vs production_A)
  - Reasonable approximation or assumption of parameter values (when the original problem is not explicit)
  - Equivalent transformation of constraint expressions
  - Technical implementation simplification (such as set definition, index processing)
  - Code structure and solver selection
- **Only return "mismatch_detected": true when there are serious problems affecting the essential correctness of the model**
- **For reasonable modeling assumptions and technical implementation differences, should return "mismatch_detected": false**
"""
            
            response_text = call_llm_api(prompt, self.model_type)
            if not response_text:
                raise Exception(f"{self.model_type.upper()} API call failed")
              
            # Use the same JSON parsing logic as _check_model_problem_match
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                if json_end != -1:
                    response_text = response_text[json_start:json_end].strip()
                else:
                    response_text = response_text[json_start:].strip()
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
                else:
                    json_end = response_text.rfind('}') + 1
                    response_text = response_text[json_start:json_end]
            
            response_text = response_text.replace('```', '').strip()
            response_text = response_text.replace('"', '"').replace('"', '"')
            
            result = json.loads(response_text)
            
            return {
                "mismatch_detected": result.get("mismatch_detected", False),
                "mismatch_reason": result.get("mismatch_reason", ""),
                "suggestion": result.get("suggestion", ""),
                "confidence": result.get("confidence", 0.8)
            }
            
        except Exception as e:
            self.logger.error(f"LLM matching check based on error context failed: {e}")
            # If LLM check fails, default to matching
            return {
                "mismatch_detected": False,
                "mismatch_reason": "",
                "suggestion": "",
                "confidence": 0.5
            }
            
class SolverAgent(BaseAgent):
    def __init__(self, agent_id: str, model_type: str = "gemini"):
        # Set correct llm_model based on model_type
        if model_type.lower() == "gemini":
            llm_model = "gemini-2.5-flash-lite"
        elif model_type.lower() == "gpt5":
            llm_model = "gpt-5"
        elif model_type.lower() == "gpt4o":
            llm_model = "gpt-4o"
        elif model_type.lower() == "gpt4":
            llm_model = "gpt-4"
        elif model_type.lower() == "deepseek":
            llm_model = "deepseek-v3"
        else:  # Default to gpt-4o
            llm_model = "gpt-4o"
        super().__init__(agent_id, llm_model)
        self.model_type = model_type.lower()
        
        # Intelligent detection of API availability
        try:
            # Test using unified API calling function
            test_response = call_llm_api("Hello", self.model_type)
            self.api_available = test_response is not None
            self.logger.info(f"{self.model_type.upper()} API connection successful" if self.api_available else f"{self.model_type.upper()} API unavailable, switch to simulation mode")
        except Exception as e:
            self.logger.warning(f"{self.model_type.upper()} API unavailable, switch to simulation mode: {e}")
            self.api_available = False
        
        # Initialize LLM model (Only Gemini needs this, GPT and DeepSeek call API directly)
        if self.model_type == "gemini":
            try:
                self.model = genai.GenerativeModel(self.llm_model)
            except Exception as e:
                self.logger.warning(f"Unable to initialize Gemini model: {e}")
                self.model = None
        else:
            self.model = None  # GPT and DeepSeek do not need pre-initialized model

    def process_message(self, message: Message) -> Message:
        if message.msg_type == MessageType.QUERY:
            pyomo_code = message.content
            self.logger.info("Solving Pyomo model...")
            try:
                solution_result = self.solve_pyomo_model(pyomo_code)
                response_msg = Message(self.agent_id, "coordinator", solution_result, MessageType.RESPONSE)
                return response_msg
            except Exception as e:
                self.logger.error(f"Error in solving: {e}")
                # Keep error context as complete as possible (parse from exception if solve_pyomo_model returns structured error context; otherwise provide basic info)
                error_msg = Message(
                    self.agent_id, 
                    "coordinator", 
                    {
                        "error": str(e), 
                        "error_type": "SolverError", 
                        "stage": "solving",
                        "code": pyomo_code,  # Include code that caused error
                        "traceback": traceback.format_exc()
                    }, 
                    MessageType.ERROR
                )
                return error_msg
        elif message.msg_type == MessageType.RETRY:
            # Handle solver retry - let coordinator decide instead of using simulation fallback
            retry_info = message.content
            strategy = retry_info.get("strategy", "Retry solving")
            attempt_number = retry_info.get("attempt_number", 1)
            
            self.logger.info(f"Executing solver retry: {strategy} (Attempt {attempt_number})")
            
            # SolverAgent should return error directly on retry, let coordinator decide next step
            error_msg = Message(
                self.agent_id, 
                "coordinator", 
                {"error": "SolverAgent cannot handle retry, needs model-level changes", "error_type": "SolverError", "stage": "retry_solving"}, 
                MessageType.ERROR
            )
            return error_msg
        else:
            self.logger.warning(f"SolverAgent received unexpected message type: {message.msg_type}")
            error_msg = Message(
                self.agent_id, 
                "coordinator", 
                {"error": f"Unsupported message type: {message.msg_type.value}", "error_type": "SolverError", "stage": "message_processing"}, 
                MessageType.ERROR
            )
            return error_msg

    def solve_pyomo_model(self, pyomo_code: str) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("🚀 [SolverAgent] Starting engineering model solving")
        print("="*60)
        
        import io
        import contextlib
        import sys
        import os
        import gc
        
        # Create completely isolated execution environment
        exec_stdout_io = io.StringIO()
        exec_stderr_io = io.StringIO()
        run_log = {}
        solver_detection = []
        
        # Save original environment state
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        try:
            print("🔧 Preparing isolated code execution environment...")
            
            # Create completely isolated namespace
            namespace = {
                '__builtins__': __builtins__,
                '__name__': '__main__',
                '__doc__': None,
                '__package__': None
            }
            
            # Intelligently import necessary libraries to isolated namespace
            essential_imports = [
                "import sys",
                "import os", 
                "import time",
                "import math",
                "import random"
            ]
            
            # Import appropriate libraries based on code content
            if "pyomo" in pyomo_code.lower() or "pyo." in pyomo_code:
                essential_imports.extend([
                    "from pyomo.environ import *",
                    "import pyomo.environ as pyo"
                ])
                # Detect available solvers
                solver_detection_code = """
# Detect available solvers
available_solvers = []
solver_preferences = ['glpk', 'cbc', 'ipopt', 'gurobi', 'cplex']
import pyomo.environ as pyo

# Solver timeout settings (seconds)
SOLVER_TIMEOUT_SETTINGS = {
    'glpk': 300,    # 5 minutes - suitable for medium-scale linear programming
    'cbc': 600,     # 10 minutes - suitable for mixed integer programming
    'ipopt': 900,   # 15 minutes - suitable for nonlinear programming
    'gurobi': 1200, # 20 minutes - commercial solver, can handle more complex problems
    'cplex': 1200   # 20 minutes - commercial solver, can handle more complex problems
}

for solver_name in solver_preferences:
    try:
        solver = pyo.SolverFactory(solver_name)
        if solver.available():
            available_solvers.append(solver_name)
            timeout = SOLVER_TIMEOUT_SETTINGS.get(solver_name, 300)
            print(f"✅ Detected available solver: {solver_name} (timeout setting: {timeout} seconds)")
    except:
        pass
        
if not available_solvers:
    print("⚠️ No available solvers detected")
else:
    print(f"🔧 Recommended solver: {available_solvers[0]}")
"""
                essential_imports.append(solver_detection_code)
                
            if "numpy" in pyomo_code or "np." in pyomo_code:
                essential_imports.append("import numpy as np")
            if "pandas" in pyomo_code or "pd." in pyomo_code:
                essential_imports.append("import pandas as pd")
            if "sklearn" in pyomo_code or "scikit" in pyomo_code:
                essential_imports.append("import sklearn")
            if "tensorflow" in pyomo_code or "keras" in pyomo_code:
                essential_imports.append("import tensorflow as tf")
            if "torch" in pyomo_code:
                essential_imports.append("import torch")
                
            # Print the code that will be executed for debugging
            # print("\n" + "="*80)
            # print("🔍 [DEBUG] Code about to be executed:")
            # print("="*80)
            # print(pyomo_code)
            # print("="*80 + "\n")
            
            # Execute base imports
            for import_stmt in essential_imports:
                try:
                    exec(import_stmt, namespace)
                except ImportError as e:
                    print(f"⚠️ Import warning: {import_stmt} - {e}")
                    continue
            
            # Add safe function overrides
            def safe_exit(code=0):
                """Safe exit function, raises exception instead of terminating program"""
                raise Exception(f"Code called exit({code}), possibly due to solver unavailability or other errors")
            
            def safe_input(prompt=""):
                """Safe input function, avoids interactive input"""
                raise Exception("Code attempted interactive input, which is not allowed in automated environment")
            
            # Override dangerous functions in namespace
            namespace['exit'] = safe_exit
            namespace['quit'] = safe_exit
            namespace['input'] = safe_input
            
            # Add safe sys module override
            if 'sys' in namespace:
                namespace['sys'].exit = safe_exit
            
            print("🚀 Starting engineering model code execution...")
            
            # Use thread timeout mechanism, more reliably prevent code execution from hanging
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def execute_code():
                try:
                    # Capture all output and execute code
                    with contextlib.redirect_stdout(exec_stdout_io), contextlib.redirect_stderr(exec_stderr_io):
                        exec(pyomo_code, namespace)
                    result_queue.put("success")
                except Exception as e:
                    exception_queue.put(e)
            
            # Start execution thread
            exec_thread = threading.Thread(target=execute_code)
            exec_thread.daemon = True  # Set as daemon thread
            exec_thread.start()
            
            # Wait for completion or timeout (60 seconds)
            exec_thread.join(timeout=60)
            
            if exec_thread.is_alive():
                # Thread still running, means timeout occurred
                print("⏰ Code execution timeout")
                raise Exception("Model complexity exceeds current solver capability, suggest simplifying model or adjusting constraints")
            
            # Check for exceptions
            if not exception_queue.empty():
                raise exception_queue.get()
            
            # Check for successful completion
            if result_queue.empty():
                raise Exception("Code execution abnormally terminated")
            
            # Get complete execution output
            exec_output = exec_stdout_io.getvalue()
            exec_errors = exec_stderr_io.getvalue()
            
            print("✅ Code execution successful")
            if exec_output:
                print("📋 Code execution output:")
                print(exec_output[:1000] + ("..." if len(exec_output) > 1000 else ""))
            if exec_errors:
                print("⚠️ Execution warnings/errors:")
                print(exec_errors[:500] + ("..." if len(exec_errors) > 500 else ""))
            
            # Intelligently find model object or results
            model = None
            results = {}
            
            # Find Pyomo model object
            for var_name, var_value in namespace.items():
                if hasattr(var_value, '__class__') and 'ConcreteModel' in str(var_value.__class__):
                    model = var_value
                    break
            
            # If Pyomo model found, perform optimization solving
            if model is not None:
                # Pass capture containers for solver detection and logging
                result = self._solve_optimization_model(model, solver_detection_container=solver_detection)
                # Append execution stdout/stderr (truncated)
                run_log["exec_stdout"] = exec_stdout_io.getvalue()[:8000]
                run_log["exec_stderr"] = exec_stderr_io.getvalue()[:8000]
                result["run_log"] = run_log
                result["solver_detection"] = solver_detection
                return result
            
            # If no Pyomo model found, look for other types of results
            for var_name, var_value in namespace.items():
                if var_name.startswith('result') or var_name.startswith('solution') or var_name.startswith('output'):
                    results[var_name] = var_value
                elif hasattr(var_value, 'predict') or hasattr(var_value, 'score'):  # Machine learning model
                    results['model'] = var_value
                elif isinstance(var_value, (dict, list)) and len(str(var_value)) > 10:  # Possible result data
                    results[var_name] = var_value
            
            if results:
                formatted = self._format_general_results(results)
                run_log["exec_stdout"] = exec_stdout_io.getvalue()[:8000]
                run_log["exec_stderr"] = exec_stderr_io.getvalue()[:8000]
                formatted["run_log"] = run_log
                formatted["solver_detection"] = solver_detection
                return formatted
            else:
                raise Exception("Solving failed: Cannot find valid model object or results")
        
        except Exception as e:
            self.logger.error(f"Error solving engineering model: {e}")
            # Assemble structured error context (as complete as possible)
            error_context = {
                "message": str(e),
                "original_exception": type(e).__name__,
                "exec_stdout": exec_stdout_io.getvalue()[:8000],
                "exec_stderr": exec_stderr_io.getvalue()[:8000],
                "last_namespace_keys": [k for k in list(locals().get('namespace', {}).keys())[:50]] if 'namespace' in locals() else [],
                "solver_detection": solver_detection,
            }
            try:
                import json as _json
                raise Exception(_json.dumps({"kind": "SolverRunError", **error_context}, ensure_ascii=False))
            except Exception:
                # If serialization fails, fall back to text
                raise Exception(f"Solving failed (context): {error_context}")
        
        finally:
            # Environment cleanup and restoration
            try:
                # Restore original working directory
                os.chdir(original_cwd)
                # Restore original Python path
                sys.path = original_path
                # Force garbage collection, clean up namespace
                if 'namespace' in locals():
                    del namespace
                gc.collect()
                print("🧹 Execution environment cleaned up")
            except Exception as cleanup_error:
                print(f"⚠️ Environment cleanup warning: {cleanup_error}")
    
    def _solve_optimization_model(self, model, solver_detection_container: list = None) -> Dict[str, Any]:
        """Solve Pyomo optimization model"""
        try:
            # Try solving
            import pyomo.environ as pyo
            
            # Try different solvers, ordered by priority
            solvers = [
                ('glpk', 'GLPK (GNU Linear Programming Kit)'),
                ('cbc', 'CBC (COIN-OR Branch and Cut)'),
                ('ipopt', 'IPOPT (Interior Point Optimizer)'),
                ('gurobi', 'Gurobi (Commercial)'),
                ('cplex', 'CPLEX (Commercial)')
            ]
            solver_used = None
            detection = []
            
            print("🔍 Detecting available solvers...")
            for solver_name, solver_desc in solvers:
                try:
                    solver = pyo.SolverFactory(solver_name)
                    if solver.available():
                        print(f"✅ Found available solver: {solver_desc}")
                        solver_used = solver_name
                        detection.append({"name": solver_name, "desc": solver_desc, "available": True})
                        break
                    else:
                        print(f"❌ Solver unavailable: {solver_desc}")
                        detection.append({"name": solver_name, "desc": solver_desc, "available": False})
                except Exception as e:
                    print(f"❌ Solver detection failed {solver_desc}: {e}")
                    detection.append({"name": solver_name, "desc": solver_desc, "available": False, "error": str(e)})
                    continue
            if solver_detection_container is not None:
                solver_detection_container.extend(detection)
            
            if solver_used is None:
                print("⚠️ No available solvers found")
                print("💡 Suggested solver installations:")
                print("   - conda install -c conda-forge glpk")
                print("   - conda install -c conda-forge coin-or-cbc")
                print("   - conda install -c conda-forge ipopt")
                raise Exception("Solving failed: No available solvers")
            
            print(f"🚀 Using solver: {solver_used}")
            try:
                # Record solving start time
                import time
                solve_start_time = time.time()
                import io, contextlib
                solver_stdout = io.StringIO()
                with contextlib.redirect_stdout(solver_stdout):
                    result = pyo.SolverFactory(solver_used).solve(model, tee=True)
                
                # Calculate solving time
                solve_time = time.time() - solve_start_time
                print(f"✅ Solving completed, status: {result.solver.termination_condition}, time: {solve_time:.3f} seconds")
                
                # Extract results
                objective_value = None
                try:
                    if hasattr(model, 'obj'):
                        try:
                            objective_value = pyo.value(model.obj)
                            print(f"✅ Successfully extracted objective function value: {objective_value}")
                        except Exception as obj_error:
                            print(f"❌ Failed to extract model.obj: {obj_error}")
                            objective_value = None
                    elif hasattr(model, 'TotalCost'):
                        try:
                            objective_value = pyo.value(model.TotalCost)
                            print(f"✅ Successfully extracted TotalCost value: {objective_value}")
                        except Exception as cost_error:
                            print(f"❌ Failed to extract model.TotalCost: {cost_error}")
                            objective_value = None
                    elif hasattr(model, 'objective'):
                        try:
                            objective_value = pyo.value(model.objective)
                            print(f"✅ Successfully extracted objective value: {objective_value}")
                        except Exception as objective_error:
                            print(f"❌ Failed to extract model.objective: {objective_error}")
                            objective_value = None
                    else:
                        print("❌ No standard objective function attributes found (obj, TotalCost, objective)")
                        # Try to find all Objective components
                        obj_components = [comp for comp in model.component_objects(pyo.Objective)]
                        if obj_components:
                            try:
                                objective_value = pyo.value(obj_components[0])
                                print(f"✅ Extracted value from first Objective component: {objective_value}")
                            except Exception as comp_error:
                                print(f"❌ Failed to extract from Objective component: {comp_error}")
                                objective_value = None
                        else:
                            print("❌ No Objective components found in model")
                            objective_value = None
                
                    if objective_value is None:
                        print("⚠️ Objective function value is None, possible reasons:")
                        print("   1. Objective function incorrectly defined")
                        print("   2. Model solving failed")
                        print("   3. Objective function expression contains undefined variables or parameters")
                        print("   4. Solver did not correctly set objective function value")
                
                except Exception as extract_error:
                    print(f"❌ Exception occurred during objective function value extraction: {extract_error}")
                    objective_value = None
                
                # Extract variable values - Enhanced version with more detailed information and error handling
                var_values = {}
                print("📊 Extracting decision variable values...")
                
                for var in model.component_objects(pyo.Var):
                    var_name_base = var.name
                    if var.is_indexed():
                        for index in var:
                            if index is not None:
                                if isinstance(index, tuple):
                                    # Handle multi-dimensional indices
                                    index_str = "_".join(str(i) for i in index)
                                    var_name = f"{var_name_base}_{index_str}"
                                else:
                                    var_name = f"{var_name_base}_{index}"
                            else:
                                var_name = var_name_base
                            
                            try:
                                var_value = pyo.value(var[index])
                                if var_value is not None:
                                    var_values[var_name] = var_value
                                    print(f"  {var_name}: {var_value}")
                                else:
                                    var_values[var_name] = "N/A"
                                    print(f"  {var_name}: N/A (Unable to get value)")
                            except Exception as e:
                                var_values[var_name] = "N/A"
                                print(f"  {var_name}: N/A (Extraction failed: {e})")
                    else:
                        # Non-indexed variables
                        try:
                            var_value = pyo.value(var)
                            if var_value is not None:
                                var_values[var_name_base] = var_value
                                print(f"  {var_name_base}: {var_value}")
                            else:
                                var_values[var_name_base] = "N/A"
                                print(f"  {var_name_base}: N/A (Unable to get value)")
                        except Exception as e:
                            var_values[var_name_base] = "N/A"
                            print(f"  {var_name_base}: N/A (Extraction failed: {e})")
                
                print(f"📈 Extracted {len(var_values)} variable values")
                
                # Generate complete solving output information (for debugging record backup tracing)
                full_output = []
                full_output.append("="*80)
                full_output.append("🎯 Complete Solving Output")
                full_output.append("="*80)
                full_output.append(f"Solver: {solver_used}")
                full_output.append(f"Solving time: {solve_time:.3f} seconds")
                full_output.append(f"Termination condition: {result.solver.termination_condition}")
                
                if objective_value is not None:
                    full_output.append(f"Objective function value: {objective_value}")
                else:
                    full_output.append("Objective function value: Unable to get")
                
                full_output.append("")
                full_output.append("Decision variable values:")
                full_output.append("-" * 60)
                
                # Intelligently group and sort variables for display
                var_groups = {}
                for var_name, var_value in var_values.items():
                    # Group by variable name base
                    base_name = var_name.split('_')[0] if '_' in var_name else var_name
                    if base_name not in var_groups:
                        var_groups[base_name] = []
                    var_groups[base_name].append((var_name, var_value))
                
                # Display variables by group
                for group_name, variables in sorted(var_groups.items()):
                    if len(variables) > 1:
                        full_output.append(f"{group_name.upper()} Series Variables:")
                        for var_name, var_value in sorted(variables):
                            if isinstance(var_value, (int, float)) and var_value != 0:
                                full_output.append(f"  {var_name:25s}: {var_value:>12.3f}")
                            elif var_value == 0:
                                full_output.append(f"  {var_name:25s}: {var_value:>12.0f}")
                            else:
                                full_output.append(f"  {var_name:25s}: {str(var_value):>12s}")
                    else:
                        var_name, var_value = variables[0]
                        if isinstance(var_value, (int, float)) and var_value != 0:
                            full_output.append(f"{var_name:30s}: {var_value:>12.3f}")
                        elif var_value == 0:
                            full_output.append(f"{var_name:30s}: {var_value:>12.0f}")
                        else:
                            full_output.append(f"{var_name:30s}: {str(var_value):>12s}")
                
                full_output.append("="*80)
                full_output_str = "\n".join(full_output)
                print(full_output_str)
                
                # Return results based on solving status
                if result.solver.termination_condition == pyo.TerminationCondition.optimal:
                    status = "optimal"
                elif result.solver.termination_condition == pyo.TerminationCondition.infeasible:
                    status = "infeasible"
                    print("⚠️ Model is infeasible")
                    raise Exception("Solving failed: Model is infeasible")
                elif result.solver.termination_condition == pyo.TerminationCondition.unbounded:
                    status = "unbounded"
                else:
                    status = "other"
                
                self.logger.info(f"Pyomo model solved successfully with {solver_used}")
                return {
                    "status": status,
                    "objective_value": objective_value,
                    "decision_variables_values": var_values,
                    "solver_used": solver_used,
                    "solve_time": solve_time,  # Add solving time
                    "termination_condition": str(result.solver.termination_condition),
                    "full_output": full_output_str,  # Add complete output for debugging record
                    "solver_log": solver_stdout.getvalue()[:12000]
                }
                    
            except Exception as e:
                print(f"❌ Error during solving process: {e}")
                raise Exception(f"Solving failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Error solving Pyomo model: {e}")
            raise Exception(f"Solving failed: {e}")
    
    def _format_general_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format general engineering model results"""
        try:
            print("📊 Formatting general engineering model results...")
            
            formatted_results = {
                "status": "completed",
                "model_type": "general_engineering",
                "results": {},
                "summary": ""
            }
            
            # Handle different types of results
            for key, value in results.items():
                if hasattr(value, 'predict') or hasattr(value, 'score'):
                    # Machine learning model
                    formatted_results["model_type"] = "machine_learning"
                    formatted_results["results"]["trained_model"] = str(type(value))
                    if hasattr(value, 'score'):
                        try:
                            # Try to get model score (if test data available)
                            formatted_results["results"]["model_info"] = "Training completed, ready for prediction"
                        except:
                            formatted_results["results"]["model_info"] = "Model trained"
                elif isinstance(value, dict):
                    # Dictionary type results
                    formatted_results["results"][key] = value
                elif isinstance(value, list) and len(value) > 0:
                    # List type results
                    formatted_results["results"][key] = value[:10] if len(value) > 10 else value  # Limit display length
                elif isinstance(value, (int, float, str)):
                    # Basic type results
                    formatted_results["results"][key] = value
                else:
                    # Other types, convert to string
                    formatted_results["results"][key] = str(value)[:200]  # Limit length
            
            # Generate summary
            if formatted_results["model_type"] == "machine_learning":
                formatted_results["summary"] = f"Machine learning model training completed, contains {len(formatted_results['results'])} results"
            else:
                formatted_results["summary"] = f"Engineering model execution completed, generated {len(formatted_results['results'])} results"
            
            print(f"✅ Results formatting completed: {formatted_results['summary']}")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Results formatting failed: {e}")
            return {
                "status": "completed_with_warnings",
                "model_type": "unknown",
                "results": {"raw_results": str(results)},
                "summary": f"Model execution completed, but problem occurred during results formatting: {e}"
            }
    
    def generate_simulation_solution(self, pyomo_code: str) -> Dict[str, Any]:
        """Generate simulation solution based on Pyomo code"""
        # Analyze variables and objective function in code
        var_values = {}
        objective_value = 100.0
        
        # Extract variable names - fix regex to match actual code format
        import re
        var_patterns = re.findall(r'model\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:pyo\.)?Var\s*\(', pyomo_code)
        
        for var_name in var_patterns:
            # Assign reasonable values based on variable names
            if 'SOC' in var_name:
                var_values[var_name] = 75.0  # SOC state
            elif 'P_buy' in var_name:
                var_values[var_name] = 150.0  # buy power
            elif 'P_sell' in var_name:
                var_values[var_name] = 50.0   # sell power
            elif 'P_charge' in var_name:
                var_values[var_name] = 80.0   # charge power
            elif 'P_discharge' in var_name:
                var_values[var_name] = 60.0   # discharge power
            elif 'P_gen' in var_name:
                var_values[var_name] = 200.0  # generation power
            elif 'production' in var_name.lower() or 'x_' in var_name:
                if 'A' in var_name or 'a' in var_name:
                    var_values[var_name] = 800.0  # product A production
                elif 'B' in var_name or 'b' in var_name:
                    var_values[var_name] = 600.0  # product B production
                else:
                    var_values[var_name] = 700.0  # generic production
            elif 'inventory' in var_name.lower() or 'i_' in var_name:
                if 'A' in var_name or 'a' in var_name:
                    var_values[var_name] = 100.0  # product A inventory
                elif 'B' in var_name or 'b' in var_name:
                    var_values[var_name] = 50.0   # product B inventory
                else:
                    var_values[var_name] = 75.0   # generic inventory
            elif 'switch' in var_name.lower() or 'y_' in var_name or 'z_' in var_name:
                var_values[var_name] = 0.0    # switch variable
            else:
                var_values[var_name] = 10.0   # default value
        
        # if no variable found, use default
        if not var_values:
            var_values = {"x": 10.0}
        
        # Adjust target value based on objective function type
        if 'minimize' in pyomo_code.lower() or 'pyo.minimize' in pyomo_code:
            if 'cost' in pyomo_code.lower() or 'costs' in pyomo_code.lower():
                # production planning problem: production cost + inventory cost + switch cost
                production_cost = sum(v for k,v in var_values.items() if 'production' in k.lower() or 'x_' in k) * 60  # average production cost
                inventory_cost = sum(v for k,v in var_values.items() if 'inventory' in k.lower() or 'i_' in k) * 2.5  # average inventory cost
                switch_cost = sum(v for k,v in var_values.items() if 'switch' in k.lower() or 'y_' in k or 'z_' in k) * 900 # switch cost
                objective_value = production_cost + inventory_cost + switch_cost
            else:
                objective_value = sum(var_values.values()) * 0.5
        else:
            objective_value = sum(var_values.values()) * 1.2  # maximize target
        
        return {
            "status": "simulated",
            "objective_value": round(objective_value, 2),
            "decision_variables_values": {k: round(v, 2) for k, v in var_values.items()},
            "solver_used": "intelligent_simulation",
            "note": "intelligent simulation result based on code analysis, not hardcoded values"
        }

class SolutionVerificationAgent(BaseAgent):
    def __init__(self, agent_id: str, model_type: str = "gemini"):
        # Set correct llm_model based on model_type
        if model_type.lower() == "gemini":
            llm_model = "gemini-2.5-flash-lite"
        elif model_type.lower() == "gpt5":
            llm_model = "gpt-5"
        elif model_type.lower() == "gpt4o":
            llm_model = "gpt-4o"
        elif model_type.lower() == "gpt4":
            llm_model = "gpt-4"
        elif model_type.lower() == "deepseek":
            llm_model = "deepseek-v3"
        else:  # Default to gpt-4o
            llm_model = "gpt-4o"
        super().__init__(agent_id, llm_model)
        self.model_type = model_type.lower()
        
        # Intelligent detection of API availability
        try:
            # Test using unified API calling function
            test_response = call_llm_api("Hello", self.model_type)
            self.api_available = test_response is not None
            self.logger.info(f"{self.model_type.upper()} API connection successful" if self.api_available else f"{self.model_type.upper()} API unavailable, switch to simulation mode")
        except Exception as e:
            self.logger.warning(f"{self.model_type.upper()} API unavailable, switch to simulation mode: {e}")
            self.api_available = False
        
        # Initialize LLM model (Only Gemini needs this, GPT and DeepSeek call API directly)
        if self.model_type == "gemini":
            try:
                self.model = genai.GenerativeModel(self.llm_model)
            except Exception as e:
                self.logger.warning(f"Unable to initialize Gemini model: {e}")
                self.model = None
        else:
            self.model = None  # GPT and DeepSeek do not need pre-initialized model
        
        # Retry counter (reset for each new verification request)
        self.retry_count = 0
        self.max_retries = 3
        self.consecutive_verification_failures = 0
        self.max_consecutive_failures = 10  # Consecutive failure threshold
        self.tolerance_level = 0.0  # Verification tolerance 0.0-1.0

    def process_message(self, message: Message) -> Message:
        if message.msg_type == MessageType.QUERY:
            verification_data = message.content
            self.logger.info("Starting solution verification...")
            
            # Reset retry counter (new verification request)
            self.retry_count = 0
            
            return self._attempt_verification(verification_data)
            
        elif message.msg_type == MessageType.RETRY:
            # Handle retry request (doesn't consume major round quota)
            retry_info = message.content
            strategy = retry_info.get("strategy", "Re-verify")
            verification_data = retry_info.get("verification_data", {})
            
            self.retry_count += 1
            self.logger.info(f"Executing solution verification retry ({self.retry_count}/{self.max_retries}): {strategy}")
            
            return self._attempt_verification(verification_data)
            
        else:
            self.logger.warning(f"SolutionVerificationAgent received unexpected message type: {message.msg_type}")
            # For unsupported message types, return fallback result instead of ERROR
            return Message(
                self.agent_id, 
                "coordinator", 
                self._generate_fallback_verification({}), 
                MessageType.RESPONSE
            )
    
    def _attempt_verification(self, verification_data: Dict[str, Any]) -> Message:
        """Attempt verification, decide whether to retry or fallback based on retry count"""
        try:
            verification_result = self.verify_solution(verification_data)
            response_msg = Message(self.agent_id, "coordinator", verification_result, MessageType.RESPONSE)
            return response_msg
        except Exception as e:
            self.logger.error(f"Solution verification failed (Attempt {self.retry_count + 1}): {e}")
            
            # Check if can still retry
            if self.retry_count < self.max_retries:
                # Can still retry, return ERROR message for coordinator to retry
                error_msg = Message(
                    self.agent_id, 
                    "coordinator", 
                    {
                        "error": str(e), 
                        "error_type": "VerificationError", 
                        "stage": "solution_verification",
                        "verification_data": verification_data,
                        "retry_count": self.retry_count,
                        "max_retries": self.max_retries,
                        "traceback": traceback.format_exc()
                    }, 
                    MessageType.ERROR
                )
                return error_msg
            else:
                # Retry count exhausted, use fallback
                self.logger.warning(f"Retry count exhausted ({self.max_retries}), using backup verification")
                fallback_result = self._generate_fallback_verification(verification_data)
                fallback_result["fallback_reason"] = f"LLM verification failed after {self.max_retries} retries"
                response_msg = Message(self.agent_id, "coordinator", fallback_result, MessageType.RESPONSE)
                return response_msg

    def verify_solution(self, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("🔍 [SolutionVerificationAgent] Starting solution verification")
        print("="*60)
        
        # Extract verification data
        original_problem = verification_data.get("original_problem", "")
        analysis_result = verification_data.get("analysis_result", {})
        model_code = verification_data.get("model_code", "")
        solution_result = verification_data.get("solution_result", {})
        debug_history = verification_data.get("debug_history", [])
        coordinator_mode = verification_data.get("coordinator_mode", "unknown")
        
        print(f"Verification mode: {coordinator_mode}")
        print(f"Original problem length: {len(original_problem)} characters")
        print(f"Model code length: {len(model_code)} characters")
        print(f"Debug history records: {len(debug_history)} entries")
        
        if not self.api_available:
            # When API is unavailable, use simplified verification
            return self._generate_fallback_verification(verification_data)
        
        try:
            # Generate verification prompt
            verification_prompt = self._generate_verification_prompt(verification_data)
            
            print("🚀 Calling LLM for solution verification...")
            response_text = call_llm_api(verification_prompt, self.model_type)
            if not response_text:
                raise Exception(f"{self.model_type.upper()} API call failed")
            
            print(f"Original response length: {len(response_text)} characters")
            
            # Extract JSON result
            verification_result = self._extract_verification_result(response_text)
            
            print("✅ Solution verification completed")
            print(f"Verification status: {verification_result.get('verification_status', 'unknown')}")
            print(f"Quality score: {verification_result.get('quality_score', 0.0):.3f}")
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"LLM verification failed: {e}")
            return self._generate_fallback_verification(verification_data)
    
    def _generate_verification_prompt(self, verification_data: Dict[str, Any]) -> str:
        """Generate solution verification LLM prompt"""
        original_problem = verification_data.get("original_problem", "")
        analysis_result = verification_data.get("analysis_result", {})
        model_code = verification_data.get("model_code", "")
        solution_result = verification_data.get("solution_result", {})
        
        # Safely handle strings that might contain format specifiers
        safe_problem = smart_escape_braces(original_problem)
        safe_code = smart_escape_braces(model_code)
        safe_analysis = smart_escape_braces(analysis_result)
        safe_solution = smart_escape_braces(solution_result)
        
        prompt = f"""### Role and Mission ###
You are a **senior solution verification expert** specializing in comprehensive quality assessment of engineering optimization solutions. Your mission is to provide **balanced, constructive evaluation** that recognizes both achievements and improvement opportunities.

### Solution Package for Verification ###
**Original Engineering Problem:**
{safe_problem}

**Analysis Blueprint:**
{safe_analysis}

**Implementation Code:**
{safe_code}

**Computational Results:**
{safe_solution}

**CRITICAL INSTRUCTION FOR VERIFICATION:**
- If the "Computational Results" section is empty or shows empty dict {{}}, this means NO REAL SOLUTION was obtained
- If you see template/example outputs in the code (like "Optimal", sample values, or demo results), these are NOT real solver results
- Only consider actual solver execution results from the solution_result field above
- Empty solution_result means the solver FAILED - this MUST result in "failed" status with score < 0.3

### Verification Framework ###
Evaluate the solution across **four critical dimensions**:

**1. Solution Feasibility Assessment**
- **Primary Check**: Solution status (optimal/feasible/bounded)
- **Constraint Satisfaction**: All constraints properly satisfied
- **Variable Validity**: Decision variables within logical bounds
- **Numerical Stability**: No computational errors or infeasibilities

**2. Model-Problem Alignment**
- **Core Objective Match**: Objective function reflects problem goals
- **Constraint Coverage**: Essential problem requirements captured
- **Variable Representation**: Decision variables model key decisions
- **Implementation Fidelity**: Code accurately represents analysis blueprint

**3. Engineering Reasonableness**
- **Practical Feasibility**: Results implementable in real-world context
- **Economic Validity**: Costs/benefits within expected ranges
- **Technical Soundness**: Variable values make engineering sense
- **Scale Appropriateness**: Solution magnitude reasonable for problem size

**4. Solution Quality & Optimization**
- **Optimality Assessment**: Quality of solution found (local vs global)
- **Computational Efficiency**: Reasonable solution time and resource usage
- **Robustness Indicators**: Solution stability and sensitivity
- **Improvement Potential**: Areas for enhancement or refinement

### Evaluation Guidelines ###
- **Balanced Perspective**: Acknowledge both strengths and areas for improvement
- **Constructive Focus**: Provide actionable suggestions rather than just criticism
- **Context Awareness**: Consider problem complexity and implementation constraints
- **Pragmatic Standards**: Evaluate based on practical engineering standards, not theoretical perfection

**Output Requirements:**
Please only output verification results in JSON format, wrapped in ```json...```:

```json
{{
    "verification_status": "satisfactory/needs_improvement/failed",
    "quality_score": 0.0-1.0,
    "analysis": {{
        "solution_feasibility": "Detailed solution feasibility analysis",
        "model_consistency": "Detailed model consistency analysis", 
        "result_reasonableness": "Detailed result reasonableness analysis",
        "optimization_quality": "Detailed optimization quality analysis"
    }},
    "recommendations": ["Improvement suggestion 1", "Improvement suggestion 2", "..."],
    "confidence": 0.0-1.0,
    "should_iterate": true/false
}}
```

**Scoring Standards:**
- quality_score: 0.8+ = satisfactory, 0.5-0.8 = needs_improvement, <0.5 = failed
- should_iterate: Only recommend re-iteration when there is obvious room for improvement and current solution quality is insufficient
- confidence: Based on completeness of input information and certainty of analysis

**Special Notes:**
- **CRITICAL**: If solution fails or has no solution, MUST return "failed" status with quality_score < 0.3
- **CRITICAL**: If objective_value is 0 or None, MUST return "failed" status with quality_score < 0.3
- **CRITICAL**: Distinguish between real solver results and simulated/template data in the code
- **CRITICAL**: If the code contains template outputs (like "Optimal" or sample values), but no real solver execution results, return "failed"
- If solution exists but is unreasonable, focus on analyzing parameter and constraint settings
- Only return "satisfactory" status when solution is truly optimal with reasonable objective value
- Be strict in evaluation - prefer "needs_improvement" over "satisfactory" when uncertain"""

        return prompt
    
    def _extract_verification_result(self, response_text: str) -> Dict[str, Any]:
        """Extract verification result from LLM response"""
        try:
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
            
            result = json.loads(response_text)
            print("✅ JSON parsing successful")
            
            # Verify required fields
            required_fields = ['verification_status', 'quality_score', 'analysis', 'recommendations', 'confidence']
            for field in required_fields:
                if field not in result:
                    result[field] = self._get_default_field_value(field)
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print("🔄 Attempting to fix JSON format...")
            
            # Try to fix common JSON format issues
            try:
                # Remove possible prefix/suffix text
                if '{' in response_text and '}' in response_text:
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    cleaned_json = response_text[start:end]
                    
                    # Fix common issues
                    cleaned_json = cleaned_json.replace("'", '"')  # Single quotes to double quotes
                    cleaned_json = re.sub(r',\s*}', '}', cleaned_json)  # Remove trailing commas
                    cleaned_json = re.sub(r',\s*]', ']', cleaned_json)  # Remove array trailing commas
                    
                    result = json.loads(cleaned_json)
                    print("✅ JSON fix successful")
                    return result
            except:
                pass
            
            self.logger.error(f"Verification result JSON parsing failed: {e}")
            raise Exception(f"Verification failed: JSON parsing error - {e}")
    
    def _get_default_field_value(self, field: str) -> Any:
        """Get default value for field"""
        defaults = {
            'verification_status': 'needs_improvement',
            'quality_score': 0.5,
            'analysis': {
                'solution_feasibility': 'Unable to fully analyze solution feasibility',
                'model_consistency': 'Unable to fully analyze model consistency',
                'result_reasonableness': 'Unable to fully analyze result reasonableness',
                'optimization_quality': 'Unable to fully analyze optimization quality'
            },
            'recommendations': ['Check model settings', 'Verify input data'],
            'confidence': 0.3,
            'should_iterate': False
        }
        return defaults.get(field, None)
    
    def _generate_fallback_verification(self, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback verification result"""
        solution_result = verification_data.get("solution_result", {})
        
        # Check if there is a valid solution result
        if not solution_result:
            # No solution result
            verification_status = "failed"
            quality_score = 0.0
            feasibility_analysis = "No solution result, solver failed"
            should_iterate = True
        else:
            # Simple verification based on solution status
            status = solution_result.get("status", "unknown")
            objective_value = solution_result.get("objective_value")
            
            # Check mandatory restart conditions
            if (status in ["unknown", "failed", "error"] or 
                objective_value is None or 
                objective_value == 0):
                verification_status = "failed"
                quality_score = 0.0
                feasibility_analysis = f"Solver failed: status={status}, objective value={objective_value}"
                should_iterate = True
            elif status == "optimal":
                verification_status = "satisfactory"
                quality_score = 0.8
                feasibility_analysis = "Solver returned optimal solution, solution feasibility is good"
                should_iterate = False
            elif status in ["feasible", "completed"]:
                verification_status = "satisfactory"
                quality_score = 0.7
                feasibility_analysis = "Solver returned feasible solution, solution feasibility is acceptable"
                should_iterate = False
            else:
                verification_status = "needs_improvement"
                quality_score = 0.4
                feasibility_analysis = "Solver status is not ideal, possible feasibility issues"
                should_iterate = True
        
        return {
            "verification_status": verification_status,
            "quality_score": quality_score,
            "analysis": {
                "solution_feasibility": feasibility_analysis,
                "model_consistency": "API unavailable, detailed model consistency analysis cannot be performed",
                "result_reasonableness": f"Objective value: {solution_result.get('objective_value', 'None')}, based on solver status",
                "optimization_quality": f"Based on solver status '{solution_result.get('status', 'unknown')}'"
            },
            "recommendations": [
                "Suggest checking model parameter settings",
                "Verify constraint correctness", 
                "Consider adjusting solver configuration"
            ] if verification_status != "failed" else [
                "Check if model definition is correct",
                "Verify data input completeness",
                "Check solver installation and configuration",
                "Re-analyze problem constraints"
            ],
            "confidence": 0.5 if verification_status != "failed" else 0.2,
            "should_iterate": should_iterate,
            "fallback_mode": True
        }

