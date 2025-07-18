"""
LLM Services module for the DSP+ Agent.
Manages all interactions with different LLM APIs.
"""

from typing import Dict, Any, List, Optional
from openai import OpenAI
from . import prompts


class LLMServices:
    """
    Manages all LLM interactions for the DSP+ Agent.
    Handles different models for different stages of the pipeline.
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize the LLM Services.
        
        Args:
            llm_config: Configuration dictionary containing LLM provider settings
        """
        self.config = llm_config
        
        # Initialize clients for different stages
        self.draft_client = self._create_client(llm_config["draft_model"])
        self.sketch_client = self._create_client(llm_config["sketch_model"])
        self.prove_client = self._create_client(llm_config["prove_model"])
        
        # Store model names
        self.draft_model_name = llm_config["draft_model"]["model_name"]
        self.sketch_model_name = llm_config["sketch_model"]["model_name"]
        self.prove_model_name = llm_config["prove_model"]["model_name"]
    
    def _create_client(self, model_config: Dict[str, str]) -> OpenAI:
        """
        Create an OpenAI client from model configuration.
        
        Args:
            model_config: Configuration for a specific model
            
        Returns:
            Configured OpenAI client
        """
        return OpenAI(
            base_url=model_config["api_url"],
            api_key=model_config["api_key"]
        )
    
    def generate_draft(self, formal_statement: str) -> str:
        """
        Generate a natural-language proof plan using the draft model.
        
        Args:
            formal_statement: The formal theorem statement
            
        Returns:
            Natural-language proof plan
        """
        prompt = prompts.DRAFT_PROMPT.format(formal_statement=formal_statement)
        
        messages = [
            {"role": "system", "content": "You are a brilliant mathematician specializing in proof strategies."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.draft_client.chat.completions.create(
                model=self.draft_model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                n=1,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating draft: {e}"
    
    def generate_sketch(self, formal_statement: str, concise_draft: str) -> str:
        """
        Generate a Lean 4 sketch using the sketch model.
        
        Args:
            formal_statement: The formal theorem statement
            concise_draft: The natural-language proof plan
            
        Returns:
            Lean 4 sketch with 'by sorry' placeholders
        """
        prompt = prompts.SKETCH_PROMPT.format(
            formal_statement=formal_statement,
            concise_draft=concise_draft
        )
        
        messages = [
            {"role": "system", "content": "You are an expert Lean 4 programmer specializing in proof formalization."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.sketch_client.chat.completions.create(
                model=self.sketch_model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=8192,
                n=1,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating sketch: {e}"
    
    def generate_sketch_with_masking(self, formal_statement: str, header: str, draft: str) -> str:
        """
        Generate a sketch using the masking-style prompt from sketch.py.
        
        Args:
            formal_statement: The formal theorem statement
            header: The header/imports
            draft: The natural-language proof plan
            
        Returns:
            Raw sketch response from the model
        """
        prompt = prompts.SKETCH_PROMPT_WITH_MASKING.format(
            draft=draft,
            header=header,
            formal_statement=formal_statement
        )
        
        messages = [
            {"role": "system", "content": "You are an expert Lean 4 programmer specializing in proof formalization."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.sketch_client.chat.completions.create(
                model=self.sketch_model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=8192,
                n=1,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating sketch with masking: {e}"
    
    def extract_subgoals(self, sketch: str, formal_statement: str) -> List[str]:
        """
        Extract subgoals from sketch using LLM-based analysis.
        
        Args:
            sketch: The Lean 4 sketch
            formal_statement: The original formal theorem statement
            
        Returns:
            List of subgoal statements formatted as standalone theorems
        """
        prompt = prompts.SUBGOAL_EXTRACTION_PROMPT.format(
            formal_statement=formal_statement,
            sketch=sketch
        )
        
        messages = [
            {"role": "system", "content": "You are an expert Lean 4 analyst specializing in proof structure extraction and context analysis."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.sketch_client.chat.completions.create(
                model=self.sketch_model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
                n=1,
            )
            
            extraction_result = response.choices[0].message.content.strip()
            return self._parse_subgoals_from_response(extraction_result)
            
        except Exception as e:
            print(f"❌ Error in LLM-based subgoal extraction: {e}")
            return []
    
    def _parse_subgoals_from_response(self, extraction_result: str) -> List[str]:
        """
        Parse subgoals from LLM response.
        
        Args:
            extraction_result: Raw response from subgoal extraction
            
        Returns:
            List of parsed subgoal statements
        """
        subgoals = []
        
        # Split by lines and find theorem declarations
        lines = extraction_result.split('\n')
        current_theorem = []
        in_theorem = False
        import_prefix = "import Mathlib"
        
        for line in lines:
            stripped_line = line.strip()
            
            # Start of a new theorem
            if stripped_line.startswith('theorem subgoal_') and ':' in stripped_line:
                # If we were building a previous theorem, finish it
                if in_theorem and current_theorem:
                    theorem_text = '\n'.join(current_theorem)
                    if 'sorry' in theorem_text:
                        # Add import if not present
                        if not theorem_text.startswith('import'):
                            theorem_text = f"{import_prefix}\n\n{theorem_text}"
                        subgoals.append(theorem_text.strip())
                
                # Start new theorem
                current_theorem = [line]
                in_theorem = True
                
            # Continue building current theorem
            elif in_theorem:
                current_theorem.append(line)
                
                # End theorem when we hit 'sorry'
                if 'sorry' in stripped_line:
                    theorem_text = '\n'.join(current_theorem)
                    # Add import if not present
                    if not theorem_text.startswith('import'):
                        theorem_text = f"{import_prefix}\n\n{theorem_text}"
                    subgoals.append(theorem_text.strip())
                    current_theorem = []
                    in_theorem = False
            
            # Handle import statements
            elif stripped_line.startswith('import') and not in_theorem:
                import_prefix = stripped_line
        
        # Handle case where last theorem doesn't end with sorry on separate line
        if in_theorem and current_theorem:
            theorem_text = '\n'.join(current_theorem)
            if 'sorry' in theorem_text:
                if not theorem_text.startswith('import'):
                    theorem_text = f"{import_prefix}\n\n{theorem_text}"
                subgoals.append(theorem_text.strip())
        
        # Remove duplicates
        unique_subgoals = []
        seen_theorems = set()
        for subgoal in subgoals:
            # Extract theorem name for deduplication
            if 'theorem subgoal_' in subgoal:
                theorem_name = subgoal.split('theorem ')[1].split(' ')[0] if 'theorem ' in subgoal else str(len(unique_subgoals))
                if theorem_name not in seen_theorems:
                    seen_theorems.add(theorem_name)
                    unique_subgoals.append(subgoal)
        
        return unique_subgoals
    
    def generate_proof_tactic(self, subgoal_statement: str, context_sketch: str, 
                            error_history: Optional[List[str]] = None, 
                            attempt_number: int = 1) -> str:
        """
        Generate proof tactics for a subgoal using the prove model.
        
        Args:
            subgoal_statement: The subgoal statement to prove
            context_sketch: The full sketch context
            error_history: Previous error messages for this subgoal
            attempt_number: Current attempt number
            
        Returns:
            Generated proof tactics
        """
        prompt = prompts.PROVE_SUBGOAL_PROMPT.format(
            full_context=context_sketch,
            subgoal_statement=subgoal_statement
        )
        
        # Base messages
        messages = [
            {"role": "system", "content": "You are an expert programmer and mathematician who helps formalizing mathematical problems in Lean 4."},
            {"role": "user", "content": prompt}
        ]
        
        # Add error feedback for subsequent attempts
        if attempt_number > 1 and error_history:
            last_error = error_history[-1]
            messages.append({
                "role": "user", 
                "content": last_error
            })
        
        try:
            response = self.prove_client.chat.completions.create(
                model=self.prove_model_name,
                messages=messages,
                temperature=0.1 if attempt_number > 1 else 0.3,  # Lower temperature for fixes
                max_tokens=16384,
                n=1,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating proof tactic: {e}"
    
    def synthesize_final_proof(self, original_statement: str, sketch: str, 
                             proven_subgoals_summary: str) -> str:
        """
        Synthesize the final proof using the draft model.
        
        Args:
            original_statement: The original theorem statement
            sketch: The original sketch
            proven_subgoals_summary: Summary of proven subgoals
            
        Returns:
            Complete Lean 4 proof for the original statement
        """
        prompt = prompts.FINALIZE_PROOF_PROMPT.format(
            original_statement=original_statement,
            original_sketch=sketch,
            proven_subgoals_summary=proven_subgoals_summary
        )
        
        messages = [
            {"role": "system", "content": "You are an expert programmer and mathematician who helps formalizing mathematical problems in Lean 4."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.draft_client.chat.completions.create(
                model=self.draft_model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=8192,
                n=1,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error synthesizing final proof: {e}"