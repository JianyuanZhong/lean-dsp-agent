"""
Orchestrator module for the DSP+ Agent.
Implements the core five-stage pipeline for automated theorem proving.
"""

import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .llm_services import LLMServices
from .lean_verifier import LeanVerifier
from .utils import extract_proof_from_text, extract_sketch_from_response, replace_statement_in_proof


@dataclass
class AttemptInfo:
    """Represents information about a single proof attempt"""
    attempt_number: int
    full_response: str
    extracted_code: str
    test_code: str
    verification_result: Dict
    is_successful: bool
    error_message: Optional[str] = None


@dataclass
class SubgoalTask:
    """Represents a single subgoal to be proven"""
    subgoal_id: str
    subgoal_statement: str
    context_sketch: str
    attempts: int = 0
    is_solved: bool = False
    solution_tactics: Optional[str] = None
    error_history: List[str] = None
    attempt_details: List[AttemptInfo] = None
    
    def __post_init__(self):
        if self.error_history is None:
            self.error_history = []
        if self.attempt_details is None:
            self.attempt_details = []


class DSPAgentOrchestrator:
    """
    Central orchestrator for the Draft, Sketch, Prove (DSP+) framework.
    Implements a five-stage pipeline for automated theorem proving.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DSP Agent Orchestrator.
        
        Args:
            config: Configuration dictionary containing:
                - lean_server_url: URL for the kimina-lean-server
                - llm_providers: Configuration for LLM providers
                - agent_settings: Agent behavior settings
        """
        self.config = config
        
        # Initialize services
        self.llm_services = LLMServices(config["llm_providers"])
        self.lean_verifier = LeanVerifier(config["lean_server_url"])
        
        # Agent settings
        self.max_proof_attempts = config["agent_settings"]["max_proof_attempts"]
        self.max_parallel_workers = config["agent_settings"]["max_parallel_workers"]
        
        # Threading lock for thread-safe operations
        self._lock = threading.Lock()
    
    def _generate_draft(self, formal_statement: str) -> str:
        """
        Stage 1: Generate a natural-language proof plan.
        
        Args:
            formal_statement: The formal theorem statement
            
        Returns:
            Natural-language proof plan
        """
        print("🎯 STAGE 1: Generating natural-language proof plan...")
        
        draft = self.llm_services.generate_draft(formal_statement)
        
        if not draft.startswith("Error"):
            print(f"✅ Draft generated ({len(draft)} characters)")
        else:
            print(f"❌ Error in draft generation: {draft}")
        
        return draft
    
    def _generate_sketch(self, formal_statement: str, concise_draft: str) -> str:
        """
        Stage 2: Generate a Lean 4 sketch with error masking.
        
        Args:
            formal_statement: The formal theorem statement
            concise_draft: The natural-language proof plan
            
        Returns:
            Lean 4 sketch with 'by sorry' placeholders
        """
        print("🏗️  STAGE 2: Generating Lean 4 sketch...")
        
        # Generate initial sketch
        sketch_response = self.llm_services.generate_sketch(formal_statement, concise_draft)
        
        if sketch_response.startswith("Error"):
            print(f"❌ Error in sketch generation: {sketch_response}")
            return sketch_response
        
        # Extract sketch from response
        sketch = extract_proof_from_text(sketch_response)
        
        if sketch == "No proof found in the output.":
            print("⚠️  No proof found in sketch response, using raw response")
            sketch = sketch_response
        
        # Apply error masking using the LeanVerifier
        print("🔧 Applying error masking to sketch...")
        masked_sketch = self.lean_verifier.verify_with_error_masking(sketch)
        
        print(f"✅ Sketch generated and masked ({len(masked_sketch)} characters)")
        print(f"Final sketch preview: {masked_sketch[:200]}...")
        
        return masked_sketch
    
    def _generate_sketch_with_masking_style(self, formal_statement: str, concise_draft: str) -> str:
        """
        Alternative sketch generation using the masking-style prompt from sketch.py.
        
        Args:
            formal_statement: The formal theorem statement
            concise_draft: The natural-language proof plan
            
        Returns:
            Lean 4 sketch with error masking applied
        """
        print("🏗️  STAGE 2: Generating Lean 4 sketch (masking style)...")
        
        # Parse header and statement
        header = "import Mathlib\nimport Aesop\nset_option maxHeartbeats 0"
        if formal_statement.startswith("import"):
            lines = formal_statement.split('\n')
            header_lines = []
            statement_lines = []
            in_header = True
            
            for line in lines:
                if line.strip().startswith(('import', 'set_option', 'open')):
                    if in_header:
                        header_lines.append(line)
                    else:
                        statement_lines.append(line)
                else:
                    in_header = False
                    statement_lines.append(line)
            
            header = '\n'.join(header_lines) if header_lines else header
            statement = '\n'.join(statement_lines)
        else:
            statement = formal_statement
        
        # Generate raw sketch
        raw_sketch = self.llm_services.generate_sketch_with_masking(statement, header, concise_draft)
        
        if raw_sketch.startswith("Error"):
            print(f"❌ Error in sketch generation: {raw_sketch}")
            return raw_sketch
        
        # Extract sketch from response
        extracted_sketch = extract_sketch_from_response(raw_sketch)
        
        # Apply error masking
        print("🔧 Applying error masking to sketch...")
        masked_sketch = self.lean_verifier.verify_with_error_masking(extracted_sketch)
        
        print(f"✅ Sketch generated and masked ({len(masked_sketch)} characters)")
        
        return masked_sketch
    
    def _extract_subgoals(self, sketch: str, formal_statement: str) -> List[str]:
        """
        Stage 3: Extract subgoals from the sketch using LLM-based extraction.
        
        Args:
            sketch: The Lean 4 sketch with 'by sorry' placeholders
            formal_statement: The original formal theorem statement
            
        Returns:
            List of subgoal statements formatted as standalone theorems
        """
        print("🔍 STAGE 3: Extracting subgoals from sketch using LLM...")
        
        subgoals = self.llm_services.extract_subgoals(sketch, formal_statement)
        
        print(f"✅ Extracted {len(subgoals)} subgoals")
        for i, subgoal in enumerate(subgoals, 1):
            print(f"  {i}. {subgoal[:100]}{'...' if len(subgoal) > 100 else ''}")
        
        return subgoals
    
    def _prove_single_subgoal(self, task: SubgoalTask) -> SubgoalTask:
        """
        Prove a single subgoal using the "Attempt -> Verify -> Correct" loop.
        
        Args:
            task: SubgoalTask containing the subgoal to prove
            
        Returns:
            Updated SubgoalTask with solution or error information
        """
        print(f"🎯 Proving subgoal {task.subgoal_id}: {task.subgoal_statement[:80]}...")
        
        for attempt in range(self.max_proof_attempts):
            task.attempts = attempt + 1
            
            try:
                # Generate proof attempt
                full_response = self.llm_services.generate_proof_tactic(
                    task.subgoal_statement,
                    task.context_sketch,
                    task.error_history,
                    attempt + 1
                )
                
                task.solution_tactics = full_response
                
                # Debug: Print model response
                print(f"🔍 Model response for subgoal {task.subgoal_id}:")
                print(f"{'='*50}")
                print(full_response[:500] + "..." if len(full_response) > 500 else full_response)
                print(f"{'='*50}")
                
                # Extract Lean code from response
                lean_code = extract_proof_from_text(full_response)
                print(f"🔍 Extracted lean_code: '{lean_code[:200]}...' " if len(lean_code) > 200 else f"🔍 Extracted lean_code: '{lean_code}'")
                
                # Replace statement in proof to ensure consistency
                test_code = replace_statement_in_proof(lean_code, task.subgoal_statement)
                
                # Verify with Lean server
                print(f"test_code: {test_code}")
                verification_result = self.lean_verifier.verify(test_code)
                
                # Create detailed attempt info
                error_message = None
                if verification_result["has_error"]:
                    error_message = self.lean_verifier.get_error_message(test_code, verification_result)
                    task.error_history.append(error_message)
                
                # Store detailed attempt information
                attempt_info = AttemptInfo(
                    attempt_number=attempt + 1,
                    full_response=full_response,
                    extracted_code=lean_code,
                    test_code=test_code,
                    verification_result=verification_result,
                    is_successful=verification_result["is_valid_no_sorry"],
                    error_message=error_message
                )
                task.attempt_details.append(attempt_info)
                
                if verification_result["is_valid_no_sorry"]:
                    print(f"✅ Subgoal {task.subgoal_id} solved in {attempt + 1} attempt(s)")
                    task.is_solved = True
                    return task
                
                elif verification_result["has_error"]:
                    print(f"❌ Attempt {attempt + 1} failed for subgoal {task.subgoal_id}")
                    print(f"verification_result: {verification_result['lean_feedback']}")
                
            except Exception as e:
                error_msg = f"Exception in attempt {attempt + 1}: {str(e)}"
                task.error_history.append(error_msg)
                print(f"❌ {error_msg}")
                
                # Store attempt info even for exceptions
                attempt_info = AttemptInfo(
                    attempt_number=attempt + 1,
                    full_response="",
                    extracted_code="",
                    test_code="",
                    verification_result={"has_error": True, "error": str(e)},
                    is_successful=False,
                    error_message=error_msg
                )
                task.attempt_details.append(attempt_info)
        
        print(f"❌ Failed to solve subgoal {task.subgoal_id} after {self.max_proof_attempts} attempts")
        return task
    
    def _prove_subgoals_parallel(self, subgoals: List[str], sketch: str) -> List[SubgoalTask]:
        """
        Stage 4: Prove subgoals in parallel using multiple workers.
        
        Args:
            subgoals: List of subgoal statements to prove
            sketch: The full sketch context
            
        Returns:
            List of SubgoalTask objects with results
        """
        print(f"⚡ STAGE 4: Proving {len(subgoals)} subgoals in parallel...")
        
        # Create tasks
        tasks = []
        for i, subgoal in enumerate(subgoals):
            task = SubgoalTask(
                subgoal_id=f"subgoal_{i+1}",
                subgoal_statement=subgoal,
                context_sketch=sketch
            )
            tasks.append(task)
        
        # Execute in parallel
        completed_tasks = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(self._prove_single_subgoal, task): task for task in tasks}
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result_task = future.result()
                    completed_tasks.append(result_task)
                except Exception as e:
                    print(f"❌ Exception in subgoal {task.subgoal_id}: {e}")
                    task.error_history.append(f"Execution exception: {e}")
                    completed_tasks.append(task)
        
        # Sort by subgoal_id to maintain order
        completed_tasks.sort(key=lambda x: x.subgoal_id)
        
        solved_count = sum(1 for task in completed_tasks if task.is_solved)
        print(f"✅ Solved {solved_count}/{len(subgoals)} subgoals in parallel")
        
        return completed_tasks
    
    def _synthesize_final_proof(self, original_statement: str, sketch: str, proven_tasks: List[SubgoalTask]) -> str:
        """
        Stage 5: Synthesize the final proof using proven subgoals.
        
        Args:
            original_statement: The original theorem statement
            sketch: The original sketch
            proven_tasks: List of SubgoalTask objects with solutions
            
        Returns:
            Complete Lean 4 proof for the original statement
        """
        print("🔧 STAGE 5: Synthesizing final proof...")
        
        # Prepare summary of proven subgoals
        proven_subgoals_summary = self._format_proven_subgoals_summary(proven_tasks)
        
        # Generate final proof
        final_proof_response = self.llm_services.synthesize_final_proof(
            original_statement, sketch, proven_subgoals_summary
        )
        
        if final_proof_response.startswith("Error"):
            print(f"❌ Error in proof finalization: {final_proof_response}")
            # Fallback: use first successfully solved subgoal
            for task in proven_tasks:
                if task.is_solved and task.solution_tactics:
                    print(f"⚠️  Using fallback proof from subgoal {task.subgoal_id}")
                    return extract_proof_from_text(task.solution_tactics)
            
            # Last resort: return original sketch
            print("⚠️  No solved subgoals found, returning original sketch")
            return sketch
        
        # Extract the final proof
        final_proof = extract_proof_from_text(final_proof_response)
        print(f"✅ Final proof synthesized ({len(final_proof)} characters)")
        
        return final_proof
    
    def _format_proven_subgoals_summary(self, proven_tasks: List[SubgoalTask]) -> str:
        """
        Extract proven code snippets for the finalization prompt.
        
        Args:
            proven_tasks: List of SubgoalTask objects
            
        Returns:
            Clean Lean code snippets from successfully proven subgoals
        """
        code_snippets = []
        
        for task in proven_tasks:
            if task.is_solved and task.solution_tactics:
                # Get the first successful attempt
                successful_attempt = None
                for attempt in task.attempt_details:
                    if attempt.is_successful:
                        successful_attempt = attempt
                        break
                
                if successful_attempt and successful_attempt.extracted_code:
                    code_snippets.append(successful_attempt.extracted_code.strip())
        
        return "\n\n".join(code_snippets) if code_snippets else "No proven subgoals available."
    
    def process(self, formal_statement: str) -> Dict[str, Any]:
        """
        Execute the complete 5-stage DSP+ pipeline.
        
        Args:
            formal_statement: The formal theorem statement to prove
            
        Returns:
            Dictionary containing all results and intermediate steps
        """
        print("=" * 80)
        print("🚀 DSP+ AGENT ORCHESTRATOR - STARTING PIPELINE")
        print("=" * 80)
        print(f"📝 Theorem: {formal_statement[:100]}{'...' if len(formal_statement) > 100 else ''}")
        print()
        
        start_time = time.time()
        results = {
            "formal_statement": formal_statement,
            "stages": {},
            "final_proof": None,
            "is_valid": False,
            "execution_time": 0,
            "success": False
        }
        
        try:
            # Stage 1: Draft
            draft = self._generate_draft(formal_statement)
            results["stages"]["draft"] = draft
            print()
            
            # Stage 2: Sketch  
            sketch = self._generate_sketch(formal_statement, draft)
            results["stages"]["sketch"] = sketch
            print()
            
            # Stage 3: Decompose
            subgoals = self._extract_subgoals(sketch, formal_statement)
            results["stages"]["subgoals"] = subgoals
            print()
            
            # Stage 4: Prove in Parallel
            proven_tasks = self._prove_subgoals_parallel(subgoals, sketch)
            results["stages"]["proven_tasks"] = [
                {
                    "subgoal_id": task.subgoal_id,
                    "subgoal_statement": task.subgoal_statement,
                    "is_solved": task.is_solved,
                    "attempts": task.attempts,
                    "solution_tactics": task.solution_tactics,
                    "error_count": len(task.error_history),
                    "error_history": task.error_history,
                    "attempt_details": [
                        {
                            "attempt_number": attempt.attempt_number,
                            "full_response": attempt.full_response,
                            "extracted_code": attempt.extracted_code,
                            "test_code": attempt.test_code,
                            "verification_result": attempt.verification_result,
                            "is_successful": attempt.is_successful,
                            "error_message": attempt.error_message
                        }
                        for attempt in task.attempt_details
                    ]
                }
                for task in proven_tasks
            ]
            print()
            
            # Stage 5: Synthesize Final Proof
            final_proof = self._synthesize_final_proof(formal_statement, sketch, proven_tasks)
            results["final_proof"] = final_proof
            results["stages"]["finalization"] = {
                "original_statement": formal_statement,
                "proven_subgoals_count": sum(1 for task in proven_tasks if task.is_solved),
                "total_subgoals_count": len(proven_tasks),
                "final_proof_length": len(final_proof)
            }
            print()
            
            # Final verification
            print("🔍 FINAL VERIFICATION: Checking synthesized proof...")
            verification = self.lean_verifier.verify(final_proof)
            results["verification"] = verification
            results["is_valid"] = verification["is_valid_no_sorry"]
            
            if results["is_valid"]:
                print("🎉 SUCCESS: Final proof is valid!")
                results["success"] = True
                print(f"🎉 Final proof:\n\n{final_proof}")
            else:
                print("❌ FAILURE: Final proof has errors")
                if verification.get("lean_feedback"):
                    error_info = self.lean_verifier.get_error_message(final_proof, verification)
                    print("🐛 Error details:")
                    print(error_info[:500] + "..." if len(error_info) > 500 else error_info)
            
        except Exception as e:
            print(f"❌ PIPELINE FAILURE: {e}")
            results["error"] = str(e)
        
        finally:
            results["execution_time"] = time.time() - start_time
            print()
            print("=" * 80)
            print(f"⏱️  Total execution time: {results['execution_time']:.2f}s")
            print(f"🎯 Final result: {'SUCCESS' if results['success'] else 'FAILURE'}")
            print("=" * 80)
        
        return results
    
    def print_detailed_results_summary(self, results: Dict[str, Any]):
        """
        Print a detailed summary of the results including all attempts.
        
        Args:
            results: The results dictionary from the process method
        """
        print("\n" + "="*80)
        print("📊 DETAILED RESULTS SUMMARY")
        print("="*80)
        
        if "stages" in results and "proven_tasks" in results["stages"]:
            for task_data in results["stages"]["proven_tasks"]:
                print(f"\n🎯 Subgoal: {task_data['subgoal_id']}")
                print(f"   Statement: {task_data['subgoal_statement'][:80]}...")
                print(f"   Status: {'✅ SOLVED' if task_data['is_solved'] else '❌ FAILED'}")
                print(f"   Total Attempts: {task_data['attempts']}")
                
                if task_data['attempt_details']:
                    print(f"   📝 Attempt Details:")
                    for attempt in task_data['attempt_details']:
                        status = "✅" if attempt['is_successful'] else "❌"
                        print(f"      {status} Attempt {attempt['attempt_number']}:")
                        print(f"         Response Length: {len(attempt['full_response'])} chars")
                        print(f"         Extracted Code Length: {len(attempt['extracted_code'])} chars")
                        print(f"         Test Code Length: {len(attempt['test_code'])} chars")
                        if attempt['error_message']:
                            error_preview = attempt['error_message'][:100] + "..." if len(attempt['error_message']) > 100 else attempt['error_message']
                            print(f"         Error: {error_preview}")
                print("-" * 60)
        
        print(f"\n📊 Overall Success Rate: {results.get('success', False)}")
        print(f"⏱️  Total Time: {results.get('execution_time', 0):.2f}s")
        print("="*80)