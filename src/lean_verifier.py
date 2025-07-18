"""
Lean Verifier module for the DSP+ Agent.
Handles all communication with the kimina-lean-server.
"""

import json
from typing import Dict, Any, List, Tuple
from .utils import parse_client_response, create_tool_message


class LeanVerifier:
    """
    Handles communication with the kimina-lean-server for Lean code verification.
    """
    
    def __init__(self, lean_server_url: str = "http://127.0.0.1:12332"):
        """
        Initialize the LeanVerifier.
        
        Args:
            lean_server_url: URL for the kimina-lean-server
        """
        self.lean_server_url = lean_server_url
        self._client = None
        
    def _get_client(self):
        """Lazy initialization of the Lean client."""
        if self._client is None:
            # Import here to avoid circular imports and allow optional dependency
            try:
                from client import Lean4Client
                self._client = Lean4Client(base_url=self.lean_server_url)
            except ImportError:
                raise ImportError(
                    "client module not found. Please ensure the Lean client is available."
                )
        return self._client
    
    def verify(self, lean_code: str) -> Dict[str, Any]:
        """
        Verify the given Lean code using the kimina-lean-server.
        
        Args:
            lean_code: The Lean 4 code to verify
            
        Returns:
            Dict containing verification results with the following structure:
            {
                "proof": str,                    # The original code
                "lean_feedback": str,            # Raw JSON response from server
                "has_error": bool,               # Whether there are errors
                "is_valid_no_sorry": bool,       # Valid without sorry statements
                "is_valid_with_sorry": bool,     # Syntactically valid (may have sorry)
                "success": bool,                 # Whether verification succeeded
                "error": str (optional)          # Error message if verification failed
            }
        """
        try:
            client = self._get_client()
            
            # Prepare the verification request
            code_request = {
                "custom_id": "verification",
                "proof": lean_code,
            }
            
            # Submit verification request
            response = client.verify([code_request], timeout=60, infotree_type="original")
            result = response["results"][0]
            
            # Parse the response
            parsed_result = parse_client_response(result)
            
            return {
                "proof": lean_code,
                "lean_feedback": json.dumps(result),
                "has_error": parsed_result["has_error"],
                "is_valid_no_sorry": parsed_result["is_valid_no_sorry"],
                "is_valid_with_sorry": parsed_result["is_valid_with_sorry"],
                "success": True
            }
            
        except Exception as e:
            return {
                "proof": lean_code,
                "lean_feedback": str(e),
                "has_error": True,
                "is_valid_no_sorry": False,
                "is_valid_with_sorry": False,
                "success": False,
                "error": str(e)
            }
    
    def verify_with_error_masking(self, lean_code: str, max_iterations: int = 10) -> str:
        """
        Verify code and apply error masking to create a syntactically valid sketch.
        This implements the error masking logic from sketch.py.
        
        Args:
            lean_code: The Lean code to verify and mask
            max_iterations: Maximum number of masking iterations
            
        Returns:
            Syntactically valid Lean code with errors masked as 'sorry'
        """
        from .utils import parse_sketch_to_tree, mask_errors_in_tree
        
        # Split the initial sketch into lines
        formal_proof_lines = lean_code.split('\n')
        last_error_lines = []
        iteration_count = 0
        
        while iteration_count < max_iterations:
            # Verify current code
            verification_result = self.verify('\n'.join(formal_proof_lines))
            
            # Extract error lines
            error_lines = self._extract_error_lines(verification_result)
            
            if not error_lines:
                break
            
            # Parse code into tree structure
            tree = parse_sketch_to_tree(formal_proof_lines)
            
            # Use strict mode if same errors repeat
            strict_mode = (error_lines == last_error_lines and iteration_count > 1)
            
            # Apply error masking
            masked_lines = mask_errors_in_tree(
                tree, error_lines, strict=strict_mode
            )
            
            formal_proof_lines = masked_lines
            last_error_lines = error_lines
            iteration_count += 1
        
        # Final verification to check for unsolved goals
        final_verification = self.verify('\n'.join(formal_proof_lines))
        final_error_lines = self._extract_error_lines(final_verification)
        
        # Append 'sorry' if unsolved goals remain
        if any('unsolved goals' in error.get('data', '') 
               for error in json.loads(final_verification.get('lean_feedback', '{}')).get('errors', [])):
            formal_proof_lines.append('  sorry')
        
        return '\n'.join(formal_proof_lines)
    
    def _extract_error_lines(self, verification_result: Dict[str, Any]) -> List[int]:
        """
        Extract line numbers with errors from verification result.
        
        Args:
            verification_result: Result from verify() method
            
        Returns:
            List of line numbers with errors
        """
        error_lines = []
        
        if not verification_result.get("success", False):
            return error_lines
        
        try:
            lean_feedback = json.loads(verification_result.get("lean_feedback", "{}"))
            
            for error in lean_feedback.get('errors', []):
                if error.get('data', '').startswith('unsolved goals'):
                    continue  # Skip unsolved goals errors for now
                
                if 'pos' in error and 'line' in error['pos']:
                    error_lines.append(error['pos']['line'])
                    
        except (json.JSONDecodeError, KeyError, TypeError):
            # If parsing fails, return empty list
            pass
        
        return error_lines
    
    def get_error_message(self, lean_code: str, verification_result: Dict[str, Any]) -> str:
        """
        Get a formatted error message from verification result.
        
        Args:
            lean_code: The Lean code that was verified
            verification_result: Result from verify() method
            
        Returns:
            Formatted error message
        """
        if not verification_result.get("has_error", False):
            return "No errors found."
        
        try:
            lean_feedback = json.loads(verification_result.get("lean_feedback", "{}"))
            return create_tool_message(lean_code, lean_feedback)
        except (json.JSONDecodeError, KeyError, TypeError):
            return f"Error parsing Lean feedback: {verification_result.get('lean_feedback', 'Unknown error')}"