"""
Utility functions for the DSP+ Agent.
Contains helper functions for parsing, processing, and extracting information.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple


def extract_proof_from_text(response: str) -> str:
    """
    Extract Lean code from model response text.
    
    Args:
        response: The model's response text
        
    Returns:
        Extracted Lean code or "No proof found in the output." if no code found
    """
    # Try different patterns for code blocks
    patterns = [
        r"```lean4\n(.*?)\n```",
        r"```lean\n(.*?)\n```", 
        r"```\n(.*?)\n```",
        r"```lean4(.*?)```",
        r"```lean(.*?)```",
        r"```(.*?)```"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return the last match (most likely to be the final solution)
            code = matches[-1].strip()
            if code and len(code) > 10:  # Basic sanity check
                return code
    
    # If no code blocks found, try to extract everything after "import Mathlib"
    if "import Mathlib" in response:
        start_idx = response.find("import Mathlib")
        code = response[start_idx:].strip()
        # Clean up any trailing text after the proof
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            clean_lines.append(line)
            # Stop at common endings
            if line.strip().startswith('#eval') or line.strip().startswith('--'):
                break
        return '\n'.join(clean_lines)
    
    return "No proof found in the output."


def parse_sketch_to_tree(code_lines: List[str]) -> List[Dict[str, Any]]:
    """
    Parse Lean code into a tree structure based on indentation.
    Migrated from sketch.py for error masking functionality.
    
    Args:
        code_lines: List of code lines
        
    Returns:
        Tree structure representing the code hierarchy
    """
    stack = []
    tree = []

    for i, line in enumerate(code_lines):
        clean_line = line.strip()
        if not clean_line:
            continue
        
        indent_level = len(line) - len(line.lstrip())
        node = {
            "content": clean_line, 
            "children": [], 
            "error": False,  # Will be set by error detection
            "line_number": i + 1
        }

        if not stack:
            tree.append(node)
            stack.append((indent_level, node))
        else:
            while stack and indent_level <= stack[-1][0]:
                stack.pop()

            if stack:
                stack[-1][1]["children"].append(node)
            else:
                tree.append(node)
            stack.append((indent_level, node))

    return tree


def mask_errors_in_tree(tree: List[Dict[str, Any]], error_lines: List[int], 
                       level: int = 0, comment: bool = False, 
                       if_sorry: bool = False, strict: bool = False) -> List[str]:
    """
    Recursively rebuild sketch from tree, replacing errors with sorry.
    Migrated from sketch.py for error masking functionality.
    
    Args:
        tree: The tree structure from parse_sketch_to_tree
        error_lines: List of line numbers with errors
        level: Current indentation level
        comment: Whether to comment out this level
        if_sorry: Whether this subtree has been replaced with sorry
        strict: If True, errors in children cause parent to be replaced with sorry
        
    Returns:
        List of output lines with errors masked
    """
    output_lines = []
    
    for node in tree:
        node_has_error = node["line_number"] in error_lines
        
        if level > 1:
            # Deeper level: directly suppress error nodes with 'sorry'
            if node_has_error:
                if not if_sorry:
                    output_lines.append("  " * level + '-- ' * comment + "sorry")
                    if_sorry = True
            else:
                if not if_sorry:
                    # Check if any child has error to decide if we still print content
                    child_has_error = any(
                        child["line_number"] in error_lines for child in node["children"]
                    )
                    if strict and child_has_error:
                        output_lines.append("  " * level + '-- ' * comment + "sorry")
                        if_sorry = True
                    else:
                        output_lines.append("  " * level + '-- ' * comment + node["content"])
            
            if node["children"]:
                child_lines = mask_errors_in_tree(
                    node["children"], error_lines, level + 1, comment, if_sorry, strict
                )
                output_lines.extend(child_lines)
                
        else:
            # Top-level: decide whether to comment or replace with 'sorry'
            if_comment = comment or node_has_error
            if not if_sorry:
                child_has_error = any(
                    child["line_number"] in error_lines for child in node["children"]
                )
                if strict and child_has_error:
                    output_lines.append("  " * level + '-- ' * if_comment + "sorry")
                    if_sorry = True
                else:
                    output_lines.append("  " * level + '-- ' * if_comment + node["content"])
            
            if node["children"]:
                child_lines = mask_errors_in_tree(
                    node["children"], error_lines, level + 1, if_comment, if_sorry, strict
                )
                output_lines.extend(child_lines)
    
    return output_lines


def create_tool_message(formal_code: str, lean_feedback: Dict[str, Any]) -> str:
    """
    Create a formatted error message from Lean feedback.
    
    Args:
        formal_code: The Lean code that was verified
        lean_feedback: The feedback from Lean server
        
    Returns:
        Formatted error message
    """
    if not lean_feedback.get("errors"):
        return "No errors found in Lean feedback."
    
    error_messages = []
    for error in lean_feedback["errors"]:
        if "pos" in error and "data" in error:
            line_num = error["pos"].get("line", "unknown")
            error_data = error["data"]
            error_messages.append(f"Line {line_num}: {error_data}")
        else:
            error_messages.append(f"Error: {error}")
    
    return f"Lean verification errors:\n" + "\n".join(error_messages)


def parse_client_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse client response to extract verification status.
    
    Args:
        response: Response from Lean client
        
    Returns:
        Parsed response with status flags
    """
    has_error = bool(response.get("errors"))
    
    # Check if proof is valid without sorry
    is_valid_no_sorry = not has_error and "sorry" not in response.get("proof", "")
    
    # Check if proof is valid with sorry (syntactically correct)
    is_valid_with_sorry = not has_error
    
    return {
        "has_error": has_error,
        "is_valid_no_sorry": is_valid_no_sorry,
        "is_valid_with_sorry": is_valid_with_sorry,
        "errors": response.get("errors", []),
        "proof": response.get("proof", "")
    }


def extract_sketch_from_response(raw_sketch: str) -> str:
    """
    Extract sketch from LLM output, handling various formats.
    Migrated from sketch.py.
    
    Args:
        raw_sketch: Raw LLM response
        
    Returns:
        Extracted and cleaned sketch
    """
    # Remove thinking sections
    sketch = raw_sketch.split('</think>')[-1].strip()
    
    # Extract from code blocks
    code_match = re.search(r'```lean4?\n(.+?)\n```', sketch, re.DOTALL)
    if code_match:
        sketch = code_match.group(1)
    
    # Remove import statements (will be added back by the system)
    sketch = re.sub(r'import .+?\n', '', sketch)
    
    return sketch.strip()


def replace_statement_in_proof(lean_code: str, original_subgoal: str) -> str:
    """
    Replace the statement in extracted Lean code with the original subgoal statement.
    
    Args:
        lean_code: The Lean code extracted from model response
        original_subgoal: The original subgoal statement
        
    Returns:
        Modified Lean code with original statement
    """
    try:
        # Extract the statement from the original subgoal
        if " : " in original_subgoal and " := by" in original_subgoal:
            start_idx = original_subgoal.find(" : ") + 3
            end_idx = original_subgoal.find(" := by")
            original_statement = original_subgoal[start_idx:end_idx].strip()
        else:
            return lean_code
        
        # Replace statement in the model's Lean code
        lines = lean_code.split('\n')
        modified_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            # Check if this line contains a statement to replace
            if ((' : ' in stripped_line and ' := by' in stripped_line) and 
                (stripped_line.startswith('have ') or stripped_line.startswith('theorem ') or 
                 stripped_line.startswith('lemma '))):
                
                # Extract the prefix and suffix
                colon_idx = stripped_line.find(' : ')
                prefix = stripped_line[:colon_idx + 3]
                
                assign_idx = stripped_line.find(' := by')
                suffix = stripped_line[assign_idx:]
                
                # Replace with original statement
                new_line = line.replace(stripped_line, f"{prefix}{original_statement}{suffix}")
                modified_lines.append(new_line)
            else:
                modified_lines.append(line)
        
        return '\n'.join(modified_lines)
        
    except Exception as e:
        # If replacement fails, return original code
        return lean_code