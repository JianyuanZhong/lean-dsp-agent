"""
Prompt templates for the DSP+ Agent.
Contains all prompts used in the five-stage pipeline.
"""

DRAFT_PROMPT = """You are a brilliant mathematician. Your task is to create a concise, high-level proof plan for the given formal statement.
Do not write the full proof. Focus only on the key steps and formulas needed to guide the proof.
Filter out any "thinking" or self-reflection; provide only the essential steps.

**Formal Statement:**
{formal_statement}

**Proof Plan:**"""

SKETCH_PROMPT = """You are a programmer specializing in the Lean 4 proof assistant.
Your task is to translate a natural-language proof plan into a formal Lean 4 sketch.
Use the original formal statement as the theorem to prove.
Convert each step from the proof plan into a 'have' or 'let' statement.
Do not prove any subgoals. Instead, use 'by sorry' as a placeholder for the proof of each step.
This creates a scaffold for the final proof.

**Original Formal Statement:**
{formal_statement}

**Natural-Language Proof Plan:**
{concise_draft}
"""

PROVE_SUBGOAL_PROMPT = """Think about and solve the following problems step by step in Lean 4.

import Mathlib

open scoped Real
open scoped Topology
{subgoal_statement}"""

FINALIZE_PROOF_PROMPT = """You are an expert in Lean 4. Your task is to create a complete, valid proof for the original theorem statement using the context and proven subgoals provided.

**Original Theorem Statement:**
{original_statement}

**Original Sketch:**
{original_sketch}

**Proven Subgoals:**
{proven_subgoals_summary}

**Instructions:**
1. Create a complete proof for the ORIGINAL theorem statement (not the subgoals)
2. You can reference or incorporate the proven subgoals as needed
3. Ensure the proof is syntactically correct and complete
4. Use proper Lean 4 syntax and formatting
5. Start with necessary imports

Think about and solve this step by step in Lean 4."""

SUBGOAL_EXTRACTION_PROMPT = """You are an expert in Lean 4. Analyze the following Lean 4 proof sketch and extract ALL subgoals that need to be proven (marked with 'sorry').

For each subgoal, you need to create a standalone theorem that includes:
1. All necessary variable declarations from the original theorem
2. All relevant hypotheses/premises that the subgoal depends on
3. The subgoal statement itself

**Original Formal Statement:**
```lean
{formal_statement}
```

**Lean 4 Sketch:**
```lean
{sketch}
```

**Instructions:**
1. Find all lines containing 'sorry' that represent subgoals to be proven
2. For each subgoal, determine which variables and hypotheses from the original theorem are needed
3. Format each subgoal as a complete standalone theorem with the structure:
   ```
   import Mathlib
   
   theorem subgoal_N (variables) (relevant_hypotheses) : STATEMENT := by sorry
   ```
4. Include only the premises that are actually needed for each specific subgoal
5. Use theorem names like subgoal_1, subgoal_2, etc.
6. Return one complete theorem per subgoal
7. Do NOT include any other text or explanations

**Extracted Subgoals as Standalone Theorems:**"""

SKETCH_PROMPT_WITH_MASKING = """informal_proof:
{draft}

Prove the theorem in Lean 4 code. You should translate steps in the informal proof in a series of 'have'/'let'/'induction'/'match'/'suffices' statements, but you do not need to prove them. You only need to use placeholder `by{{new_line}}prove_with[h1, step5, ...{{hypothesises used here which are proposed ahead}}]`. We want to have as many lemmas as possible, and every lemma must be easy to proof.

When using a / b, you must specify **a's or b's type**, because (1:ℝ) / 2 is 0.5, but (1:ℤ) / 2 is 0.
When using a - b, you must specify **a's or b's type**, because (1:ℤ) - 2 is -1, but (1:ℕ) - 2 is 0.
n! is incorrect, you should use (n)!.

Here is an example:
```lean4
import Mathlib

example (x y : ℝ) (h1 : x ≤ 1 / 2) (h2 : x > 0) (t: y < Real.sin (x)): y < 1 / 2 := by
  -- Step 1
  have h3 : y < (1:ℝ) / 2 := by
    -- Step 2
    have h4 : Real.sin x ≤ x := by
      prove_with[h2]
    -- Step 3
    have h5 : y < x := by
      prove_with[h4, t]
    prove_with[h1, h5]
  exact h3
```

formal_statement:
```lean4
{header}
{formal_statement}
```"""