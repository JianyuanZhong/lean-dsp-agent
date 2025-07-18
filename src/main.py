"""
Main entry point for the Lean DSP+ Agent.
Handles command-line argument parsing and initializes the orchestrator.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

from .orchestrator import DSPAgentOrchestrator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Please create a config.json file from the template:")
        print("   cp config.json.template config.json")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        sys.exit(1)


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save results to JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Lean DSP+ Agent - Automated theorem proving using Draft, Sketch, Prove framework"
    )
    
    # Required arguments
    parser.add_argument(
        "--theorem", 
        required=True,
        help="The theorem statement to prove (must include imports)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    # Output options
    parser.add_argument(
        "--output-file", 
        default="dsp_results.json",
        help="Output file for results (default: dsp_results.json)"
    )
    
    parser.add_argument(
        "--detailed-summary", 
        action="store_true",
        help="Print detailed summary of all proof attempts"
    )
    
    # Override configuration options
    parser.add_argument(
        "--max-attempts", 
        type=int,
        help="Maximum proof attempts per subgoal (overrides config)"
    )
    
    parser.add_argument(
        "--max-workers", 
        type=int,
        help="Maximum parallel workers (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.max_attempts is not None:
        config["agent_settings"]["max_proof_attempts"] = args.max_attempts
    
    if args.max_workers is not None:
        config["agent_settings"]["max_parallel_workers"] = args.max_workers
    
    # Print configuration summary
    print("=" * 80)
    print("🚀 LEAN DSP+ AGENT - CONFIGURATION")
    print("=" * 80)
    print(f"🔧 Lean Server: {config['lean_server_url']}")
    print(f"🤖 Draft Model: {config['llm_providers']['draft_model']['model_name']}")
    print(f"🤖 Sketch Model: {config['llm_providers']['sketch_model']['model_name']}")
    print(f"🤖 Prove Model: {config['llm_providers']['prove_model']['model_name']}")
    print(f"⚙️  Max Attempts: {config['agent_settings']['max_proof_attempts']}")
    print(f"⚙️  Max Workers: {config['agent_settings']['max_parallel_workers']}")
    print(f"📄 Output File: {args.output_file}")
    print("=" * 80)
    print()
    
    # Initialize orchestrator
    try:
        orchestrator = DSPAgentOrchestrator(config)
        print("✅ DSP+ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DSP+ Agent: {e}")
        sys.exit(1)
    
    # Run the pipeline
    try:
        results = orchestrator.process(args.theorem)
        
        # Print detailed summary if requested
        if args.detailed_summary:
            orchestrator.print_detailed_results_summary(results)
        
        # Save results
        save_results(results, args.output_file)
        
        # Exit with appropriate code
        sys.exit(0 if results.get("success", False) else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()