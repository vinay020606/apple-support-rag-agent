"""
main.py - Master Pipeline Execution Script
AI Customer Support Agent & Evaluation Pipeline (@AppleSupport)

Executes the complete pipeline end-to-end:
1. Data Pipeline: Process raw Kaggle/synthetic Twitter support dataset & Golden Evaluation Set
2. Vector Store: Index brand resolutions into ChromaDB
3. Agent & Eval Harness: Benchmark baselines (Trivial vs Simple vs RAG Agent), LLM-as-a-Judge, and alignment proof
4. Report Generator: Generate README.md and REPORT.md
"""

import sys
import time
import json
import os

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from data_pipeline import load_or_process_data, generate_golden_evaluation_set
from vector_store import ResolutionVectorStore
from eval_harness import EvaluationHarness
from report_gen import generate_markdown_reports

def run_main_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("[+] Starting AI Customer Support Agent & Evaluation Pipeline (@AppleSupport)")
    print("=" * 70)

    # Step 1: Data Ingestion & Thread Extraction
    print("\n--- STEP 1: Data Ingestion & Thread Extraction ---")
    df_threads = load_or_process_data()
    golden_set = generate_golden_evaluation_set()

    # Step 2: ChromaDB Vector Store Indexing
    print("\n--- STEP 2: Building ChromaDB Vector Index ---")
    vs = ResolutionVectorStore()
    indexed_count = vs.build_index()

    # Step 3: Baseline Evaluation & LLM-as-a-Judge
    print("\n--- STEP 3: Running Evaluation Harness & LLM-as-a-Judge ---")
    evaluator = EvaluationHarness()
    eval_results = evaluator.run_full_evaluation()

    # Step 4: Markdown Report & README Generation
    print("\n--- STEP 4: Generating Markdown Reports & Technical Decision Log ---")
    generate_markdown_reports()

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] PIPELINE COMPLETED IN {minutes}m {seconds}s! (Under 15 min requirement)")
    print("=" * 70)

    # Print Terminal Benchmark Table
    print("\n[BENCHMARK SUMMARY TABLE]")
    print("-" * 75)
    print(f"{'Baseline Mode':<20} | {'Intent F1':<10} | {'Escalation F1':<13} | {'LLM Judge Score':<15}")
    print("-" * 75)
    
    b = eval_results["baselines"]
    for mode in ["TRIVIAL", "SIMPLE", "RAG_AGENT"]:
        intent_f1 = b[mode]["intent_metrics"]["f1_score"]
        esc_f1 = b[mode]["escalation_metrics"]["f1_score"]
        judge_score = b[mode]["llm_judge"]["mean_overall_score"]
        print(f"{mode:<20} | {intent_f1:<10.4f} | {esc_f1:<13.4f} | {judge_score:<15.2f} / 5.0")
    print("-" * 75)

    align = eval_results.get("judge_alignment", {})
    print(f"\n[Proof of Human-vs-LLM Judge Alignment]")
    print(f"   * Pearson Correlation Coefficient (r): {align.get('pearson_r', 'N/A')}")
    print(f"   * Cohen's Kappa (kappa):              {align.get('cohens_kappa', 'N/A')}")
    print(f"   * Alignment Status:                   {align.get('alignment_verdict', 'N/A')}")
    print("\nOutput Artifacts Generated:")
    print("   * data/processed/apple_support_threads.csv")
    print("   * data/golden_eval_set.json")
    print("   * data/eval_results.json")
    print("   * README.md")
    print("   * REPORT.md")
    print("=" * 70)


if __name__ == "__main__":
    run_main_pipeline()
