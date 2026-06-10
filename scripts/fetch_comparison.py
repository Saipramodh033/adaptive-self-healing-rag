import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

def main():
    client = Client()
    
    # Get all projects
    projects = list(client.list_projects())
    
    # Find latest traditional and adaptive runs
    trad_projects = [p for p in projects if p.name.startswith("Traditional-RAG-Baseline")]
    adapt_projects = [p for p in projects if p.name.startswith("Adaptive-RAG-System")]
    
    if not trad_projects or not adapt_projects:
        print("Could not find projects.")
        return
        
    # Sort by start_time (assuming they have one, or just take the last)
    # The LangSmith API projects are often returned in chronological order
    trad_project = sorted(trad_projects, key=lambda p: p.start_time, reverse=True)[0]
    adapt_project = sorted(adapt_projects, key=lambda p: p.start_time, reverse=True)[0]
    
    print(f"Using Traditional: {trad_project.name}")
    print(f"Using Adaptive: {adapt_project.name}")
    
    # Fetch runs
    trad_runs = list(client.list_runs(project_name=trad_project.name))
    adapt_runs = list(client.list_runs(project_name=adapt_project.name))
    
    # Group by example_id
    comparisons = defaultdict(dict)
    
    for r in trad_runs:
        if r.reference_example_id:
            comparisons[r.reference_example_id]["traditional"] = r
            
    for r in adapt_runs:
        if r.reference_example_id:
            comparisons[r.reference_example_id]["adaptive"] = r
            
    # Also fetch the examples so we get the question string and expected output
    dataset_name = "CUSTOMER-SUPPORT-2"
    examples = {e.id: e for e in client.list_examples(dataset_name=dataset_name)}
    
    md_lines = []
    md_lines.append("# RAG Systems Output Comparison")
    md_lines.append("This document contains a side-by-side output comparison of the Traditional vs. Adaptive RAG for each question in the benchmark.")
    md_lines.append("")
    
    for ex_id, runs in comparisons.items():
        ex = examples.get(ex_id)
        if not ex: continue
            
        question = ex.inputs.get("question", "Unknown Question")
        expected_route = ex.outputs.get("expected_route", "N/A")
        
        trad_run = runs.get("traditional")
        adapt_run = runs.get("adaptive")
        
        # safely extract answers
        trad_ans = trad_run.outputs.get("answer", "No answer") if trad_run and trad_run.outputs else "Error/No Run"
        adapt_ans = adapt_run.outputs.get("answer", "No answer") if adapt_run and adapt_run.outputs else "Error/No Run"
        
        trad_esc = trad_run.outputs.get("is_escalated", False) if trad_run and trad_run.outputs else False
        adapt_esc = adapt_run.outputs.get("is_escalated", False) if adapt_run and adapt_run.outputs else False
        
        md_lines.append(f"## Question: {question}")
        md_lines.append(f"**Expected Route:** `{expected_route}`\n")
        
        md_lines.append("### Traditional RAG")
        md_lines.append(f"**Escalated?** {trad_esc}")
        md_lines.append(f"> {trad_ans}\n")
        
        md_lines.append("### Adaptive RAG")
        md_lines.append(f"**Escalated?** {adapt_esc}")
        md_lines.append(f"> {adapt_ans}\n")
        
        md_lines.append("---\n")
        
    with open(r"C:\Users\appis\.gemini\antigravity\brain\e3530693-741a-420b-8649-1419deca1075\system_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print("Done! Overrode system_comparison.md")

if __name__ == "__main__":
    main()
