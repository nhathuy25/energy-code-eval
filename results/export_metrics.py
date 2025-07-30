# Huy: File created with vibe-coding using Claude Sonnet 4
import json
import csv
import os
import glob
import re
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

def extract_model_name(filename):
    """Extract model name and evaluator name from filename"""
    # Remove the file extension
    base_name = Path(filename).stem
    
    # Define all possible benchmark names
    benchmark_names = [
        "codesearchnet-python",
        "codesearchnet-java", 
        "codesearchnet-javascript",
        "humaneval",
        "humanevalplus",
        "mbpp",
        "mbppplus",
        "humanevalexplainsynthesize-python",
        "humanevalexplainsynthesize-java",
        "humanevalexplainsynthesize-javascript"
    ]
    
    # Special case for HQQ models with humanevalexplainsynthesize
    if "_HQQ_" in base_name and any(benchmark in base_name for benchmark in ["humanevalexplainsynthesize-python", "humanevalexplainsynthesize-java", "humanevalexplainsynthesize-javascript"]):
        # The format is typically model_name_4bits64gs_HQQ_evaluator_model_benchmark
        parts = base_name.split('_')
        hqq_idx = parts.index("HQQ") if "HQQ" in parts else -1
        
        if hqq_idx > 0:
            # Extract the model name including the HQQ part
            describe_model = "_".join(parts[:hqq_idx+1])
            
            # Find evaluator model and benchmark
            evaluator_model = None
            benchmark = None
            
            # Look through the remaining parts for benchmark
            for i, part in enumerate(parts[hqq_idx+1:], hqq_idx+1):
                if any(benchmark_name in part for benchmark_name in benchmark_names):
                    evaluator_model = "_".join(parts[hqq_idx+1:i])
                    benchmark = part
                    break
            
            if evaluator_model and benchmark:
                return describe_model, evaluator_model
    
    # Check for standard humanevalexplainsynthesize pattern with two model names
    elif any(benchmark in base_name for benchmark in ["humanevalexplainsynthesize-python", "humanevalexplainsynthesize-java", "humanevalexplainsynthesize-javascript"]):
        # Split by underscore to get potential model names and benchmark
        parts = base_name.split('_')
        
        # Check if we have at least 3 parts (describe_model, evaluator_model, benchmark)
        if len(parts) >= 3:
            # The first part is the describe model
            describe_model = parts[0]
            
            # Check for HQQ pattern in model name
            if "4bits64gs_HQQ" in base_name:
                # Find the HQQ part
                hqq_parts = [i for i, part in enumerate(parts) if "HQQ" in part]
                if hqq_parts:
                    hqq_idx = hqq_parts[0]
                    describe_model = "_".join(parts[:hqq_idx+1])
                    remaining_parts = parts[hqq_idx+1:]
                else:
                    remaining_parts = parts[1:]
            else:
                remaining_parts = parts[1:]
                
            # Find evaluator model and benchmark
            evaluator_model = None
            benchmark = None
            
            for i, part in enumerate(remaining_parts):
                if any(benchmark_name in part for benchmark_name in benchmark_names):
                    evaluator_model = "_".join(remaining_parts[:i])
                    benchmark = part
                    break
            
            if evaluator_model and benchmark:
                return describe_model, evaluator_model
    
    # Regular pattern for single model name
    model_name = base_name
    
    # Check for HQQ pattern in regular model name
    if "4bits64gs_HQQ" in base_name:
        # Keep the HQQ part as part of the model name
        pass  # No special handling needed - we'll extract using the regex pattern
    
    # Create a pattern that matches underscore followed by any combination of benchmarks
    benchmark_pattern = '|'.join(re.escape(name) for name in benchmark_names)
    pattern = rf'_(?=(?:{benchmark_pattern})(?:,(?:{benchmark_pattern}))*$)'
    
    match = re.search(pattern, base_name)
    if match:
        # Extract everything before the matched underscore
        model_name = base_name[:match.start()]
    else:
        # Fallback: look for any underscore followed by comma-separated benchmark names
        for i in range(len(base_name) - 1, -1, -1):
            if base_name[i] == '_':
                # Check if everything after this underscore consists only of benchmark names and commas
                suffix = base_name[i+1:]
                suffix_parts = [part.strip() for part in suffix.split(',')]
                if all(part in benchmark_names for part in suffix_parts if part):
                    model_name = base_name[:i]
                    break
    
    return model_name, None  # Return None as evaluator for regular files

def extract_benchmark_results(json_data):
    """Extract benchmark results from JSON data"""
    results = {}
    
    # Define the benchmarks we're interested in
    benchmarks = ["codesearchnet-python", "codesearchnet-java", "codesearchnet-javascript", "humaneval", "mbpp", "humanevalplus", "mbppplus",
                  "humanevalexplainsynthesize-python", "humanevalexplainsynthesize-java", "humanevalexplainsynthesize-javascript",]
    
    for benchmark in benchmarks:
        if benchmark in json_data:
            benchmark_data = json_data[benchmark]
            
            # For codesearchnet benchmarks, we'll use cosine similarity as the main metric
            if benchmark.startswith("codesearchnet"):
                results[f"{benchmark}_cosine"] = benchmark_data.get("cosine", None)
                results[f"{benchmark}_euclidean"] = benchmark_data.get("euclidean", None)
                results[f"{benchmark}_BERTScore"] = benchmark_data.get("BERTScore", None)
                results[f"{benchmark}_generation_time"] = benchmark_data.get("generation_time", None)

            if benchmark.startswith("humanevalexplainsynthesize"):
                results[f"{benchmark}_pass@1"] = benchmark_data.get("pass@1", None)
                results[f"{benchmark}_generation_time"] = benchmark_data.get("generation_time", None)
            
            # For other tasks, we'll use pass@1 and pass@10 as the main metric
            elif benchmark in ["mbpp", "humaneval","humanevalplus","mbppplus"]:
                results[f"{benchmark}_pass@1"] = benchmark_data.get("pass@1", None)
                results[f"{benchmark}_pass@10"] = benchmark_data.get("pass@10", None)
                results[f"{benchmark}_generation_time"] = benchmark_data.get("generation_time", None)
    
    return results

def process_json_files(directory_path, output_csv_path, policy="greedy"):
    """Process all JSON files in directory and create CSV"""
    
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    # Filter out files containing 'humanevalexplaindescribe'
    json_files = [f for f in json_files if "humanevalexplaindescribe" not in f]
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return None
    
    # Collect all evaluator models first
    evaluator_models = set()
    model_results = defaultdict(dict)  # To store results by model name
    all_columns = set()  # Keep track of all columns
    
    # First pass: collect all data grouped by model name
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = extract_model_name(os.path.basename(json_file))
            if isinstance(result, tuple):
                model_name, evaluator_model = result
            else:
                model_name, evaluator_model = result, None
            
            benchmark_results = extract_benchmark_results(data)
            
            # Regular benchmark results for the model
            if evaluator_model is None:
                # Add regular benchmark results
                model_results[model_name].update(benchmark_results)
            else:
                # For evaluator results, create a specific column with evaluator name
                evaluator_models.add(evaluator_model)
                
                # Find the humanevalexplainsynthesize pass@1 result
                for key, value in benchmark_results.items():
                    if key.startswith("humanevalexplainsynthesize") and key.endswith("_pass@1"):
                        # Create a column specific to this evaluator
                        evaluator_column = f"heesynthesize_{evaluator_model}_pass@1"
                        model_results[model_name][evaluator_column] = value
                        break
            
            # Update all possible columns
            all_columns.update(model_results[model_name].keys())
            
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if not model_results:
        print("No valid results found")
        return None
    
    # Add model_name and policy to all_columns
    all_columns.add("model_name")
    all_columns.add("policy")
    
    # Sort columns for consistent output
    # First model_name, then policy, then regular benchmarks, then evaluator results
    evaluator_cols = sorted([col for col in all_columns 
                            if col.startswith("heesynthesize_") and col.endswith("_pass@1")])
    other_columns = sorted([col for col in all_columns 
                           if col not in ["model_name", "policy"] 
                           and not col in evaluator_cols])
    
    sorted_columns = ["model_name", "policy"] + other_columns + evaluator_cols
    
    # Prepare final results list
    final_results = []
    for model_name, results in model_results.items():
        row = {"model_name": model_name, "policy": policy}
        row.update(results)
        final_results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(final_results)
    # Reorder columns
    df = df.reindex(columns=sorted_columns, fill_value=None)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    
    return df

def merge_without_duplicates(df1, df2, on='model_name'):
    """Merge dataframes without creating duplicates with .1 suffix"""
    if df1 is None and df2 is None:
        return None
    elif df1 is None:
        return df2
    elif df2 is None:
        return df1
    
    # Get columns from second dataframe, excluding the merge column
    cols_to_use = [col for col in df2.columns if col != on and col not in df1.columns]
    
    # Merge only using these non-duplicate columns
    if cols_to_use:
        return pd.merge(df1, df2[[on] + cols_to_use], on=on, how='outer')
    else:
        return df1  # Nothing new to merge

def process_policy_directories(base_dir=".", policies=None):
    """Process all policy directories and combine results"""
    
    if policies is None:
        policies = ["greedy", "nucleus", "mix"]
    
    print("Correctness Data Extractor")
    print("=" * 50)
    
    all_dataframes = []
    
    for policy in policies:
        # Define directory paths for regular and HEE
        correctness_dir = os.path.join(base_dir, "correctness", policy, "metrics")
        correctness_hee_dir = os.path.join(base_dir, "correctness_hee", policy, "metrics")
        
        # Check if directories exist
        correctness_exists = os.path.exists(correctness_dir) and os.path.isdir(correctness_dir)
        correctness_hee_exists = os.path.exists(correctness_hee_dir) and os.path.isdir(correctness_hee_dir)
        
        if not correctness_exists:
            print(f"Warning: Directory '{correctness_dir}' does not exist.")
        if not correctness_hee_exists:
            print(f"Warning: Directory '{correctness_hee_dir}' does not exist.")
        
        if not correctness_exists and not correctness_hee_exists:
            print(f"Error: Neither correctness directories exist for policy '{policy}'.")
            continue
        
        # Process each directory if it exists
        df_correctness = None
        df_correctness_hee = None
        
        if correctness_exists:
            df_correctness = process_json_files(correctness_dir, f"temp_correctness_{policy}.csv", policy)
        
        if correctness_hee_exists:
            df_correctness_hee = process_json_files(correctness_hee_dir, f"temp_correctness_hee_{policy}.csv", policy)
        
        # Merge the dataframes for this policy
        df_policy = merge_without_duplicates(df_correctness, df_correctness_hee)
        
        if df_policy is not None:
            all_dataframes.append(df_policy)
            
            # Save individual policy file
            policy_output_path = f"correctness_{policy}_metrics.csv"
            #df_policy.to_csv(policy_output_path, index=False)
            print(f"✓ Policy '{policy}' processed successfully: {len(df_policy)} models")
            
            # Clean up temporary files
            temp_files = [f"temp_correctness_{policy}.csv", f"temp_correctness_hee_{policy}.csv"]
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    # Combine all policies into one dataframe
    if all_dataframes:
        df_combined = pd.concat(all_dataframes, ignore_index=True)
        
        # Save the combined dataframe
        output_path = "correctness_combined.csv"
        # Drop duplicate rows based on model_name, keeping the first occurrence
        df_combined = df_combined.drop_duplicates(subset=["model_name", "policy"], keep="first")
        df_combined.to_csv(output_path, index=False)
        
        print(f"\n✓ Combined results saved to: {output_path}")
        print(f"Total rows: {len(df_combined)}")
        print(f"Total columns: {len(df_combined.columns)}")
        
        # Verify no duplicate columns with .1 suffix
        duplicate_cols = [col for col in df_combined.columns if '.1' in col]
        if duplicate_cols:
            print(f"Warning: Found duplicate columns: {duplicate_cols}")
        else:
            print("✓ No duplicate columns found")
            
        return df_combined
    else:
        print("Error: No data was processed from any policy directories.")
        return None

def main():
    """Main function that processes correctness directories"""
    process_policy_directories()

if __name__ == "__main__":
    main()