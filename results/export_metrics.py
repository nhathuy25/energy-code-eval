import json
import csv
import os
import glob
from pathlib import Path

def extract_model_name(filename):
    """Extract model name from filename by removing the benchmark suffix"""
    # Remove the file extension
    base_name = Path(filename).stem
    
    # Define all possible benchmark names
    benchmark_names = [
        "codesearchnet-python",
        "codesearchnet-java", 
        "codesearchnet-javascript",
        "humaneval",
        "mbpp",
        "humanevalexplainsynthesize-python",
        "humanevalexplainsynthesize-java",
        "humanevalexplainsynthesize-javascript"
    ]
    
    # Find the last underscore that precedes any benchmark name
    model_name = base_name
    
    # Look for underscore followed by benchmark names
    import re
    
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
    
    return model_name

def extract_benchmark_results(json_data):
    """Extract benchmark results from JSON data"""
    results = {}
    
    # Define the benchmarks we're interested in
    benchmarks = ["codesearchnet-python", "codesearchnet-java", "codesearchnet-javascript", "humaneval", "mbpp",
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
                results[f"{benchmark}_pass@10"] = benchmark_data.get("pass@10", None)
                results[f"{benchmark}_generation_time"] = benchmark_data.get("generation_time", None)
            
            # For humanevalexplain, we'll use pass@1 as the main metric
            elif benchmark in ["mbpp", "humaneval"]:
                results[f"{benchmark}_pass@1"] = benchmark_data.get("pass@1", None)
                results[f"{benchmark}_pass@10"] = benchmark_data.get("pass@10", None)
                results[f"{benchmark}_generation_time"] = benchmark_data.get("generation_time", None)
    
    return results

def process_json_files(directory_path, output_csv_path):
    """Process all JSON files in directory and create CSV"""
    
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    # Filter out files containing 'humanevalexplaindescribe'
    json_files = [f for f in json_files if "humanevalexplaindescribe" not in f]
    
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    all_results = []
    all_columns = set(["model_name"])  # Start with model_name column
    
    # First pass: collect all data and determine all possible columns
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            model_name = extract_model_name(os.path.basename(json_file))
            benchmark_results = extract_benchmark_results(data)
            
            # Add model name to results
            benchmark_results["model_name"] = model_name
            
            all_results.append(benchmark_results)
            all_columns.update(benchmark_results.keys())
            
            print(f"Processed: {model_name}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if not all_results:
        print("No valid results found")
        return
    
    # Sort columns for consistent output
    sorted_columns = ["model_name"] + sorted([col for col in all_columns if col != "model_name"])
    
    # Write to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted_columns)
        writer.writeheader()
        
        for result in all_results:
            # Fill missing columns with None/empty values
            row = {col: result.get(col, None) for col in sorted_columns}
            writer.writerow(row)
    
    print(f"\nResults saved to: {output_csv_path}")
    print(f"Processed {len(all_results)} models")
    print(f"Columns: {', '.join(sorted_columns)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract benchmark results from JSON files and create a CSV summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_metrics.py -d /path/to/results
  python export_metrics.py --directory ./model_results
  python export_metrics.py -d results/experiment1 -o custom_name.csv
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        required=True,
        help='Directory containing JSON result files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV filename (default: auto-generated from directory name)'
    )
    
    args = parser.parse_args()
    
    # Validate directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return
    
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a directory.")
        return
    
    # Generate output filename based on directory name if not provided
    if args.output is None:
        # Convert directory path to filename (similar to your example)
        filename = args.directory.replace('/', '_').replace('\\', '_').lstrip('_') + ".csv"
        output_path = os.path.join(os.curdir, filename)
    else:
        # Use custom output filename
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            output_path = os.path.join(args.directory, args.output)
    
    print("Benchmark Results Extractor")
    print("=" * 40)
    print(f"Input directory: {args.directory}")
    print(f"Output CSV: {output_path}")
    print()
    
    # Process the files
    process_json_files(args.directory, output_path)

if __name__ == "__main__":
    main()