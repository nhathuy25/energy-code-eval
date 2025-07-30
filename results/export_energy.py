import os
import pandas as pd
import argparse
import math

# Define a function to extract model family and assign suffix priority
def get_model_sort_key(model_name):
    # Define the suffix priority (base: 0, AWQ: 1, GPTQ: 2, HQQ: 3)
    if "_4bits64gs_HQQ" in model_name or "_4bitgs64_HQQ" in model_name:
        suffix_priority = 3
        base_name = model_name.split("_4bit")[0]
    elif "GPTQ" in model_name:
        suffix_priority = 2
        base_name = model_name.replace("-GPTQ", "")
    elif "AWQ" in model_name:
        suffix_priority = 1
        base_name = model_name.replace("-AWQ", "")
    else:
        suffix_priority = 0  # Base model
        base_name = model_name
        
    # Return tuple for sorting: first by base model name, then by suffix priority
    return (base_name, suffix_priority)


def process_experiment_files(base_directory, experiment_type):
    """
    Process energy files for different experiment types (batching or scheduler)
    Args:
        base_directory: Root directory containing experiment results
        experiment_type: Either 'batching' or 'scheduler'
    """
    tasks = ["humaneval", "mbpp", "codesearchnet-python"]
    all_data = []
    processed_files = 0

    # Define subdirectories based on experiment type
    if experiment_type == 'batching':
        subdirs = [d for d in os.listdir(base_directory) if d.startswith('n') and os.path.isdir(os.path.join(base_directory, d))]
        column_name = 'n_samples'
    else:  # scheduler
        subdirs = ['single_step', 'multi_step', 'chunked_prefill']
        column_name = 'scheduler'

    for subdir in subdirs:
        # Construct path to energy folder
        energy_dir = os.path.join(base_directory, subdir, 'energy')
        
        if not os.path.isdir(energy_dir):
            print(f"Warning: Energy directory not found in {subdir}")
            continue

        # Process each CSV file in the energy directory
        for filename in os.listdir(energy_dir):
            if filename.endswith(".csv"):
                for task in tasks:
                    if task in filename:
                        filepath = os.path.join(energy_dir, filename)
                        try:
                            df = pd.read_csv(filepath)
                            processed_files += 1

                            # Extract model name
                            model_name = filename.replace(f"_{task}.csv", "")

                            # Extract mns, max-toks, n_samples from window_name
                            import re
                            def extract_window_info(window_name):
                                match = re.search(r"mns(\d+)_max-toks(\d+)_n(\d+)", window_name)
                                if match:
                                    return int(match.group(1)), int(match.group(2)), int(match.group(3))
                                else:
                                    return None, None, None

                            df[["mns", "max_tokens", "n_samples"]] = df["window_name"].apply(
                                lambda x: pd.Series(extract_window_info(x))
                            )

                            # Add metadata
                            df["model"] = model_name
                            df["task"] = task
                            if experiment_type != "batching":
                                df[column_name] = subdir  # Add experiment type information for scheduling experiments

                            # Compute derived metrics
                            df["energy_per_token"] = df["gpu0_energy"] / df["num_out_tokens"].replace(0, pd.NA)
                            df["time_per_token_ms"] = df["elapsed_time"] / df["num_out_tokens"].replace(0, pd.NA) * 1000
                            df["token_per_second"] = df["num_out_tokens"] / df["elapsed_time"].replace(0, pd.NA) 
                            df["avg_power"] = df["gpu0_energy"] / df["elapsed_time"].replace(0, pd.NA) 

                            # Extract the Throughput for only decoding parts
                            df["decode_token_per_second"] = (df["num_out_tokens"] - df["n_samples"]) / (df["elapsed_time"] - df["first_token_time"]*(df["n_samples"]/df["mns"])).replace(0, pd.NA)
                            # Inter-token latency for only decoding parts
                            df["decode_time_per_token_ms"] = (df["elapsed_time"] - df["first_token_time"]*(df["n_samples"]/df["mns"])) / (df["num_out_tokens"] - df["n_samples"]).replace(0, pd.NA) * 1000

                            # Number of tokens Prefill:Decode ratio 
                            df["PD_ratio"] = df["num_out_tokens"] / df["num_in_tokens"].replace(0, pd.NA)

                            # Reorder columns: metadata first, then metrics
                            meta_cols = ["model", "task"]
                            if experiment_type != 'batching':
                                meta_cols.append(column_name)
                            cols = meta_cols + [col for col in df.columns if col not in meta_cols]
                            df = df[cols]

                            all_data.append(df)
                        except Exception as e:
                            print(f"Failed to process {filename}: {e}")
                        break

    # Combine and save
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Drop duplicates based on model, mns, n_samples, and max_tokens
        if experiment_type == 'batching':
            combined_df = combined_df.drop_duplicates(subset=["model", "mns", "n_samples", "max_tokens"])
        # For scheduler experiments, drop duplicates based on scheduler type also
        else: 
            combined_df = combined_df.drop_duplicates(subset=["model", "scheduler", "mns", "n_samples", "max_tokens"])
        
        # Create output filename based on experiment type
        output_name = f"{experiment_type}_combined.csv"
        combined_df.to_csv(output_name, index=False)
        print(f"Saved combined file to: {output_name}")
        print(f"Processed {processed_files} files")
        
        return combined_df
    else:
        print("No valid CSV files found to combine.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine energy metrics from experiments.")
    parser.add_argument("directory", type=str, help="Base directory containing experiment results")
    parser.add_argument("--type", choices=['batching', 'scheduler'], required=True,
                      help="Type of experiment to process")
    args = parser.parse_args()
    
    df = process_experiment_files(args.directory, args.type)