import os
import pandas as pd
import argparse

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
        subdirs = [d for d in os.listdir(base_directory) if d.startswith('mns')]
        column_name = 'batching'
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

                            # Add metadata
                            df["model"] = model_name
                            df["task"] = task
                            df[column_name] = subdir  # Add experiment type information

                            # Compute derived metrics safely
                            df["energy_per_token"] = df["gpu0_energy"] / df["num_out_tokens"].replace(0, pd.NA)
                            df["time_per_token_ms"] = df["elapsed_time"] / df["num_out_tokens"].replace(0, pd.NA) * 1000
                            df["token_per_second"] = df["num_out_tokens"] / df["elapsed_time"].replace(0, pd.NA) 

                            # Reorder columns: metadata first, then metrics
                            cols = ["model", "task", column_name] + [col for col in df.columns if col not in ["model", "task", column_name]]
                            df = df[cols]

                            all_data.append(df)
                        except Exception as e:
                            print(f"Failed to process {filename}: {e}")
                        break

    # Combine and save
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
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