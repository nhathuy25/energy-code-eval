import os
import pandas as pd
import argparse

def process_experiment_files(base_directory):
    """
    Args:
        base_directory: Root directory containing experiment results
    """
    tasks = ["humaneval", "mbpp", "codesearchnet-python"]
    all_data = []
    processed_files = 0


    # Process each CSV file in the energy directory
    for filename in os.listdir(base_directory):
        if filename.endswith(".csv"):
            for task in tasks:
                if task in filename:
                    filepath = os.path.join(base_directory, filename)
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
                        
                        # Compute derived metrics safely
                        df["energy_per_token"] = df["gpu0_energy"] / df["num_out_tokens"].replace(0, pd.NA)
                        df["time_per_token_ms"] = df["finished_time"] / df["num_out_tokens"].replace(0, pd.NA) * 1000
                        df["token_per_second"] = df["num_out_tokens"] / df["elapsed_time"].replace(0, pd.NA) 

                        # Reorder columns: metadata first, then metrics
                        meta_cols = ["model", "task"]
                        cols = meta_cols + [col for col in df.columns if col not in meta_cols]
                        df = df[cols]

                        all_data.append(df)
                    except Exception as e:
                        print(f"Failed to process {filename}: {e}")
                    break

    # Combine and save
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create output filename based on experiment type
        output_name = f"processed.csv"
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
    args = parser.parse_args()

    df = process_experiment_files(args.directory)
