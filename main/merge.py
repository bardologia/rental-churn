from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def merge_parquet_files(input_dir: str, output_file: str) -> str:
    base_path = Path(input_dir)
 
    pattern = "**/*.parquet" 
    parquet_files = sorted(base_path.glob(pattern))
   
    frames = []
    resolved_user_id_col = "usuarioId"
    for file_path in parquet_files:
        table = pq.read_table(file_path)
        frame = table.to_pandas(ignore_metadata=True, strings_to_categorical=False)
        chunk_unique_users = frame[resolved_user_id_col].nunique(dropna=True)
        print(f"{file_path.name}: {chunk_unique_users} unique {resolved_user_id_col}")
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged_unique_users = merged[resolved_user_id_col].nunique(dropna=True)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    print(f"Merged: {merged_unique_users} unique {resolved_user_id_col}")

    return str(output_path)


def main() -> None:
    input_dir   = r"C:\Users\victo\Desktop\rental-churn - 2\raw_data\chunks"
    output_path = r"C:\Users\victo\Desktop\rental-churn - 2\raw_data\raw.parquet"

    output_path = merge_parquet_files(
        input_dir=input_dir,
        output_file=output_path,
    )
    print(f"Merged parquet saved to: {output_path}")


if __name__ == "__main__":
    main()
