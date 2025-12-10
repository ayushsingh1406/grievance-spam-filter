import os
import pandas as pd
from src.data_fetch import load_sms

def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    sms_small_path = os.path.join(base, "data", "raw", "sms_small.csv")
    sms_full_path = os.path.join(base, "data", "raw", "sms_full.tsv")
    grievance_path = os.path.join(base, "data", "raw", "grievance_500.csv")

    # Load datasets using your unified loader
    print("Loading SMS small dataset...")
    df_small = load_sms(sms_small_path)

    print("Loading SMS full dataset...")
    df_full = load_sms(sms_full_path)

    print("Loading grievance dataset...")
    df_grievance = load_sms(grievance_path)

    print("Shapes:")
    print("Small:", df_small.shape)
    print("Full:", df_full.shape)
    print("Grievance:", df_grievance.shape)

    # Combine
    df = pd.concat([df_small, df_full, df_grievance], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    output_path = os.path.join(base, "data", "processed", "merged_dataset.csv")
    os.makedirs(os.path.join(base, "data", "processed"), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Merged dataset saved to:", output_path)
    print("Final shape:", df.shape)
    print(df.label.value_counts())

if __name__ == "__main__":
    main()
