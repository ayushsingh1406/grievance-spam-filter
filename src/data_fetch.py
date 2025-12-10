import pandas as pd
import os

def load_sms(path: str = "data/raw/sms_spam_collection.csv") -> pd.DataFrame:
    """
    Load SMS or grievance datasets from CSV or TSV.
    Automatically detects tab-separated (.tsv) files.
    Expected columns: [label, text].
    """

    # If file extension is TSV → use tab separator
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", header=0, names=["label", "text"])
    else:
        # CSV loader with fallback for irregular formatting
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except Exception:
            df = pd.read_csv(path, encoding='utf-8', engine='python', quotechar='"', skipinitialspace=True)

        # keep only first two columns
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
        df.columns = ["label", "text"]

    # Map ham/spam → 0/1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1}).fillna(df['label'])

    # If any labels are still non-numeric, try converting
    try:
        df['label'] = df['label'].astype(int)
    except:
        pass

    return df


if __name__ == "__main__":
    df = load_sms("data/raw/sms_full.tsv")  # test loading TSV
    print(df.head())
    print("Counts:\n", df.label.value_counts())
