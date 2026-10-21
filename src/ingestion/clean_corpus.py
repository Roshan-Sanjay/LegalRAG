import pandas as pd
import re


MIN_TEXT_LENGTH = 500


def normalize_text(text: str) -> str:
    """
    Clean and normalize legal text.
    """

    if not isinstance(text, str):
        return ""

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove weird unicode characters
    text = text.replace("\xa0", " ")

    # Strip leading/trailing spaces
    text = text.strip()

    return text

def is_valid_case(title: str) -> bool:
    """
    Keep likely judicial cases only.
    """

    if not isinstance(title, str):
        return False

    title = title.lower()

    blacklist = [
        "constitution of india",
        "section ",
        "article ",
        "act,",
        "rules,"
    ]

    for word in blacklist:
        if word in title:
            return False

    return True


def clean_dataset(input_path: str, output_path: str):

    print("\nLoading dataset...")

    df = pd.read_csv(input_path)

    print(f"Original dataset size: {len(df)}")

    # ---------------------------------
    # REMOVE NULLS
    # ---------------------------------
    df = df.dropna(subset=["text"])

    print(f"After null removal: {len(df)}")

    # ---------------------------------
    # NORMALIZE TEXT
    # ---------------------------------
    print("\nNormalizing text...")

    df["text"] = df["text"].apply(normalize_text)

    # ---------------------------------
    # REMOVE SHORT DOCUMENTS
    # ---------------------------------
    print("\nRemoving short cases...")

    df = df[df["text"].str.len() > MIN_TEXT_LENGTH]

    print(f"After short text removal: {len(df)}")

    # ---------------------------------
    # REMOVE DUPLICATES
    # ---------------------------------
    print("\nRemoving duplicate cases...")

    before = len(df)

    df = df.drop_duplicates(subset=["text"])

    after = len(df)

    print(f"Removed {before - after} duplicate cases")

    # ---------------------------------
    # RESET INDEX
    # ---------------------------------
    df = df.reset_index(drop=True)

    # ---------------------------------
    # SAVE CLEANED DATASET
    # ---------------------------------
    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to:\n{output_path}")

    print("\nFinal dataset preview:\n")
    print(df.head())


def main():

    input_path = "data/raw/judgments.csv"

    output_path = "data/cleaned/cleaned_judgments.csv"

    clean_dataset(input_path, output_path)


if __name__ == "__main__":
    main()