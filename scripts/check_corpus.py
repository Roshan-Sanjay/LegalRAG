import pandas as pd

df = pd.read_csv(
    "data/chunks/chunked_judgments.csv"
)

keywords = [
    "employee",
    "employer",
    "termination",
    "dismissal",
    "labour",
    "industrial dispute",
    "retrenchment"
]

for word in keywords:
    count = (
        df["chunk_text"]
        .str.contains(
            word,
            case=False,
            na=False
        )
        .sum()
    )

    print(word, count)