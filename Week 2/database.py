import pandas as pd
import os
from datetime import datetime

CSV_PATH = "data/submissions.csv"
COLUMNS = ["timestamp", "input_text", "classification"]

def init_db():
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=COLUMNS).to_csv(CSV_PATH, index=False)

def add_entry(input_text: str, classification: str):
    init_db()
    df = pd.read_csv(CSV_PATH)
    new_row = {"timestamp": datetime.now().isoformat(),
               "input_text": input_text, "classification": classification}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

def load_db() -> pd.DataFrame:
    init_db()
    return pd.read_csv(CSV_PATH)