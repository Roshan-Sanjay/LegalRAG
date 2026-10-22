import math
from numbers import Real
import numpy as np
import pandas as pd


class CitationGenerator:
    @staticmethod
    def generate(retrieved_chunks):
        citations = []
        for rank, chunk in enumerate(retrieved_chunks or [], start=1):
            title = chunk.get("case_title")
            missing_title = title is None
            if not missing_title:
                try:
                    missing = pd.isna(title)
                    missing_title = isinstance(missing, (bool, np.bool_)) and bool(missing)
                except (TypeError, ValueError):
                    missing_title = False
            if missing_title or (isinstance(title, str) and not title.strip()):
                continue
            citation = {"rank": rank, "case_title": title}
            for key in ("case_id", "citation", "court", "year", "source"):
                value = chunk.get(key)
                if value is not None and not (isinstance(value, str) and not value):
                    citation[key] = value
            for key in ("final_score", "hybrid_score", "score"):
                value = chunk.get(key)
                if isinstance(value, Real):
                    numeric_score = float(value)
                    citation["score"] = round(numeric_score, 4) if math.isfinite(numeric_score) else None
                    break
            citations.append(citation)
        return citations
