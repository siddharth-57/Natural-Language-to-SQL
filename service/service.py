import re
import torch
import bentoml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

DANGEROUS_KEYWORDS = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "REPLACE", "ATTACH", "DETACH"]

def validate_sql(sql, header):
    sql_upper = sql.upper().strip()

    if not sql_upper.startswith("SELECT"):
        return False, "Not a SELECT statement"

    for kw in DANGEROUS_KEYWORDS:
        if re.search(rf'\b{kw}\b', sql_upper):
            return False, f"Contains disallowed keyword: {kw}"

    if re.search(r';\s*\S', sql):
        return False, "Multiple statements not allowed"

    quoted_cols = re.findall(r'"([^"]+)"', sql)
    for col in quoted_cols:
        if col not in header and col != "table":
            return False, f"Unknown column referenced: {col}"

    return True, "Valid"


@bentoml.service
class NLToSQLService:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

        base = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")
        self.model = PeftModel.from_pretrained(base, "siddharth57/codet5-wikisql-run2")
        self.model.eval()
        self.model.to(self.device)

        self.cache = {}

    def _generate(self, question, columns):
        columns_str = ", ".join(columns)
        input_text = f"translate to SQL: {question} | columns: {columns_str}"

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=64)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    @bentoml.api
    def query(self, question: str, columns: list[str]) -> dict:
        cache_key = (question.strip().lower(), tuple(columns))

        if cache_key in self.cache:
            return {"sql": self.cache[cache_key], "cached": True, "valid": True}

        sql = self._generate(question, columns)
        is_valid, reason = validate_sql(sql, columns)

        result = {"sql": sql, "cached": False, "valid": is_valid}
        if not is_valid:
            result["reason"] = reason
        else:
            self.cache[cache_key] = sql

        return result