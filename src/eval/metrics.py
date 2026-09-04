import json
import logging
import os
from typing import Optional


class DiagnosisMetrics:
    """
    An optimized metrics class that:
    - Uses streaming processing to keep memory usage low.
    - Writes records in real time to preserve results.
    - Improves performance compared with the original implementation.
    """
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.total = 0
        self.correct = 0
        self.partial = 0
        
        # Resume-friendly behavior: preserve existing results and append only missing cases.
        try:
            if not os.path.exists(self.output_path):
                with open(self.output_path, "w") as f:
                    pass
        except IOError as e:
            logging.error(f"Could not initialize output file {self.output_path}: {e}")


    def add_result(self, hadm_id: int, ground_truth: str, assistant_diag: str, probability: Optional[float], reason: str, rprob: Optional[float], parsed: dict):
        """Update counters and write one result record immediately."""
        # 1. Update in-memory counters.
        self.total += 1
        decision = parsed.get("decision", False)  # Safely read the decision value.
        partial_credit = parsed.get("partial_credit", False)
        if decision:
            self.correct += 1
        elif partial_credit:
            self.partial += 1

        # 2. Prepare and persist the result record.
        record = {
            "hadm_id": hadm_id,
            "ground_truth": ground_truth,
            "assistant_diagnosis": assistant_diag,
            "geometric_mean_probability": probability,
            "assistant_reasoning": reason, 
            "reason_geometric_mean_probability": rprob,
            "decision": decision,
            "partial_credit": partial_credit,
            "missed_gold_diagnoses": parsed.get("missed_gold_diagnoses", []),
            "reasoning": parsed.get("reasoning"),
            # "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(self.output_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except IOError as e:
            logging.error(f"Could not write record for HADM_ID {hadm_id} to file: {e}")

    def summary(self):
        """Print a summary from the counters accumulated in memory."""
        if self.total == 0:
            print("No results to summarize.")
            return
            
        accuracy = (self.correct / self.total * 100) if self.total > 0 else 0
        partial_rate = (self.partial / self.total * 100) if self.total > 0 else 0
        print(f"✅ Diagnosis Accuracy: {self.correct}/{self.total} ({accuracy:.2f}%)")
        print(f"🟡 Partial Credit: {self.partial}/{self.total} ({partial_rate:.2f}%)")
        print(f"📄 Full results saved to {self.output_path}")
