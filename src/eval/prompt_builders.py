
import json
from typing import Any, Optional


class PromptBuilder:
    def __init__(self, match_criterion: str):
        self.criterion = match_criterion

    def build(
        self,
        ground_truth: Any,
        assistant_diag: str,
        assistant_reasoning: Optional[str] = None,
    ) -> str:
        if isinstance(ground_truth, dict) and "gold_diagnoses" in ground_truth:
            return self._build_vivabench_prompt(
                ground_truth,
                assistant_diag,
                assistant_reasoning,
            )

        eval_prompt = f"""You are a medical expert. You will be provided with a ground truth diagnosis and an assistant’s diagnosis. 
                These can be on different levels of specificity. Therefore you also receive a matching criterion. 
                Your task is to determine if the assistant's diagnosis matches the ground truth diagnosis based on the matching criterion.
                Respond with the given json schema, providing a reasoning for your answer and a boolean decision (True if they match, False otherwise).
                Mostly consider the overall 'Matching criterion', not any specific details. 
                Decide false if the ground truth and assistant diagnoses conflict with each other. 

                Example 1:
                Ground Truth: 'Appendicitis'
                Assistant: 'Complicated appendicitis with local peritonitis and perforation'
                Matching criterion: 'appendicitis'
                Decision: 'True'
                Example 2:
                Ground Truth: 'Appendicitis'
                Assistant: 'Appendicitis with local peritonitis'
                Matching criterion: 'appendicitis'
                Decision: 'True'
                Example 3:
                Ground Truth: 'Acute appendicitis with appendicolith'
                Assistant: 'Appendicitis without local peritonitis'
                Matching criterion: 'appendicitis'
                Decision: 'True'
                Example 4:
                Ground Truth: 'Acute cholecystitis'
                Assistant: 'Appendicitis with local peritonitis'
                Matching criterion: 'cholecystitis'
                Decision: 'False'
                Example 5:
                Ground Truth: 'Acute cholecystitis'
                Assistant: 'Choledocholithiasis with possible cholecystitis'
                Matching criterion: 'cholecystitis'
                Decision: 'True'
                Example 6:
                Ground Truth: 'other pulmonary embolism and infarction'
                Assistant: 'massive pulmonary embolism with right ventricular strain'
                Matching criterion: 'Pulmonary Embolism'
                Decision: 'True'
                Example 7:
                Ground Truth: 'Sepsis, unspecified organism'
                Assistant: 'Diverticulitis with perforation'
                Matching criterion: 'diverticulitis'
                Decision: 'True'

                Ground Truth: {ground_truth}
                Assistant: {assistant_diag}
                Matching criterion: {self.criterion}

                First, analyze the relationship between the ground truth diagnosis, the assistant's diagnosis, and the matching criterion. Explain relevant clinical reasoning.
                Then, output your judgment in the following strict JSON format: 
                ```json
                {{
                "reasoning": "Explain in clear and concise clinical logic why the diagnoses match or not, in one paragraph.",
                "decision": true  // or false
                }}
                """
     
        return eval_prompt

    def _build_vivabench_prompt(
        self,
        ground_truth: dict[str, Any],
        assistant_diag: str,
        assistant_reasoning: Optional[str],
    ) -> str:
        gold_diagnoses = ground_truth.get("gold_diagnoses", [])
        gold_differentials = ground_truth.get("gold_differentials", [])

        return f"""
        You are a medical expert evaluating a benchmark diagnosis.

        You will be given:
        - gold diagnoses for the case
        - accepted differential diagnoses
        - the assistant's final diagnosis

        The case may have multiple correct diagnoses.

        Your task is to judge whether the assistant's diagnosis is:
        - fully correct,
        - partially correct,
        - or incorrect.

        Rules:
        - Set "decision" to true only if the assistant identifies all clinically central gold diagnoses, or gives an equivalent diagnosis that correctly covers them, without adding a major contradictory diagnosis.
        - Set "partial_credit" to true if the assistant identifies at least one important gold diagnosis, or a closely related accepted differential, but misses one or more important gold diagnoses.
        - If the assistant identifies only an accepted differential and none of the gold diagnoses, it cannot be fully correct.
        - If the assistant diagnosis is clinically unrelated to the gold diagnoses, set both "decision" and "partial_credit" to false.
        - If the assistant includes a major contradictory diagnosis, set "decision" to false.
        - List any important missed gold diagnoses in "missed_gold_diagnoses".
        - Use clinical meaning, not exact wording.

        Examples:
        Example 1:
        Gold diagnoses: ["Legionnaires' disease (Legionella pneumonia)", "Rhabdomyolysis", "Acute kidney failure"]
        Accepted differentials: []
        Assistant diagnosis: "Legionnaires' disease, rhabdomyolysis, and acute kidney injury"
        Output:
        ```json
        {{
          "reasoning": "The assistant captures all major gold diagnoses with clinically equivalent wording. Acute kidney injury is an acceptable wording variant for acute kidney failure, so the answer is overall correct.",
          "decision": true,
          "partial_credit": false,
          "missed_gold_diagnoses": []
        }}
        ```

        Example 2:
        Gold diagnoses: ["Legionnaires' disease (Legionella pneumonia)", "Rhabdomyolysis", "Acute kidney failure"]
        Accepted differentials: []
        Assistant diagnosis: "Legionnaires' disease"
        Output:
        ```json
        {{
          "reasoning": "The assistant correctly identifies one important gold diagnosis but misses the associated rhabdomyolysis and acute kidney failure. The answer is incomplete rather than fully correct.",
          "decision": false,
          "partial_credit": true,
          "missed_gold_diagnoses": ["Rhabdomyolysis", "Acute kidney failure"]
        }}
        ```

        Example 3:
        Gold diagnoses: ["New-onset thyrotoxicosis"]
        Accepted differentials: ["Acalculous cholecystitis"]
        Assistant diagnosis: "Acalculous cholecystitis"
        Output:
        ```json
        {{
          "reasoning": "The assistant names only an accepted differential and fails to identify the gold diagnosis of new-onset thyrotoxicosis. The answer is not overall correct, but it is clinically related enough to merit partial credit.",
          "decision": false,
          "partial_credit": true,
          "missed_gold_diagnoses": ["New-onset thyrotoxicosis"]
        }}
        ```

        Example 4:
        Gold diagnoses: ["Giant cell arteritis (temporal arteritis)"]
        Accepted differentials: []
        Assistant diagnosis: "Parapneumonic effusion secondary to pneumonia"
        Output:
        ```json
        {{
          "reasoning": "The assistant provides a diagnosis that is clinically unrelated to the gold diagnosis of giant cell arteritis. The gold diagnosis is entirely missed and the answer is incorrect.",
          "decision": false,
          "partial_credit": false,
          "missed_gold_diagnoses": ["Giant cell arteritis (temporal arteritis)"]
        }}
        ```

        Example 5:
        Gold diagnoses: ["Acute pulmonary embolism", "Right heart strain"]
        Accepted differentials: []
        Assistant diagnosis: "Pulmonary embolism"
        Output:
        ```json
        {{
          "reasoning": "The assistant identifies the main disease process of pulmonary embolism but misses the associated right heart strain, which is an important gold diagnosis. The answer is clinically meaningful but incomplete.",
          "decision": false,
          "partial_credit": true,
          "missed_gold_diagnoses": ["Right heart strain"]
        }}
        ```

        Example 6:
        Gold diagnoses: ["Acute pancreatitis"]
        Accepted differentials: ["Biliary colic"]
        Assistant diagnosis: "Acute pancreatitis with acute appendicitis"
        Output:
        ```json
        {{
          "reasoning": "The assistant identifies the gold diagnosis of acute pancreatitis, but also adds acute appendicitis, which is a major contradictory diagnosis not supported by the gold labels. Therefore the answer is not fully correct, though it still captures an important gold diagnosis.",
          "decision": false,
          "partial_credit": true,
          "missed_gold_diagnoses": []
        }}
        ```

        Gold diagnoses: {json.dumps(gold_diagnoses, ensure_ascii=False)}
        Accepted differentials: {json.dumps(gold_differentials, ensure_ascii=False)}
        Assistant diagnosis: {assistant_diag}

        Return strict JSON only:
        ```json
        {{
          "reasoning": "Explain briefly why the assistant is correct, partially correct, or incorrect.",
          "decision": true,
          "partial_credit": false,
          "missed_gold_diagnoses": []
        }}
        ```"""