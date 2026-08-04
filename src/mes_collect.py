import json
import os
from termcolor import colored
from datetime import datetime
from typing import Optional, Union


class EvaluationOutputCollector:
    """
    A class to collect and display output from a simulation of patient and medical doctor interaction and actions.
    """

    def __init__(self, dataset_name, hadm_id, save_dir):
        # directory = os.path.join(save_dir, dataset_name)
        # os.makedirs(directory, exist_ok=True)

        base_filename = f"{dataset_name}_{hadm_id}_conversation"
        extension = ".json"
        counter = 1
        file_path = os.path.join(save_dir, f"{base_filename}_{counter}{extension}")

        while os.path.exists(file_path):
            counter += 1
            file_path = os.path.join(save_dir, f"{base_filename}_{counter}{extension}")

        # Initialize empty JSON
        with open(file_path, "w") as file:
            json.dump([], file)

        self.filename = file_path
        self.items = []  # In-memory log

    def _save_event(self, event: dict):
        """Append an event to the JSON file."""
        with open(self.filename, "r+") as file:
            data = json.load(file)
            data.append(event)
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.truncate()

    # def display_message(self, name: str, message: str):
    def display_message(self, name: str, message: str, thinking: str | None = None):
        """
        Display and store a conversational message from the assistant or user.

        Parameters:
        - name (str): Who sent the message (e.g., "Doctor", "Patient").
        - message (str): The raw message content (Markdown).
        """

        entry = {
            "type": "message",
            "role": name,
            "content": message,
            "thinking": thinking,
            "timestamp": datetime.now().isoformat(),
        }

        self.items.append(entry)
        self._save_event(entry)

        # Console preview
        color = "green" if name == "Doctor" else "blue"
        print(colored(f"{name}: {message}", color))

    def display_action(self, message: str, function_name: str, arguments: Optional[Union[str, list, dict]] = None):
        """
        Display and store a system action (e.g., function call result).

        Parameters:
        - message (str): Description or result from the tool.
        - function_name (str): The name of the function/tool used.
        """
        if isinstance(message, list):
            message = "\n".join(str(m) for m in message)

        entry = {
            "type": "action",
            "role": "System",
            "function": function_name,
            "content": message,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat(),
        }

        self.items.append(entry)
        self._save_event(entry)

        print(colored(f"[{function_name.upper()}] {message}", "magenta"))
    
    # Add this method to EvaluationOutputCollector.
    def display_logprob_analysis(self, function_name: str, arguments: dict, analysis_results: dict):
        """
        Display and store logprob analysis for a function call.
        """
        entry = {
            "type": "logprob_analysis",
            "role": "System",
            "function": function_name,
            "arguments": arguments,
            "analysis": analysis_results,  # Dictionary containing all analysis results.
        }

        self.items.append(entry)
        self._save_event(entry)


        # ---- Confidence preview for legacy and current structures. ----
        confidence = "N/A"

        # 1) First try the legacy structure with a top-level geometric_mean_probability.
        if "geometric_mean_probability" in analysis_results:
            confidence = analysis_results["geometric_mean_probability"]

        else:
            # 2) In the current structure, prefer diagnosis and then reasoning.
            diag_stats = analysis_results.get("diagnosis")
            if isinstance(diag_stats, dict) and "geometric_mean_probability" in diag_stats:
                confidence = diag_stats["geometric_mean_probability"]
            else:
                reason_stats = analysis_results.get("reasoning")
                if isinstance(reason_stats, dict) and "geometric_mean_probability" in reason_stats:
                    confidence = reason_stats["geometric_mean_probability"]

        # Format for display.
        if isinstance(confidence, float):
            confidence_str = f"{confidence:.2%}"
        else:
            confidence_str = str(confidence)

        print(colored(
            f"[LOGPROB ANALYSIS] Confidence for '{function_name}' args: {confidence_str}",
            "yellow"
        ))

    def display_run_metadata(self, metadata: dict):
        """Append one case-level metadata summary after the conversation completes."""
        entry = {
            "type": "run_metadata",
            "role": "System",
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
        }

        self.items.append(entry)
        self._save_event(entry)
        print(colored(f"[RUN METADATA] {metadata}", "cyan"))
