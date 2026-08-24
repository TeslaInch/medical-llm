import requests
import json
import time

# 1. Import from your custom MedEval Framework
from medeval.models.base import BaseModelConnector
from medeval.benchmark import BenchmarkLoader
from medeval.runner import BenchmarkRunner
from medeval.safety import SickleCellSafetyChecker
from medeval.report import export_report_to_json

# 2. Define the Custom Connector that queries your live HuggingFace API
class SCDAPIConnector(BaseModelConnector):
    def __init__(self, api_url: str):
        super().__init__(model_name="SCD-Medical-LLM-API")
        self.api_url = api_url
        self.last_confidence = 0.0  # Store confidence for the probabilities method

    def generate(self, prompt: str) -> str:
        """Sends the question to the live API and returns the generated answer."""
        payload = {
            "question": prompt,
            "case": "Evaluation via MedEval Framework",
            "history": []
        }
        print(f"[*] Querying API: {prompt[:50]}...")
        
        try:
            # We add a longer timeout since the free CPU tier takes a few minutes
            response = requests.post(self.api_url, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            
            # Store the RAG confidence score to expose to MedEval
            self.last_confidence = data.get("confidence", 0.0)
            return data.get("answer", "")
            
        except Exception as e:
            print(f"[!] API Request Failed: {e}")
            return "Error generating response."

    def generate_probabilities(self, prompt: str) -> list[float]:
        """Returns the confidence score stored from the last generate() call."""
        # MedEval expects a list of probabilities (e.g., for multi-choice). 
        # We just return the single RAG confidence score as a list.
        return [self.last_confidence]

if __name__ == "__main__":
    
    LIVE_API_URL = "https://teslainch-scd-medical-llm.hf.space/predict"
    
    print("="*50)
    print("Running MedEval Framework against SCD-Medical-LLM")
    print("="*50)
    
    # Initialize the custom API connector
    connector = SCDAPIConnector(api_url=LIVE_API_URL)
    
    # Initialize your custom Sickle Cell Safety Checker
    safety_checker = SickleCellSafetyChecker()
    
    # Using 3 samples from the MedQA benchmark (limited because of CPU inference time)
    print("Loading benchmark dataset...")
    loader = BenchmarkLoader(split="test", max_samples=3)
    samples = loader.load_medqa()
    
    # Run the evaluation
    print(f"Starting evaluation on {len(samples)} samples. This may take ~15 minutes on the free tier CPU...")
    runner = BenchmarkRunner(model=connector, safety_checker=safety_checker, ignore_errors=True)
    report = runner.run(samples)
    
    # Export the report
    output_file = "evaluation_report.json"
    export_report_to_json(report, output_file)
    print(f"\n✅ Evaluation complete! Structured report saved to {output_file}")
