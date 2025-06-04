import subprocess
import json
from predict_url import predict_from_json

def run_script(script_name, args=[]):
    """Run a Python script as a subprocess with optional arguments."""
    try:
        result = subprocess.run(["python", script_name] + args, check=True, capture_output=True, text=True)
        print(f"{script_name} output:\n", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e.stderr}")
        return False

def run_phishing_detection(url):
    """Execute all scripts sequentially and return the final phishing prediction."""
    
    print("Running web scraper...")
    if not run_script("website_scraper.py", [url, "-o", "analysis.json"]):
        return {"error": "Failed to run website_scraper.py"}
    
    print("Converting features...")
    if not run_script("convert_to_features.py"):  
        return {"error": "Failed to run convert_features.py"}
    
    print("Making prediction...")
    prediction_result = predict_from_json("phishing_features.json")
    
    return prediction_result

if __name__ == "__main__":
    url = "https://forms.fillout.com/t/fZG6yy2SxRus"
    final_result = run_phishing_detection(url)
    print("Final Prediction Result:", json.dumps(final_result, indent=4))
