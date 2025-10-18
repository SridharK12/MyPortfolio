import json
import joblib
import pandas as pd
import numpy as np
import os

# Global variable for the model
model = None

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'diabetes_pipeline.joblib')
    model = joblib.load(model_path)
    print("Model loaded successfully from", model_path)

def run(raw_data):
    try:
        # Parse the input JSON data
        data = json.loads(raw_data)
        
        # Convert to DataFrame (Azure sends JSON as a dict)
        df = pd.DataFrame(data['data'])
        
        # Make predictions
        predictions = model.predict(df)
        
        # Optionally convert numpy types to Python native types
        preds = predictions.tolist()
        return json.dumps({"predictions": preds})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
