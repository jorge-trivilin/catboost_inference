import json
import os
import numpy as np
import pandas as pd
import tarfile
from io import StringIO, BytesIO
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


def model_fn(model_dir: str):
    """Load the CatBoost model from the model_dir.
    
    Args:
        model_dir (str): The directory where model files are stored.
        
    Returns:
        The loaded CatBoost model.
    """
    # Check if the model is saved as a tar.gz file
    model_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(model_path):
        with tarfile.open(model_path) as tar:
            tar.extractall(path=model_dir)
    
    # Look for model file
    model_file = os.path.join(model_dir, "model")
    
    # Determine if the model is a classifier or regressor
    # This could be determined by a metadata file, here we try both
    try:
        model = CatBoostClassifier()
        model.load_model(model_file)
        return model
    except Exception:
        try:
            model = CatBoostRegressor()
            model.load_model(model_file)
            return model
        except Exception as e:
            raise ValueError(f"Unable to load CatBoost model: {str(e)}")


def input_fn(request_body, request_content_type):
    """Parse input data payload.
    
    Args:
        request_body (str or bytes): The raw input data.
        request_content_type (str): The content type of the request.
        
    Returns:
        A CatBoost-compatible data format (DataFrame or Pool)
    """
    if request_content_type == 'application/json':
        # Parse as JSON
        data = json.loads(request_body)
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
            
        return df
    elif request_content_type == 'text/csv':
        # Parse the CSV
        if isinstance(request_body, bytes):
            data = request_body.decode('utf-8')
        else:
            data = request_body
            
        df = pd.read_csv(StringIO(data), header=None)
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions using the CatBoost model.
    
    Args:
        input_data: Parsed input data from input_fn.
        model: CatBoost model loaded by model_fn.
        
    Returns:
        Model predictions.
    """
    # Check if model is a classifier with predict_proba method
    if hasattr(model, 'predict_proba'):
        predictions = model.predict_proba(input_data)
    else:
        predictions = model.predict(input_data)
        
    return predictions


def output_fn(prediction, accept):
    """Format predictions for return.
    
    Args:
        prediction: The prediction result from predict_fn.
        accept: The accept content type header expected by the client.
        
    Returns:
        Response data in the requested format.
    """
    if accept == 'application/json':
        return json.dumps(prediction.tolist()), accept
    
    elif accept == 'text/csv':
        if prediction.ndim == 1:
            # For regression or binary classification
            output = '\n'.join([str(pred) for pred in prediction])
        else:
            # For multi-class classification
            output = '\n'.join([','.join([str(pred) for pred in row]) for row in prediction])
        
        return output, 'text/csv'
    
    else:
        # Default to JSON
        return json.dumps(prediction.tolist()), 'application/json'