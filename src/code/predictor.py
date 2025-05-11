# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import io
import sys
import signal
import traceback
import csv
import flask
import time
import pandas as pd
import numpy as np
import copy
from catboost import CatBoostClassifier, Pool as CatboostPool, cv
from pandas.api.types import CategoricalDtype
import tarfile
from loguru import logger

# Configure loguru
logger.remove()  # Remove default handler
logger.add(sys.stdout, format="{time} | {level} | {message}")

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

def ret_pool_obj(X, text_features=None, cat_features=None):
    """Create a CatBoost Pool with the available features"""
    # Filter to include only columns that exist in the dataframe
    valid_text_features = [col for col in (text_features or []) if col in X.columns]
    valid_cat_features = [col for col in (cat_features or []) if col in X.columns]
    
    pool_obj = CatboostPool(
        data=X,
        label=None,
        text_features=valid_text_features,
        cat_features=valid_cat_features,
        feature_names=list(X.columns)
    )
    return pool_obj


def pre_process(df, numerical_cols=None, categorical_cols=None, text_cols=None):
    """Preprocess dataframe with the specified column types"""
    # Handle text columns
    for col in (text_cols or []):
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Handle categorical columns
    for col in (categorical_cols or []):
        if col in df.columns:
            df[col] = df[col].fillna('Unk')
    
    # Handle numerical columns
    for col in (numerical_cols or []):
        if col in df.columns:
            df[col] = df[col].fillna(-9999)
    
    return df


def pre_process_cat(df, model, categorical_cols=None):
    """Apply categorical preprocessing to the dataframe"""
    # Use provided categorical columns or empty list if None
    cat_cols = categorical_cols or []
    
    for col_name in cat_cols:
        if col_name in df.columns and col_name in model[0]:
            # Convert to categorical type
            cat_type = CategoricalDtype(categories=model[0].get(col_name), ordered=False)
            df[col_name] = df[col_name].astype(cat_type, copy=False)
            df[col_name] = df[col_name].fillna('Unk')
    
    return df


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded
    column_config = None 

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            logger.info('Loading model...')
            model_file = CatBoostClassifier()
            
            tar_path = os.path.join(model_path, 'model.tar.gz')
            if os.path.exists(tar_path):
                logger.info(f'Found model archive at {tar_path}')
                try:
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(path=model_path)
                    logger.info('Successfully extracted model archive')
                    
                    logger.debug(f"Contents of {model_path}:")
                    for f in os.listdir(model_path):
                        logger.debug(f"  - {f}")
                except Exception as e:
                    logger.error(f'Error extracting model archive: {str(e)}')
                    return None

            model_files = [f for f in os.listdir(model_path) if f.endswith(('.cbm', '.dump'))]
            if not model_files:
                logger.error(f'No model files (.cbm or .dump) found in {model_path}')
                return None
                    
            model_file_path = os.path.join(model_path, model_files[0])
            logger.info(f'Loading model from {model_file_path}')
            
            try:
                model_file.load_model(model_file_path)
                logger.info(f'Model loaded successfully from {model_file_path}')
            except Exception as e:
                logger.error(f'Error loading model: {str(e)}')
                
                try:
                    logger.info("Attempting to load as CatBoostRegressor instead...")
                    from catboost import CatBoostRegressor
                    model_file = CatBoostRegressor()
                    model_file.load_model(model_file_path)
                    logger.info('Model loaded successfully as CatBoostRegressor')
                except Exception as e2:
                    logger.error(f'Error loading model as regressor: {str(e2)}')
                    return None

            obj_col_categories = {}
            try:
                cat_path = os.path.join(model_path, 'obj_col_categories.pkl')
                if os.path.exists(cat_path):
                    with open(cat_path, 'rb') as inp:
                        obj_col_categories = pickle.load(inp)
                        logger.info('Categories loaded successfully')
                else:
                    logger.warning('No categories file found at expected path')
            except Exception as e:
                logger.error(f'Error loading categories: {str(e)}')
            
            try:
                config_path = os.path.join(model_path, 'column_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        cls.column_config = json.load(f)
                        logger.info('Column configuration loaded')
                else:
                    cls.column_config = None
                    logger.warning('No column configuration file found at expected path')
            except Exception as e:
                cls.column_config = None
                logger.error(f'Error loading column configuration: {str(e)}')
            
            cls.model = [obj_col_categories, model_file]
            logger.info('Model initialization complete')
            
        return cls.model

    @classmethod
    def predict(cls, input_df):
        """
        For the input, do the predictions and return them.
        Args:
            input_df (a pandas dataframe): The data on which to do the predictions
        """
        model = cls.get_model()
        logger.info('Running inference...')
        
        # Check if this is a matrix-only input
        is_matrix_input = 'matrix_input' in input_df.columns
        
        # Get column configuration
        if cls.column_config is not None:
            numerical_cols = cls.column_config.get('numerical_cols', [])
            categorical_cols = cls.column_config.get('categorical_cols', [])
            text_cols = cls.column_config.get('text_cols', [])
        else:
            # Only detect numerical columns when no config is available
            numerical_cols = [col for col in input_df.columns if pd.api.types.is_numeric_dtype(input_df[col]) and col != 'matrix_input']
            categorical_cols = []
            text_cols = []
        
        # For matrix input, we only use numerical features
        if is_matrix_input:
            # Create pool with only available numerical columns
            available_num_cols = [col for col in numerical_cols if col in input_df.columns] or [col for col in input_df.columns if col != 'matrix_input']
            input_pool = CatboostPool(
                data=input_df[available_num_cols],
                label=None,
                feature_names=available_num_cols
            )
        else:
            # Regular flow with all available feature types
            # Process the input data
            input_processed = pre_process(input_df, numerical_cols, categorical_cols, text_cols)
            input_processed = pre_process_cat(input_processed, model, categorical_cols)
            
            # Identify available columns of each type
            available_num_cols = [col for col in numerical_cols if col in input_processed.columns]
            available_cat_cols = [col for col in categorical_cols if col in input_processed.columns]
            available_text_cols = [col for col in text_cols if col in input_processed.columns]
            
            # Create feature columns list with all available columns
            feature_cols = available_num_cols + available_cat_cols + available_text_cols
            
            # If no feature columns found, use all columns as a fallback
            if not feature_cols:
                feature_cols = list(input_processed.columns)
            
            # Create pool with available features
            input_pool = ret_pool_obj(
                input_processed[feature_cols],
                text_features=available_text_cols,
                cat_features=available_cat_cols
            )
        
        # Get probabilities
        prob = model[1].predict_proba(input_pool)[:, 1]
        logger.info(f'Completed inference on {prob.shape[0]} records')
        
        # Add scores to output
        result_df = pd.DataFrame()
        
        # Add ID column if available
        id_col = None
        for possible_id in ['id', 'review_id', 'customer_id', 'product_id']:
            if possible_id in input_df.columns:
                id_col = possible_id
                result_df[id_col] = input_df[id_col]
                break
        
        # If no ID column, create index
        if id_col is None:
            result_df['id'] = range(len(prob))
            id_col = 'id'
        
        # Add target if available
        if 'target' in input_df.columns:
            result_df['target'] = input_df['target']
        
        # Add prediction score
        result_df['score'] = prob
        
        return result_df


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(
        response='\n',
        status=status,
        mimetype='application/json'
        )

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    start = time.time()
    
    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        first_line = data.split('\n')[0].strip()
        
        logger.info(f'Input data first line sample: {first_line[:100]}...')
        
        # Check if this is a numerical matrix (no headers)
        first_line = data.split('\n')[0].strip()
        
        is_numeric_matrix = all(
            val.strip().replace('.', '').replace('-', '').isdigit() 
            for val in first_line.split(',') if val.strip()
        )
        
        if is_numeric_matrix:
            logger.info('Detected numerical matrix input without headers')
            try:
                # Detect separator
                sep = '\t' if '\t' in first_line else ','
                
                matrix_df = pd.read_csv(io.StringIO(data), header=None, sep=sep)
                
                # Read matrix without headers
                matrix_df.columns = [f'col_{i+1}' for i in range(len(matrix_df.columns))]
                
                # Flag as matrix input
                matrix_df['matrix_input'] = True
                
                data = matrix_df
                logger.info(
                    f'Matrix processed successfully. Shape: {data.shape}, Columns: {matrix_df.columns}')
            except Exception as e:
                logger.error(f"Error processing matrix: {str(e)}")
                # Fall back to default CSV handling
                is_numeric_matrix = False
        
        if not is_numeric_matrix:
            # Check if the data has headers
            first_char = first_line.split(',')[0].strip()[0] if first_line else ''
            has_headers = not first_char.isdigit()
            
            try:
                sep = '\t' if '\t' in first_line else ','
                
                if has_headers:
                    logger.info('Processing CSV with headers')
                    data = pd.read_csv(io.StringIO(data), sep=sep, lineterminator='\n', 
                                     escapechar='\\', quotechar='"', keep_default_na=False)
                else:
                    logger.info('Processing CSV without headers')
                    data = pd.read_csv(io.StringIO(data), header=None, sep=sep, 
                                     lineterminator='\n', escapechar='\\', quotechar='"',
                                     keep_default_na=False)
                
                logger.info(f'CSV processed successfully. Shape: {data.shape}, Columns: {data.columns.tolist()}')
            except Exception as e:
                logger.error(f"Error processing CSV: {str(e)}")
                return flask.Response(response=f'Error processing input data: {str(e)}', 
                                   status=400, mimetype='text/plain')
    else:
        return flask.Response(response='This predictor only supports CSV data', 
                           status=415, mimetype='text/plain')

    logger.info(f'Invoked with {data.shape[0]} records')

    # Do the prediction
    predictions = ScoringService.predict(data)
    logger.debug(f'Shape of predictions: {predictions.shape}')
    
    # Convert from dataframe back to CSV
    out = io.StringIO()
    predictions.to_csv(out, header=False, index=False, sep='\t', 
                      quotechar='"', escapechar='\\', quoting=csv.QUOTE_NONE)
    result = out.getvalue()
    end = time.time()

    logger.info(f'Time to execute: {end - start:.3f} seconds')
    return flask.Response(response=result, status=200, mimetype='text/csv')
