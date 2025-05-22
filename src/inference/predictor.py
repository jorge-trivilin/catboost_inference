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
from typing import Dict, List, Optional, Union, Tuple, Any


logger.add(sys.stdout, format="{time} | {level} | {message}")

prefix: str = '/opt/ml/'
model_path: str = os.path.join(prefix, 'model')

def ret_pool_obj(
    X: pd.DataFrame, 
    text_features: Optional[List[str]] = None, 
    cat_features: Optional[List[str]] = None
    ) -> CatboostPool:
    """Creates a CatBoost Pool object with the specified features.

    Filters provided text and categorical features to only include those present
    in the input dataframe columns.

    Args:
        X (pd.DataFrame): Input data to create pool from
        text_features (Optional[List[str]]): List of column names for text features
        cat_features (Optional[List[str]]): List of column names for categorical features

    Returns:
        CatboostPool: Pool object with configured features for model input

    Example:
        >>> pool = ret_pool_obj(df, text_features=['description'], cat_features=['category'])
    """
    valid_text_features: List[str] = [col for col in (text_features or []) if col in X.columns]
    valid_cat_features: List[str] = [col for col in (cat_features or []) if col in X.columns]
    
    pool_obj: CatboostPool = CatboostPool(
        data=X,
        label=None,
        text_features=valid_text_features,
        cat_features=valid_cat_features,
        feature_names=list(X.columns)
    )
    return pool_obj


def pre_process(
    df: pd.DataFrame, 
    numerical_cols: Optional[List[str]] = None, 
    categorical_cols: Optional[List[str]] = None, 
    text_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
    """Preprocesses dataframe by handling missing values for different column types.

    Fills missing values based on column type:
    - Text columns: Empty string ('')
    - Categorical columns: 'Unk' 
    - Numerical columns: -9999

    Args:
        df (pd.DataFrame): Input dataframe to preprocess
        numerical_cols (Optional[List[str]]): List of numerical column names
        categorical_cols (Optional[List[str]]): List of categorical column names
        text_cols (Optional[List[str]]): List of text column names

    Returns:
        pd.DataFrame: Preprocessed dataframe with filled missing values

    Example:
        >>> df_processed = pre_process(df, 
        ...                           numerical_cols=['age', 'salary'],
        ...                           categorical_cols=['department'],
        ...                           text_cols=['comments'])
    """

    for col in (text_cols or []):
        if col in df.columns:
            df[col] = df[col].fillna('')
    

    for col in (categorical_cols or []):
        if col in df.columns:
            df[col] = df[col].fillna('Unk')
    

    for col in (numerical_cols or []):
        if col in df.columns:
            df[col] = df[col].fillna(-9999)
    
    return df


def pre_process_cat(
    df: pd.DataFrame, 
    model: List[Any], 
    categorical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
    """Applies categorical preprocessing using model-specific categories.

    Converts categorical columns to CategoricalDtype using predefined categories
    from the model configuration. This ensures categorical encoding matches
    the model's training data.

    Args:
        df (pd.DataFrame): Input dataframe to process
        model (List[Any]): Model object containing category mappings
        categorical_cols (Optional[List[str]]): List of categorical column names

    Returns:
        pd.DataFrame: Dataframe with processed categorical columns

    Example:
        >>> df_cat = pre_process_cat(df, model, ['category', 'department'])
    """

    cat_cols: List[str] = categorical_cols or []
    
    for col_name in cat_cols:
        if col_name in df.columns and col_name in model[0]:

            cat_type: CategoricalDtype = CategoricalDtype(categories=model[0].get(col_name), ordered=False)
            df[col_name] = df[col_name].astype(cat_type, copy=False)
            df[col_name] = df[col_name].fillna('Unk')
    
    return df


class ScoringService(object):
    """Service class for model loading and inference.
    
    Handles model loading, preprocessing, and prediction for both
    binary classification and regression tasks.

    Attributes:
        model (Optional[List[Any]]): Loaded model and category mappings
        column_config (Optional[Dict[str, List[str]]]): Column type configuration
    """
    model: Optional[List[Any]] = None 
    column_config: Optional[Dict[str, List[str]]] = None 

    @classmethod
    def get_model(cls) -> Optional[List[Any]]:
        """Loads and initializes the model for inference.

        Handles:
        1. Loading model from tar.gz archive
        2. Loading model weights (.cbm or .dump file)
        3. Loading categorical feature mappings
        4. Loading column configuration

        Returns:
            Optional[List[Any]]: List containing:
                - [0]: Dictionary of categorical feature mappings
                - [1]: Loaded CatBoost model
                Returns None if loading fails

        Raises:
            Exception: Various exceptions during file operations or model loading
        """
        if cls.model is None:
            logger.info('Loading model...')
            model_file: CatBoostClassifier = CatBoostClassifier()
            
            tar_path: str = os.path.join(model_path, 'model.tar.gz')
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

            model_files: List[str] = [f for f in os.listdir(model_path) if f.endswith(('.cbm', '.dump'))]
            if not model_files:
                logger.error(f'No model files (.cbm or .dump) found in {model_path}')
                return None
                    
            model_file_path: str = os.path.join(model_path, model_files[0])
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

            obj_col_categories: Dict[str, List[str]] = {}
            try:
                cat_path: str = os.path.join(model_path, 'obj_col_categories.pkl')
                if os.path.exists(cat_path):
                    with open(cat_path, 'rb') as inp:
                        obj_col_categories = pickle.load(inp)
                        logger.info('Categories loaded successfully')
                else:
                    logger.warning('No categories file found at expected path')
            except Exception as e:
                logger.error(f'Error loading categories: {str(e)}')
            
            try:
                config_path: str = os.path.join(model_path, 'column_config.json')
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
    def predict(cls, input_df: pd.DataFrame) -> pd.DataFrame:
        """Performs model inference on input data.

        Handles both matrix and structured inputs:
        - Matrix input: Direct numerical features
        - Structured input: Mixed numerical, categorical and text features

        Args:
            input_df (pd.DataFrame): Input data for prediction

        Returns:
            pd.DataFrame: Predictions dataframe containing:
                - ID column if present in input
                - Target column if present in input 
                - 'score' column with model predictions

        Example:
            >>> predictions = ScoringService.predict(input_data)
            >>> predictions.head()
               id  score
            0   1  0.823
            1   2  0.654
        """
        model: Optional[List[Any]] = cls.get_model()
        logger.info('Running inference...')
        
        is_matrix_input: bool = 'matrix_input' in input_df.columns
        
        if cls.column_config is not None:
            numerical_cols: List[str] = cls.column_config.get('numerical_cols', [])
            categorical_cols: List[str] = cls.column_config.get('categorical_cols', [])
            text_cols: List[str] = cls.column_config.get('text_cols', [])
        else:
            numerical_cols = [col for col in input_df.columns if pd.api.types.is_numeric_dtype(input_df[col]) and col != 'matrix_input']
            categorical_cols = []
            text_cols = []
        
        if is_matrix_input:
            available_num_cols: List[str] = [col for col in numerical_cols if col in input_df.columns] or [col for col in input_df.columns if col != 'matrix_input']
            input_pool: CatboostPool = CatboostPool(
                data=input_df[available_num_cols],
                label=None,
                feature_names=available_num_cols
            )
        else:
            input_processed: pd.DataFrame = pre_process(input_df, numerical_cols, categorical_cols, text_cols)
            input_processed = pre_process_cat(input_processed, model, categorical_cols)
            
            available_num_cols = [col for col in numerical_cols if col in input_processed.columns]
            available_cat_cols: List[str] = [col for col in categorical_cols if col in input_processed.columns]
            available_text_cols: List[str] = [col for col in text_cols if col in input_processed.columns]
            
            feature_cols: List[str] = available_num_cols + available_cat_cols + available_text_cols
            
            if not feature_cols:
                feature_cols = list(input_processed.columns)
            
            input_pool = ret_pool_obj(
                input_processed[feature_cols],
                text_features=available_text_cols,
                cat_features=available_cat_cols
            )
        
        prob: np.ndarray = model[1].predict_proba(input_pool)[:, 1]
        logger.info(f'Completed inference on {prob.shape[0]} records')
        
        result_df: pd.DataFrame = pd.DataFrame()
        
        id_col: Optional[str] = None
        for possible_id in ['id', 'review_id', 'customer_id', 'product_id']:
            if possible_id in input_df.columns:
                id_col = possible_id
                result_df[id_col] = input_df[id_col]
                break
            
        if id_col is None:
            result_df['id'] = range(len(prob))
            id_col = 'id'
        
        if 'target' in input_df.columns:
            result_df['target'] = input_df['target']
        
        result_df['score'] = prob
        
        return result_df


app: flask.Flask = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping() -> flask.Response:
    """Health check endpoint for the inference service.

    Verifies the service is healthy by attempting to load the model.
    Returns 200 if successful, 404 if model loading fails.

    Returns:
        flask.Response: HTTP response with status code indicating health
    """
    health: bool = ScoringService.get_model() is not None 

    status: int = 200 if health else 404
    return flask.Response(
        response='\n',
        status=status,
        mimetype='application/json'
        )

@app.route('/invocations', methods=['POST'])
def transformation() -> flask.Response:
    """Inference endpoint that handles prediction requests.

    Accepts CSV input data in two formats:
    1. Numerical matrix without headers
    2. Structured data with/without headers

    The endpoint:
    1. Parses and validates input CSV
    2. Preprocesses the data
    3. Runs model inference
    4. Returns predictions as tab-separated values

    Returns:
        flask.Response: Prediction results as CSV or error message

    Raises:
        Exception: For CSV parsing or prediction errors
    """
    data: Optional[Union[str, pd.DataFrame]] = None
    start: float = time.time()
    
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        first_line: str = data.split('\n')[0].strip()
        
        logger.info(f'Input data first line sample: {first_line[:100]}...')
        
        first_line = data.split('\n')[0].strip()
        
        is_numeric_matrix: bool = all(
            val.strip().replace('.', '').replace('-', '').isdigit() 
            for val in first_line.split(',') if val.strip()
        )
        
        if is_numeric_matrix:
            logger.info('Detected numerical matrix input without headers')
            try:

                sep: str = '\t' if '\t' in first_line else ','
                
                matrix_df: pd.DataFrame = pd.read_csv(io.StringIO(data), header=None, sep=sep)
                
                matrix_df.columns = [f'col_{i+1}' for i in range(len(matrix_df.columns))]
                
                matrix_df['matrix_input'] = True
                
                data = matrix_df
                logger.info(
                    f'Matrix processed successfully. Shape: {data.shape}, Columns: {matrix_df.columns}')
            except Exception as e:
                logger.error(f"Error processing matrix: {str(e)}")
                is_numeric_matrix = False
        
        if not is_numeric_matrix:
            first_char: str = first_line.split(',')[0].strip()[0] if first_line else ''
            has_headers: bool = not first_char.isdigit()
            
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

    predictions: pd.DataFrame = ScoringService.predict(data)
    logger.debug(f'Shape of predictions: {predictions.shape}')
    
    out: io.StringIO = io.StringIO()
    predictions.to_csv(out, header=False, index=False, sep='\t', 
                      quotechar='"', escapechar='\\', quoting=csv.QUOTE_NONE)
    result: str = out.getvalue()
    end: float = time.time()

    logger.info(f'Time to execute: {end - start:.3f} seconds')
    return flask.Response(response=result, status=200, mimetype='text/csv')