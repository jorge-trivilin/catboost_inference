import pytest
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool as CatboostPool
from unittest.mock import patch, MagicMock
import os
import json
import pickle
import flask

from inference.predictor import (
    ret_pool_obj, 
    pre_process, 
    pre_process_cat, 
    ScoringService,
    ping,
    transformation
)

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'numeric': [1.0, 7.0, 3.0],
        'category': ['A', 'B', 'C'],
        'text': ['hello', 'YEP', 'world']
    })

@pytest.fixture
def mock_model():
    """Create a mock model with category mappings"""
    categories = {'category': ['A', 'B', 'C', 'Unk']}
    model = MagicMock(spec=CatBoostClassifier)
    
    model.is_fitted.return_value = True
    model.tree_count_ = 10
    
    model.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.1, 0.9]])
    
    return [categories, model]

def test_ret_pool_obj(sample_df):
    """Test CatBoost Pool object creation"""
    pool = ret_pool_obj(
        sample_df,
        text_features=['text'],
        cat_features=['category']
    )
    
    assert isinstance(pool, CatboostPool)
    assert pool.get_feature_names() == list(sample_df.columns)
    assert pool.get_text_feature_indices() == [3]  # 'text' column index
    assert pool.get_cat_feature_indices() == [2]   # 'category' column index

def test_pre_process(sample_df):
    """Test preprocessing of different column types"""
    processed_df = pre_process(
        sample_df,
        numerical_cols=['numeric'],
        categorical_cols=['category'],
        text_cols=['text']
    )
    
    assert processed_df['numeric'].fillna(-9999).equals(sample_df['numeric'].fillna(-9999))
    assert processed_df['category'].fillna('Unk').equals(sample_df['category'].fillna('Unk'))
    assert processed_df['text'].fillna('').equals(sample_df['text'].fillna(''))

def test_pre_process_cat(sample_df, mock_model):
    """Test categorical preprocessing"""
    processed_df = pre_process_cat(
        sample_df,
        mock_model,
        categorical_cols=['category']
    )
    
    assert processed_df['category'].dtype == 'category'
    assert list(processed_df['category'].cat.categories) == ['A', 'B', 'C', 'Unk']
    assert processed_df['category'].isna().sum() == 0

class TestScoringService:
    """Test cases for ScoringService class"""
    
    @pytest.fixture
    def mock_model_files(self, tmp_path):
        """Create mock model files"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        
        model_path = model_dir / "model.cbm"
        model_path.write_text("")
        
        cat_path = model_dir / "obj_col_categories.pkl"
        categories = {'category': ['A', 'B', 'C']}
        with open(cat_path, 'wb') as f:
            pickle.dump(categories, f)
            
        config_path = model_dir / "column_config.json"
        config = {
            'numerical_cols': ['numeric'],
            'categorical_cols': ['category'],
            'text_cols': ['text']
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        return model_dir

    @patch('inference.predictor.ScoringService.get_model')
    def test_predict_matrix_input(self, mock_get_model, mock_model, sample_df):
        """Test prediction with matrix input"""
        mock_get_model.return_value = mock_model
        sample_df['matrix_input'] = True
        
        predictions = ScoringService.predict(sample_df)
        
        assert 'score' in predictions.columns
        assert len(predictions) == len(sample_df)
        assert 'id' in predictions.columns

    @patch('inference.predictor.ScoringService.get_model')
    def test_predict_structured_input(self, mock_get_model, mock_model, sample_df):
        """Test prediction with structured input"""
        mock_get_model.return_value = mock_model
        
        predictions = ScoringService.predict(sample_df)
        
        assert 'score' in predictions.columns
        assert len(predictions) == len(sample_df)
        assert 'id' in predictions.columns

def test_ping():
    """Test health check endpoint"""
    with patch('inference.predictor.ScoringService.get_model') as mock_get_model:
        mock_get_model.return_value = True
        response = ping()
        assert response.status_code == 200
        
        mock_get_model.return_value = None
        response = ping()
        assert response.status_code == 404

def test_transformation():
    """Test inference endpoint"""
    with patch('inference.predictor.ScoringService.predict') as mock_predict:
        mock_predict.return_value = pd.DataFrame({
            'id': [1, 2],
            'score': [0.1, 0.9]
        })
        
        app: flask.Flask = flask.Flask(__name__)
        
        with app.test_client() as client:
            mock_request = MagicMock()
            mock_request.content_type = 'text/csv'
            mock_request.data = b'1,2,3\n4,5,6'
    
            with patch('flask.request', mock_request):
                with app.test_request_context(
                    data=b'1,2,3\n4,5,6',
                    content_type='text/csv'
                ):
                    response = transformation()
                    assert response.status_code == 200
                    assert response.mimetype == 'text/csv'
            
            mock_request = MagicMock()
            mock_request.content_type = 'application/json'
            
            with patch('flask.request', mock_request):
                with app.test_request_context(
                    content_type='application/json'
                ):
                    response = transformation()
                    assert response.status_code == 415