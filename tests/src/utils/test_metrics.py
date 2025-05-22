import pytest
import numpy as np
import pandas as pd
import os
import shutil
from matplotlib import pyplot as plt
from unittest.mock import patch
from typing import Tuple, Dict, Any
import PIL.Image as Image

from utils.metrics import MetricsCalculator


@pytest.fixture
def sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create sample predictions and labels for testing.
    
    Returns:
        Tuple containing labels and predictions arrays.
    """
    np.random.seed(42)  
    labels = np.array([0, 1] * 50)
    
    predictions = np.random.beta(2, 2, size=100)
    predictions = np.where(labels == 1, 
                          predictions * 0.7 + 0.3,  # Higher values for positive class
                          predictions * 0.7)        # Lower values for negative class
    
    predictions = np.clip(predictions, 0, 1)
    
    return labels, predictions


@pytest.fixture
def metrics_calculator() -> MetricsCalculator:
    """Create a metrics calculator instance with guaranteed image directory.
    
    Returns:
        MetricsCalculator instance with properly configured image directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, 'test_output', 'images')
    
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir, exist_ok=True)
    
    calculator = MetricsCalculator()
    calculator.images_dir = os.path.join(image_dir, '')  # Ensure trailing separator
    
    yield calculator
    
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)


class TestMetricsCalculator:
    """Test suite for MetricsCalculator functionality."""
    
    def test_accepts_proper_initialization(self, metrics_calculator: MetricsCalculator) -> None:
        """Test that metrics calculator initializes with valid image directory."""
        assert os.path.exists(metrics_calculator.images_dir.rstrip('/'))
        assert hasattr(metrics_calculator, 'images_dir')
    
    def test_accepts_valid_threshold_cert_analysis(self, metrics_calculator: MetricsCalculator, 
                                            sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test certification threshold analysis with valid input data."""
        labels, predictions = sample_data
        result = metrics_calculator.threshold_cert(labels, predictions)
        
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['Threshold', 'True Positive Rate', 'False Discovery Rate', 'False Positive Rate']
        assert list(result.columns) == expected_columns
        
        assert list(result['Threshold']) == sorted(result['Threshold'])
        
        assert result['Threshold'].min() >= 0
        assert result['Threshold'].max() <= 1.05  # Implementation goes to 1.05
        assert result['True Positive Rate'].min() >= 0
        assert result['True Positive Rate'].max() <= 100
    
    def test_accepts_valid_threshold_decert_analysis(self, metrics_calculator: MetricsCalculator,
                                              sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test decertification threshold analysis with valid input data."""
        labels, predictions = sample_data
        result = metrics_calculator.threshold_decert(labels, predictions)
        
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['Threshold', 'True Positive Rate', 'False Discovery Rate', 'False Positive Rate']
        assert list(result.columns) == expected_columns
        
        assert list(result['Threshold']) == sorted(result['Threshold'])
        
        assert result['Threshold'].min() >= 0
        assert result['Threshold'].max() <= 1.0
        assert result['True Positive Rate'].min() >= 0
        assert result['True Positive Rate'].max() <= 100
    
    @patch('matplotlib.pyplot.show')
    def test_accepts_complete_metrics_calculation(self, mock_show, metrics_calculator: MetricsCalculator,
                                           sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test overall metrics calculation returns expected structure."""
        labels, predictions = sample_data
        
        metrics = metrics_calculator.run_metrics(labels, predictions)
        
        expected_keys = [
            'confusion_matrix', 'recall_score', 'precision_score', 'F1_Score',
            'accuracy_score', 'auc', 'roc_auc_score', 'threshold_decert', 
            'threshold_cert', 'pass_rate'
        ]
        
        assert all(key in metrics for key in expected_keys)
        
        cm_keys = ['TN', 'FP', 'FN', 'TP']
        assert all(key in metrics['confusion_matrix'] for key in cm_keys)
        
        assert 0 <= float(metrics['recall_score']) <= 1
        assert 0 <= float(metrics['precision_score']) <= 1
        assert 0 <= float(metrics['F1_Score']) <= 1
        assert 0 <= float(metrics['accuracy_score']) <= 1
        assert 0 <= float(metrics['auc']) <= 1
        assert 0 <= float(metrics['roc_auc_score']) <= 1
    
    @patch('matplotlib.pyplot.show')
    def test_accepts_perfect_prediction_edge_case(self, mock_show, metrics_calculator: MetricsCalculator) -> None:
        """Test metrics calculation with perfect predictions."""
        perfect_labels = np.array([0, 1] * 10)
        perfect_predictions = perfect_labels.astype(float)
        
        metrics = metrics_calculator.run_metrics(perfect_labels, perfect_predictions)
        
        assert float(metrics['accuracy_score']) == 1.0
        assert float(metrics['precision_score']) == 1.0
        assert float(metrics['recall_score']) == 1.0
    
    @patch('matplotlib.pyplot.show')  
    def test_accepts_worst_case_prediction_edge_case(self, mock_show, metrics_calculator: MetricsCalculator) -> None:
        """Test metrics calculation with completely incorrect predictions."""
        incorrect_labels = np.array([0, 1] * 10)
        incorrect_predictions = np.array([1, 0] * 10).astype(float)
        
        metrics = metrics_calculator.run_metrics(incorrect_labels, incorrect_predictions)
        
        assert float(metrics['accuracy_score']) == 0.0
    
    @patch('matplotlib.pyplot.show')
    def test_accepts_random_classifier_edge_case(self, mock_show, metrics_calculator: MetricsCalculator) -> None:
        """Test metrics calculation with random classifier behavior."""
        # Random classifier (all same predictions)
        random_labels = np.array([0, 1] * 10) 
        random_predictions = np.array([0.5] * 20)
        
        metrics = metrics_calculator.run_metrics(random_labels, random_predictions)
        
        assert abs(float(metrics['roc_auc_score']) - 0.5) < 0.01
    
    def test_accepts_zero_predictions_without_error(self, metrics_calculator: MetricsCalculator) -> None:
        """Test handling of zero predictions without division errors."""
        labels = np.array([0, 1] * 10)
        zero_predictions = np.zeros(20)
        
        cert_result = metrics_calculator.threshold_cert(labels, zero_predictions)
        decert_result = metrics_calculator.threshold_decert(labels, zero_predictions)
        
        assert isinstance(cert_result, pd.DataFrame)
        assert isinstance(decert_result, pd.DataFrame)
        assert len(cert_result) > 0
        assert len(decert_result) > 0


class TestImageSaving:
    """Test suite for plot generation and saving functionality."""
    
    @patch('matplotlib.pyplot.show')
    def test_succeeds_saving_roc_curve_image(self, mock_show, metrics_calculator: MetricsCalculator,
                                       sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test that ROC curve image is saved correctly."""
        labels, predictions = sample_data
        
        metrics_calculator.run_metrics(labels, predictions)
        
        roc_path = os.path.join(metrics_calculator.images_dir, 'roc_curve.png')
        assert os.path.exists(roc_path), f"ROC curve not found at {roc_path}"
        
        img = Image.open(roc_path)
        assert img.size[0] > 0
        assert img.size[1] > 0
        
        assert os.path.getsize(roc_path) > 5000  
    
    @patch('matplotlib.pyplot.show')
    def test_succeeds_saving_confusion_matrix_image(self, mock_show, metrics_calculator: MetricsCalculator,
                                             sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test that confusion matrix image is saved correctly."""
        labels, predictions = sample_data
        
        metrics_calculator.run_metrics(labels, predictions)
        
        cm_path = os.path.join(metrics_calculator.images_dir, 'confusion_matrix.png')
        assert os.path.exists(cm_path), f"Confusion matrix not found at {cm_path}"
        
        img = Image.open(cm_path)
        assert img.size[0] > 0
        assert img.size[1] > 0
        
        assert os.path.getsize(cm_path) > 5000  # At least 5KB
    
    @patch('matplotlib.pyplot.show')
    def test_succeeds_custom_image_directory_creation(self, mock_show, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test saving plots to custom directory locations."""
        labels, predictions = sample_data
        
        temp_dir = os.path.join(os.path.dirname(__file__), 'custom_plots')
        
        try:
            calculator = MetricsCalculator()
            calculator.images_dir = os.path.join(temp_dir, '')
            
            os.makedirs(temp_dir, exist_ok=True)
            
            calculator.run_metrics(labels, predictions)
            
            assert os.path.exists(os.path.join(temp_dir, 'roc_curve.png'))
            assert os.path.exists(os.path.join(temp_dir, 'confusion_matrix.png'))
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    @patch('matplotlib.pyplot.show')
    def test_accepts_generated_plot_dimensions(self, mock_show, metrics_calculator: MetricsCalculator,
                                        sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test that generated plots have reasonable dimensions."""
        labels, predictions = sample_data
        
        metrics_calculator.run_metrics(labels, predictions)
        
        roc_path = os.path.join(metrics_calculator.images_dir, 'roc_curve.png')
        cm_path = os.path.join(metrics_calculator.images_dir, 'confusion_matrix.png')
        
        roc_img = Image.open(roc_path)
        cm_img = Image.open(cm_path)
        
        assert roc_img.size[0] > 400 and roc_img.size[1] > 300
        assert cm_img.size[0] > 400 and cm_img.size[1] > 300


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_raises_value_error_with_empty_arrays(self, metrics_calculator: MetricsCalculator) -> None:
        """Test that empty input arrays raise appropriate errors."""
        empty_labels = np.array([])
        empty_predictions = np.array([])
        
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            metrics_calculator.run_metrics(empty_labels, empty_predictions)
    
    def test_raises_value_error_with_mismatched_lengths(self, metrics_calculator: MetricsCalculator) -> None:
        """Test that mismatched array lengths raise appropriate errors."""
        labels = np.array([0, 1, 0])
        predictions = np.array([0.1, 0.9])  # Different length
        
        with pytest.raises(ValueError, match="Labels and predictions must have the same length"):
            metrics_calculator.run_metrics(labels, predictions)
    
    def test_accepts_single_class_labels(self, metrics_calculator: MetricsCalculator) -> None:
        """Test handling of labels with only one class."""
        single_class_labels = np.array([0] * 10)
        single_class_predictions = np.random.random(10)
        
        cert_result = metrics_calculator.threshold_cert(single_class_labels, single_class_predictions)
        decert_result = metrics_calculator.threshold_decert(single_class_labels, single_class_predictions)
        
        assert isinstance(cert_result, pd.DataFrame)
        assert isinstance(decert_result, pd.DataFrame)
    
    def test_accepts_extreme_prediction_values(self, metrics_calculator: MetricsCalculator) -> None:
        """Test handling of extreme prediction values."""
        labels = np.array([0, 1] * 10)
        
        extreme_predictions = np.array([0.0, 1.0] * 10)
        
        cert_result = metrics_calculator.threshold_cert(labels, extreme_predictions)
        decert_result = metrics_calculator.threshold_decert(labels, extreme_predictions)
        
        assert isinstance(cert_result, pd.DataFrame)
        assert isinstance(decert_result, pd.DataFrame)
        
        assert not cert_result.isnull().any().any()
        assert not decert_result.isnull().any().any()
    
    @patch('matplotlib.pyplot.show')
    @patch('sklearn.metrics.confusion_matrix')
    def test_accepts_metrics_calculation_with_sklearn_errors(self, mock_confusion_matrix, mock_show, 
                                                      metrics_calculator: MetricsCalculator) -> None:
        """Test graceful handling of sklearn calculation errors."""
        mock_confusion_matrix.side_effect = Exception("Sklearn calculation failed")
        
        labels = np.array([0, 1] * 10)
        predictions = np.array([0.3, 0.7] * 10)
        
        metrics = metrics_calculator.run_metrics(labels, predictions)
        
        # Should return default values when sklearn fails
        assert float(metrics['recall_score']) == 0.0
        assert float(metrics['precision_score']) == 0.0
        assert float(metrics['F1_Score']) == 0.0
        assert float(metrics['accuracy_score']) == 0.0