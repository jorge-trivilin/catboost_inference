import numpy as np
import os
import matplotlib
import pandas as pd
from loguru import logger
from typing import Dict, Any, Tuple

matplotlib.use('agg', force=True)
from matplotlib import pyplot as plt


class MetricsCalculator:

    def __init__(self) -> None:
        """Initialize MetricsCalculator with image directory setup.
        
        Creates an images directory relative to the current script location
        for saving generated plots and visualizations.
        """
        script_dir = os.path.dirname(__file__)
        self.images_dir = os.path.join(script_dir, 'images/')
        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir)

    def threshold_cert(self, labels: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
        """Analyze certification thresholds for binary classification.
        
        Args:
            labels: Binary ground truth labels (0 or 1)s
            predictions: Predicted probabilities (0 to 1)
            
        Returns:
            DataFrame with threshold analysis including TPR, FDR, and FPR
        """
        result_cert = []
        for i in [x / 100.0 for x in range(0, 110, 5)]:
            predicted_0 = np.where(predictions <= i, 1, 0).sum()

            actual_0 = np.where(labels == 0, 1, 0).sum()
            actual_1 = np.where(labels == 1, 1, 0).sum()

            pred_0_actual_0 = np.where((predictions <= i) & (labels == 0), 1, 0).sum()
            pred_0_actual_1 = np.where((predictions <= i) & (labels == 1), 1, 0).sum()

            recall = (pred_0_actual_0 * 100 / actual_0) if actual_0 > 0 else 0.0
            fdr = (1 - pred_0_actual_0 / predicted_0) * 100 if predicted_0 > 0 else 0.0
            fpr = (pred_0_actual_1 * 100 / actual_1) if actual_1 > 0 else 0.0

            res = [i, recall, fdr, fpr]
            result_cert.append(res)

        result_cert = pd.DataFrame(result_cert)
        result_cert.columns = ['Threshold', 'True Positive Rate', 'False Discovery Rate', 'False Positive Rate']
        return result_cert

    def threshold_decert(self, labels: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
        """Analyze decertification thresholds for binary classification.
        
        Args:
            labels: Binary ground truth labels (0 or 1)
            predictions: Predicted probabilities (0 to 1)
            
        Returns:
            DataFrame with threshold analysis including TPR, FDR, and FPR
        """
        result_decert = []
        for i in [x / 100.0 for x in range(0, 101, 5)]:
            predicted_1 = np.where(predictions > i, 1, 0).sum()

            actual_1 = np.where(labels == 1, 1, 0).sum()
            actual_0 = np.where(labels == 0, 1, 0).sum()

            pred_1_actual_1 = np.where((predictions > i) & (labels == 1), 1, 0).sum()
            pred_1_actual_0 = np.where((predictions > i) & (labels == 0), 1, 0).sum()

            recall = (pred_1_actual_1 * 100 / actual_1) if actual_1 > 0 else 0.0
            fdr = (1 - pred_1_actual_1 / predicted_1) * 100 if predicted_1 > 0 else 0.0
            fpr = (pred_1_actual_0 / actual_0) * 100 if actual_0 > 0 else 0.0

            res = [i, recall, fdr, fpr]
            result_decert.append(res)
            
        result_decert = pd.DataFrame(result_decert)
        result_decert.columns = ['Threshold', 'True Positive Rate', 'False Discovery Rate', 'False Positive Rate']
        return result_decert

    def run_metrics(self, labels: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive metrics for binary classification predictions.
        
        Args:
            labels: Binary ground truth labels (0 or 1)
            predictions: Predicted probabilities (0 to 1)
            
        Returns:
            Dictionary containing all calculated metrics and analysis results
            
        Raises:
            ValueError: If input arrays have mismatched lengths or invalid values
        """
        from sklearn import metrics

        if len(labels) != len(predictions):
            raise ValueError("Labels and predictions must have the same length")
        
        if len(labels) == 0:
            raise ValueError("Input arrays cannot be empty")

        result = []
        fpr, tpr, _ = metrics.roc_curve(labels, predictions)
        roc_auc_score = metrics.roc_auc_score(labels, predictions)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f"data 1, auc={roc_auc_score:.4f}")
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.legend(loc=4)
        plt.savefig(self.images_dir + 'roc_curve.png')
        plt.close() 

        prediction_res = []

        logger.info(f"ROC AUC Score: {roc_auc_score}")

        prediction_res.append(roc_auc_score)
        prediction_res.append(roc_auc_score)
        
        lb = [x / 100.0 for x in range(0, 101, 5)]
        ub = [x / 100.0 for x in range(5, 106, 5)]

        predict_flip = predictions
        target_flip = 1 - labels
        
        pass_fc = []
        for i in range(len(lb)):
            total_count = np.where((predict_flip.astype(float) >= lb[i]) & 
                                (predict_flip.astype(float) < ub[i]), 1, 0).sum()
            total_pass = np.where((predict_flip.astype(float) >= lb[i]) & 
                               (predict_flip.astype(float) < ub[i]), target_flip, 0).sum()
            
            pass_rate = np.round(total_pass/total_count, 4) if total_count > 0 else 0.0
            pass_fc.append([lb[i], ub[i], total_count, total_pass, pass_rate])

        pass_fc = pd.DataFrame(pass_fc)
        pass_fc.columns = ['LB', 'UB', 'Count', 'Pass', 'Pass_rate']

        precision, recall, thresholds = metrics.precision_recall_curve(labels, predictions)
        auc = metrics.auc(recall, precision)
        logger.info(f"PR-AUC: {auc}")

        decert = str(self.threshold_decert(labels, predictions).to_dict())
        cert = str(self.threshold_cert(labels, predictions).to_dict())

        rounded_predictions = np.round(predictions.copy())

        try:
            confusion_matrix = metrics.confusion_matrix(labels, rounded_predictions)
            recall_score = metrics.recall_score(labels, rounded_predictions, pos_label=1, zero_division=0)
            precision_score = metrics.precision_score(labels, rounded_predictions, pos_label=1, zero_division=0)
            f1_score = metrics.f1_score(labels, rounded_predictions, pos_label=1, zero_division=0)
            accuracy_score = metrics.accuracy_score(labels, rounded_predictions)
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            confusion_matrix = np.zeros((2, 2))
            recall_score = 0.0
            precision_score = 0.0
            f1_score = 0.0
            accuracy_score = 0.0

        # Plot confusion matrix
        plt.figure()
        ax = plt.subplot(111)
        cax = ax.matshow(confusion_matrix)
        plt.title('Confusion matrix of the classifier')
        fig = plt.gcf()
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.images_dir + 'confusion_matrix.png')
        plt.close()
        
        logger.info(f"Confusion matrix: {confusion_matrix}")
        logger.info(f"Recall Score: {recall_score}")
        logger.info(f"Precision Score: {precision_score}")
        logger.info(f"F1 Score: {f1_score}")
        logger.info(f"Accuracy score: {accuracy_score}")

        prediction_res.extend([recall_score, precision_score, f1_score, accuracy_score])
        result.append(prediction_res)
        logger.info(f'The result of prediction metrics is {result}')

        metrics_dict = {
            'confusion_matrix': {
                'TN': str(confusion_matrix[0, 0]),
                'FP': str(confusion_matrix[0, 1]),
                'FN': str(confusion_matrix[1, 0]),
                'TP': str(confusion_matrix[1, 1])
            },
            'recall_score': str(recall_score),
            'precision_score': str(precision_score),
            'F1_Score': str(f1_score),
            'accuracy_score': str(accuracy_score),
            'auc': str(auc),
            'roc_auc_score': str(roc_auc_score),
            'threshold_decert': decert,
            'threshold_cert': cert,
            'pass_rate': str(pass_fc.to_dict())
        }

        return metrics_dict