
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.model.training import fit_model
from src.config import Config

class TestMLflowLogging(unittest.TestCase):
    @patch('src.model.training.mlflow')
    @patch('src.model.training.Config')
    def test_fit_model_logs_metrics(self, mock_config, mock_mlflow):
        # Setup mock config
        mock_config.return_value.device = 'cpu'
        
        # Setup dummy model and data
        model = nn.Linear(10, 1)
        X = torch.randn(10, 10)
        y = torch.randn(10, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2)
        
        # Run training
        fit_model(model, loader, loader, epochs=2, lr=0.01)
        
        # Verify mlflow.log_metric was called
        # We expect calls for train_loss, val_loss, and learning_rate for each epoch (2 epochs)
        # Total calls should be 3 metrics * 2 epochs = 6 calls
        self.assertTrue(mock_mlflow.log_metric.called)
        
        # Check specific calls
        calls = mock_mlflow.log_metric.call_args_list
        metric_names = [c[0][0] for c in calls]
        self.assertIn('train_loss', metric_names)
        self.assertIn('val_loss', metric_names)
        self.assertIn('learning_rate', metric_names)
        
        print("✅ MLflow logging verification passed!")

if __name__ == '__main__':
    unittest.main()
