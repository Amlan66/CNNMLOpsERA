import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model import MNISTModel
from src.utils import get_test_accuracy

def test_model_parameters():
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_input_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    model = MNISTModel()
    latest_model = max([f for f in os.listdir('.') if f.startswith('model_')], key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    accuracy = get_test_accuracy(model)
    assert accuracy > 95.0, f"Model accuracy is {accuracy}%, should be > 95%" 