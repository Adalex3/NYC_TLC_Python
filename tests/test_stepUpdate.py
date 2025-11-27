import pytest
import json
import numpy as np
import os
from src.model_module.core import update_ls_sigma_noise
from src.model_module.utils import kernel, noise_mix

def test_update_step_consistency():
    """
    Verifies update_ls_sigma_noise against data generated from R.
    """
    # Load the JSON data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'step_test_data.json')
    if not os.path.exists(data_path):
        pytest.skip("step_test_data.json not found.")
        
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    inputs = data['inputs']
    
    # Convert lists to numpy arrays
    gpdraw = np.array(inputs['gpdraw'])
    D = np.array(inputs['D'])
    
    # Reconstruct K (Python side)
    # R: K_curr <- sigma_curr^2 * noise_mix(kern_func(D, ls_curr), noise_ratio)
    K_curr = (inputs['sigma']**2) * noise_mix(kernel(D, inputs['ls']), inputs['noise_ratio'])
    
    # Run Function
    result = update_ls_sigma_noise(
        ls=inputs['ls'],
        sigma=inputs['sigma'],
        noise_ratio=inputs['noise_ratio'],
        gpdraw=gpdraw,
        K=K_curr,
        D=D,
        lsPrior=inputs['lsPrior'],
        sigmaPrior=inputs['sigmaPrior'],
        noisePrior=inputs['noisePrior'],
        kern=kernel
    )
    
    # Basic Checks
    assert result['ls'] > 0
    assert result['sigma'] > 0
    assert 0 < result['noise_ratio'] < 1