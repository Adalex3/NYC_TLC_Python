import numpy as np
from src.model_module import ZINB_GP, make_y_Vs_Vt

def test_run():
    print("Initializing dummy data...")
    # 1. Define dimensions
    n_loc = 10   # Number of spatial locations
    n_time = 4   # Number of time points
    N = n_loc * n_time
    p = 2        # Number of predictors (columns in X)
    
    # 2. Generate dummy Observation Matrix (Rows=Space, Cols=Time)
    obs_matrix = np.random.randint(0, 5, size=(n_loc, n_time))
    
    # 3. Process using the helper function
    data_struct = make_y_Vs_Vt(obs_matrix)
    y = data_struct['y']
    Vs = data_struct['Vs']
    Vt = data_struct['Vt']
    
    # 4. Generate Predictors (X)
    # X must match the total number of flattened observations N
    X = np.random.normal(size=(N, p))
    
    # 5. Generate Coords
    # CRITICAL: This must match n_loc (rows of obs_matrix) exactly.
    # The model internals will handle the "n-1" logic.
    coords = np.random.normal(size=(n_loc, 2)) 
    
    # 6. Generate Distance Matrices
    # The model expects dimensions matching 'coords'
    Ds = np.random.rand(n_loc, n_loc)
    Ds = (Ds + Ds.T) / 2 # Symmetric
    np.fill_diagonal(Ds, 0)
    
    Dt = np.random.rand(n_time, n_time)
    Dt = (Dt + Dt.T) / 2 # Symmetric
    np.fill_diagonal(Dt, 0)
    
    print(f"Data Shapes: X={X.shape}, y={y.shape}, Vs={Vs.shape}, coords={coords.shape}")
    print("Running ZINB_GP model (very short chain)...")
    
    try:
        results = ZINB_GP(
            X=X, y=y, coords=coords, 
            Vs=Vs, Vt=Vt, Ds=Ds, Dt=Dt, 
            nsim=5, burn=2, thin=1, 
            print_iter=1, print_progress=True
        )
        print("\nSuccess! Model ran.")
        print("Returned keys:", list(results.keys()))
        print("Beta shape:", results['Beta'].shape)
    except Exception as e:
        print("\nModel crashed:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_run()