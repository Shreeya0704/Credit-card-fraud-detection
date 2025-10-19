import os
import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def dummy_model_and_data():
    """
    Creates and saves a dummy model pipeline and provides sample data.
    This runs only once per test module.
    """
    # Create dummy data
    X = pd.DataFrame({
        "Time": [0, 1, 2, 3, 4],
        "Amount": [10, 20, 5, 50, 100],
        "V1": [0.1, -0.2, 0.3, -0.4, 0.5],
    })
    y = pd.Series([0, 0, 1, 0, 1])

    # Create a simple pipeline (preprocessor + model)
    # Using a subset of transformers for simplicity
    preprocessor = StandardScaler()
    model = LogisticRegression()
    pipeline = Pipeline([("preprocessing", preprocessor), ("model", model)])

    # Fit the pipeline
    pipeline.fit(X, y)

    # Save the dummy model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dummy_inference.joblib")
    joblib.dump(pipeline, model_path)

    # Sample inference data (one row)
    inference_data = pd.DataFrame([{
        "Time": 10, "Amount": 99.99, "V1": -1.5
    }])

    yield model_path, inference_data

    # Teardown: remove the dummy model file
    os.remove(model_path)


def test_inference_pipeline(dummy_model_and_data):
    """Test loading the model and making a prediction."""
    model_path, inference_data = dummy_model_and_data

    # Load the saved pipeline
    loaded_pipeline = joblib.load(model_path)
    assert loaded_pipeline is not None
    assert "preprocessing" in loaded_pipeline.named_steps
    assert "model" in loaded_pipeline.named_steps

    # Make a prediction
    try:
        prediction = loaded_pipeline.predict(inference_data)
        pred_proba = loaded_pipeline.predict_proba(inference_data)
    except Exception as e:
        pytest.fail(f"Model prediction failed with error: {e}")

    # Check output shapes and types
    assert prediction.shape == (1,)
    assert pred_proba.shape == (1, 2)
    assert isinstance(prediction[0], np.int64) or isinstance(prediction[0], np.int32)
    assert isinstance(pred_proba[0, 1], float)
