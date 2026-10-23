import os
import mlflow

# Use a local directory for Render, or read from .env for local development
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment("smartfind_embeddings_and_matching")

def log_inference_event(
    event_type: str, 
    description: str, 
    inference_time_ms: float, 
    top_matches_count: int = 0,
    top_score: float = 0.0
):
    """Logs embedding inference metrics and parameters into MLflow."""
    with mlflow.start_run(run_name=f"{event_type}_report_run"):
        # Log Model Meta Parameters
        mlflow.log_param("model_architecture", "clip-ViT-B-32")
        mlflow.log_param("event_type", event_type)
        mlflow.log_param("description_length", len(description))
        
        # Log Key Metrics
        mlflow.log_metric("embedding_inference_time_ms", inference_time_ms)
        mlflow.log_metric("top_matches_returned", top_matches_count)
        mlflow.log_metric("top_similarity_score", top_score)