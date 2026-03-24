import os
import sys
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

THRESHOLD = 0.85

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

print(f"Run ID:    {run_id}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Threshold: {THRESHOLD}")

if accuracy is None:
    print("ERROR: No accuracy metric found!")
    sys.exit(1)

if accuracy < THRESHOLD:
    print(f"FAILED: {accuracy:.4f} is below {THRESHOLD}")
    sys.exit(1)

print(f"PASSED: {accuracy:.4f} meets threshold")
