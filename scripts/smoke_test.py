import mlflow

mlflow.set_tracking_uri("http://localhost:5001")   # point the client at your server
mlflow.set_experiment("smoke-test")                 # created if it doesn't exist

with mlflow.start_run(run_name="first-run"):
    mlflow.log_param("lr", 0.001)                   # a setting
    for step in range(5):
        mlflow.log_metric("loss", 1.0 / (step + 1), step=step)  # a metric over time
    with open("hello.txt", "w") as f:
        f.write("artifact test")
    mlflow.log_artifact("hello.txt")                # a file → goes to the artifacts volume