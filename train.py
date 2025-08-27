import sys, time, json, os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import mlflow, mlflow.spark
from mlflow.tracking import MlflowClient

# Params for reproducibility
params = {
    "lr": {"maxIter": 50, "regParam": 0.0, "elasticNetParam": 0.0},
    "spark": {"sql_shuffle_partitions": 16},
    "seed": 42,
    "tvs": {"trainRatio": 0.8}
}
if Path("params.yaml").exists():
    import yaml
    with open("params.yaml") as f:
        loaded = yaml.safe_load(f)
        for k in loaded or {}:
            if isinstance(loaded[k], dict):
                params[k].update(loaded[k])
            else:
                params[k] = loaded[k]

input_path = sys.argv[1]  # data/processed_single
model_out = Path(sys.argv[2])  # models/titanic
metrics_out = Path(sys.argv[3])  # reports/metrics.json
cm_png_out = Path(sys.argv[4])  # reports/confusion_matrix.png
fi_png_out = Path(sys.argv[5])  # reports/feature_importance.png
model_out.mkdir(parents=True, exist_ok=True)
metrics_out.parent.mkdir(parents=True, exist_ok=True)
cm_png_out.parent.mkdir(parents=True, exist_ok=True)
fi_png_out.parent.mkdir(parents=True, exist_ok=True)

spark = SparkSession.builder \
    .appName("TitanicTrain") \
    .config("spark.sql.shuffle.partitions", params["spark"]["sql_shuffle_partitions"]) \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load data
df = spark.read.csv(input_path, header=True, inferSchema=True)
df = df.withColumnRenamed("Survived", "label")
feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_idx", "Embarked_idx"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df).select("label", "features")

# Split data
train, test = df.randomSplit([params["tvs"]["trainRatio"], 1.0 - params["tvs"]["trainRatio"]], seed=params["seed"])

# Define model and grid
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=params["lr"]["maxIter"])
grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.0, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [50, 100]) \
    .build()

# Evaluator
bin_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# TrainValidationSplit
tvs = TrainValidationSplit(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=bin_eval,
    trainRatio=params["tvs"]["trainRatio"],
    seed=params["seed"]
)

# Train
t0 = time.perf_counter()
tvs_model = tvs.fit(train)
train_time_s = round(time.perf_counter() - t0, 3)

# Evaluate
best = tvs_model.bestModel
pred = best.transform(test)
auc = float(bin_eval.evaluate(pred))
f1 = float(MulticlassClassificationEvaluator(labelCol="label", metricName="f1").evaluate(pred))
acc = float(MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy").evaluate(pred))

# Confusion matrix
pdf = pred.select("label", "prediction").toPandas()
cm = confusion_matrix(pdf["label"], pdf["prediction"], labels=[0, 1])
plt.figure()
plt.imshow(cm, interpolation="nearest", labels=[0, 1])
plt.imshow(cm, interpolation="nearest")  # remove labels=...
plt.title("Confusion Matrix (Titanic)")
plt.xticks([0, 1], [0, 1]); plt.yticks([0, 1], [0, 1])
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig(cm_png_out, dpi=120)
plt.close()

# Feature importance (coefficients)
coef = best.coefficients.toArray()
plt.figure()
plt.barh(feature_cols, coef)
plt.title("LR Coefficients (Titanic)")
plt.tight_layout()
plt.savefig(fi_png_out, dpi=120)
plt.close()

# Save model and metrics
best.write().overwrite().save(str(model_out))
metrics = {
    "auc": round(auc, 4),
    "f1": round(f1, 4),
    "accuracy": round(acc, 4),
    "train_time_s": train_time_s,
    "best_params": {
        "regParam": best._java_obj.getRegParam(),
        "elasticNetParam": best._java_obj.getElasticNetParam(),
        "maxIter": best._java_obj.getMaxIter()
    },
    "spark_shuffle_partitions": params["spark"]["sql_shuffle_partitions"],
    "seed": params["seed"]
}
metrics_out.write_text(json.dumps(metrics, indent=2))

# MLflow logging
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("titanic_spark")
with mlflow.start_run():
    mlflow.log_params({
        "regParam": metrics["best_params"]["regParam"],
        "elasticNetParam": metrics["best_params"]["elasticNetParam"],
        "maxIter": metrics["best_params"]["maxIter"],
        "spark_shuffle_partitions": params["spark"]["sql_shuffle_partitions"],
        "seed": params["seed"]
    })
    mlflow.log_metrics({"auc": auc, "f1": f1, "accuracy": acc, "train_time_s": train_time_s})
    mlflow.log_artifact(str(cm_png_out))
    mlflow.log_artifact(str(fi_png_out))
    mlflow.spark.log_model(best, artifact_path="model")

    # Register and transition to Staging
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    reg = mlflow.register_model(f"runs:/{run_id}/model", "titanic_lr")
    client.transition_model_version_stage(name="titanic_lr", version=reg.version, stage="Staging", archive_existing_versions=False)

spark.stop()
print("Done. Metrics:", json.dumps(metrics))
