import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

input_path = sys.argv[1]
output_path = sys.argv[2]

spark = SparkSession.builder.appName("TitanicPreprocess").getOrCreate()
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Drop high-cardinality/ID-like columns
df = df.drop("PassengerId", "Name", "Ticket", "Cabin")

# Impute numerics
imputer = Imputer(inputCols=["Age", "Fare"], outputCols=["Age", "Fare"]).setStrategy("median")

# Index categoricals (create one indexer per column)
sex_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_idx", handleInvalid="keep")
emb_indexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_idx", handleInvalid="keep")

# Assemble (if you want a vector column; optional for CSV output below)
assembler = VectorAssembler(
    inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_idx", "Embarked_idx"],
    outputCol="features"
)

pipeline = Pipeline(stages=[imputer, sex_indexer, emb_indexer, assembler])
model = pipeline.fit(df)
dfp = model.transform(df)

# Choose what to persist; here we keep tabular columns (simple CSV)
cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_idx", "Embarked_idx"]
dfp.select(*cols).write.mode("overwrite").csv(output_path, header=True)
spark.stop()
