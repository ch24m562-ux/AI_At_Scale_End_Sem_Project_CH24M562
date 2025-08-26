import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.sql.functions import col

input_path = sys.argv[1]
output_path = sys.argv[2]

spark = SparkSession.builder.appName("TitanicPreprocess").getOrCreate()
df = spark.read.csv(input_path, header=True, inferSchema=True)
df = df.drop("PassengerId", "Name", "Ticket", "Cabin")

# Impute missing values
imputer = Imputer(inputCols=["Age", "Fare"], outputCols=["Age", "Fare"]).setStrategy("median")
df = imputer.fit(df).transform(df)

# Encode categorical
indexer = StringIndexer(inputCols=["Sex", "Embarked"], outputCols=["Sex_idx", "Embarked_idx"], handleInvalid="keep")
df = indexer.fit(df).transform(df)

# Assemble features (still needed for intermediate step)
assembler = VectorAssembler(inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_idx", "Embarked_idx"], outputCol="features")
df = assembler.transform(df)

# Convert features vector to individual columns
feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_idx", "Embarked_idx"]
processed_df = df.select("Survived", *feature_cols)

# Save processed data
processed_df.write.mode("overwrite").csv(output_path, header=True)
spark.stop()
