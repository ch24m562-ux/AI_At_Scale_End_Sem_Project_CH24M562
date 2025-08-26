import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, StringIndexer, StandardScaler, VectorAssembler

input_path = sys.argv[1]
output_path = sys.argv[2]

spark = SparkSession.builder.appName("TitanicPreprocess").getOrCreate()
df = spark.read.csv(input_path, header=True, inferSchema=True)
df = df.drop("PassengerId", "Name", "Ticket", "Cabin")
imputer = Imputer(inputCols=["Age", "Fare"], outputCols=["Age", "Fare"]).setStrategy("median")
df = imputer.fit(df).transform(df)
indexer = StringIndexer(inputCols=["Sex", "Embarked"], outputCols=["Sex_idx", "Embarked_idx"], handleInvalid="keep")
df = indexer.fit(df).transform(df)
assembler = VectorAssembler(inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_idx", "Embarked_idx"], outputCol="features")
df = assembler.transform(df)
processed_df = df.select("features", "Survived")
processed_df.write.mode("overwrite").csv(output_path, header=True)
spark.stop()
