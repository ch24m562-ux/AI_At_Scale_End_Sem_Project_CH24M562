from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("combine").getOrCreate()
df = spark.read.csv("data/processed", header=True, inferSchema=True)
df.coalesce(1).write.mode("overwrite").csv("data/processed_single", header=True)
spark.stop()
