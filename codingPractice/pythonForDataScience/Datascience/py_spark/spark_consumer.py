from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import col, from_json

# 1. Create Spark session
spark = SparkSession.builder \
    .appName("KafkaSparkConsumer") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Define schema
schema = StructType([
    StructField("FIRST_NAME", StringType(), True),
    StructField("LAST_NAME", StringType(), True),
    StructField("EMAIL", StringType(), True),
    StructField("PHONE_NUMBER", StringType(), True),
    StructField("HIRE_DATE", StringType(), True),
    StructField("SALARY", DoubleType(), True),
    StructField("DEPARTMENT_ID", IntegerType(), True),
    StructField("Image", StringType(), True),
    StructField("id", IntegerType(), True)
])

# 3. Read from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "employee-topic") \
    .option("startingOffsets", "earliest") \
    .load()

# 4. Parse Kafka value (which is in JSON string format)
parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*")

# 5. Output to console
query = parsed_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
