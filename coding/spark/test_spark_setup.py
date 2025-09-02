import findspark
findspark.init()

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Spark Test") \
    .getOrCreate()

# Test Spark by creating a simple DataFrame
data = [("John", 28), ("Jane", 30), ("Mike", 35)]
columns = ["Name", "Age"]

df = spark.createDataFrame(data, columns)

# Show the DataFrame
df.show()

# Perform a basic operation (e.g., count)
count = df.count()
print(f"Total number of rows in DataFrame: {count}")

# Stop the Spark session
spark.stop()

####==========================================================================================2nd Test with local[1] and increased memory

# Initialize Spark Session with increased memory
spark = SparkSession.builder \
    .appName("Test Spark Setup") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .master("local[1]") \
    .getOrCreate()
print("Spark session initialized successfully.")

# Parallelize a list and compute sum
rdd = spark.sparkContext.parallelize(range(1, 11))
print("RDD created with numbers from 1 to 10.")
total = rdd.sum()
print(f"Sum of numbers from 1 to 10 is: {total}")

# Stop the Spark session
spark.stop()
