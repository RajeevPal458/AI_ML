#We're about to build a local Kafka + Python producer + PySpark streaming consumer setup — a powerful pattern for real-time data pipelines.

# start Zookeeper -> C:\kafka_2.13-3.9.1>.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties

# start server broker -> C:\kafka_2.13-3.9.1>.\bin\windows\kafka-server-start.bat .\config\server.properties

# now veryfy sertup using cmd producer and consumer -> 
    # producer ->  C:\kafka_2.13-3.9.1\bin\windows>Kafka-console-producer.sh --bootstrap-server localhost:9092 --topic my-topic
    # Consumer -> C:\kafka_2.13-3.9.1\bin\windows>kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic my-topic --from-beginning

# if all good then start with writing python code pyspark setup

    # create python producer.py
    # create python  spark-consumer.py file
    # now run in terminal use command->$ spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 spark_consumer.py



