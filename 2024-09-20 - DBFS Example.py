# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/tips.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.csv(file_location,header=True,inferSchema=True)
df.show()

# COMMAND ----------

df.columns

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# COMMAND ----------

indexer = StringIndexer(inputCol="sex",outputCol="sex_index")
df_r = indexer.fit(df).transform(df) 
df_r.show()

# COMMAND ----------


indexer=StringIndexer(inputCols=["smoker","day","time"],outputCols=["smoker_indexed","day_indexed",
                                                                  "time_index"])
df_r=indexer.fit(df_r).transform(df_r)
df_r.show()

# COMMAND ----------


from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

feature_assembler= VectorAssembler(inputCols=["tip","size","smoker_indexed","day_indexed","time_index"],outputCol="Independent_feature")
df = feature_assembler.transform(df_r)

# COMMAND ----------

df.show()

# COMMAND ----------

final_data = df.select("Independent_feature","total_bill")

# COMMAND ----------

final_data.show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

train_data,test_data = final_data.randomSplit([0.75,0.25])
regessor = LinearRegression(featuresCol="Independent_feature",labelCol="total_bill")
regessor = regessor.fit(train_data)

# COMMAND ----------


regessor.coefficients
     

# COMMAND ----------


regessor.intercept

# COMMAND ----------


### Predictions
pred_results=regessor.evaluate(test_data)

# COMMAND ----------

pred_results.predictions.show()

# COMMAND ----------


### PErformance Metrics
pred_results.r2,pred_results.meanAbsoluteError,pred_results.meanSquaredError

# COMMAND ----------


