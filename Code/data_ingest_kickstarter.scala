/* 1. Read json files */
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val path = "/user/kll482/final_project/Kickstarter_20191017.json"
var df = sqlContext.read.option("inferSchema", "true").json(path)

