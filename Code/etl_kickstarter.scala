/* 1. Read json files */
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val path = "/user/kll482/final_project/Kickstarter_20191017.json"
var df = sqlContext.read.option("inferSchema", "true").json(path)

// Parse Json File
df = df.select("data.*").
drop("country_displayable_name", "friends", "permissions", "photo", "profile").
drop("currency_symbol", "currency_trailing_code", "current_currency", "disable_communication").
drop("fx_rate", "is_backing", "is_starrable", "is_starred", "slug", "source_url").
drop("spotlight", "state_changed_at", "static_usd_rate", "usd_type", "pledged", "location")

// date format
df = df.withColumn("launched_at", to_date(from_unixtime($"launched_at".cast("string"),"yyyy-MM-dd"))).
withColumn("created_at", to_date(from_unixtime($"created_at".cast("string"),"yyyy-MM-dd"))).
withColumn("deadline", to_date(from_unixtime($"deadline".cast("string"),"yyyy-MM-dd")))

df = df.withColumn("mainCategory", split($"category.slug", "/")(0)).
withColumn("subCategory", col("category.name")).
withColumn("creatorID", col("creator.id")).
withColumn("url", col("urls.web.project")).
drop("category", "creator", "urls")

// duplicated row?
df.registerTempTable("kickstarter")
sqlContext.sql("select count(distinct(name)) from kickstarter").collect // 179782; duplicated id of the projects?
sqlContext.sql("select count(*) from kickstarter").collect // 205227; each row is distinct.
val duplicatedName = sqlContext.sql("SELECT name, count(*) as number FROM kickstarter GROUP BY name ORDER BY number DESC")
sqlContext.sql("select count(distinct(url)) from kickstarter").collect // 179782

// Duplicated ID
val sameURL = sqlContext.sql("select k1.id, k2.id, k1.name, k2.name, k1.launched_at, k2.launched_at, k1.url, k2.url from kickstarter k1, kickstarter k2 where k1.url = k2.url and k1.name != k2.name")
sameURL.take(1) // Having same id but different name

df = df.dropDuplicates("id")




