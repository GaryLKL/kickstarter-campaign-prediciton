df.registerTempTable("kickstarter")

/* Maximal length for string-type column*/
sqlContext.sql("select max(length(name)), max(length(state)), max(length(blurb)), max(length(country)), max(length(currency)) from kickstarter").take(1)
sqlContext.sql("select max(length(mainCategory)), max(length(subCategory)), max(length(url)) from kickstarter").take(1)

/* Count the range of each integer and double type column*/
sqlContext.sql("select max(backers_count), min(backers_count), max(converted_pledged_amount), min(converted_pledged_amount), max(usd_pledged), min(usd_pledged), max(goal), min(goal), max(id), min(id), max(creatorID), min(creatorID) from kickstarter").collect

/* Date range */
sqlContext.sql("select max(created_at), min(created_at), max(launched_at), min(launched_at), max(deadline), min(deadline) from kickstarter").collect
// Returns Array([2019-10-17,2009-04-25,2019-12-16,2009-05-16])]
import org.apache.spark.sql.types._
for (col <- df.columns){
df.schema(col).dataType match {
case StringType => df.select(length(df(col)).as(col)).describe(col).show
case LongType => df.select(max(df(col)).as("maxValue")).describe("maxValue").show
case _ => print("The data type is neither long or string")
}
}

/* category percentage */
sqlContext.sql("select mainCategory, count(*)/(select count(*) from kickstarter) as perc from kickstarter group by mainCategory").show

/* Year vs. Number*/
// How many campaigns per year?
spark.sql("select year(launched_at) as year, count(*) from kickstarter group by year order by year").collect
// The mean number of campaigns each month
spark.sql("select month, round(mean(number), 2) from (select month(launched_at) as month, year(launched_at) as year, count(*) as number from kickstarter group by month, year) as t1 group by month order by month").collect