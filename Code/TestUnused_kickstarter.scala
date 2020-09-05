df.printSchema
df.count

/* sub-Category percentage*/
import org.apache.spark.sql.functions.countDistinct
dfTech.agg(countDistinct($"subCategory")).collect // 16

val total = dfTech.count
dfTech.groupBy($"subCategory").count.select($"subCategory", $"count"/total).orderBy(desc("count")).show()

dfTech.groupBy($"newsubCategory").count.select($"newsubCategory", $"count"/total).orderBy(desc("count")).show()

dfTech.groupBy($"goalBin").count.select($"goalBin", $"count"/total).orderBy(desc("count")).show()

dfTech.groupBy($"goalBin").count.select($"goalBin", $"count"/total).orderBy(desc("count")).show()

val myAssembler = new VectorAssembler().
setInputCols(Array("blurblen", "namelen", "campaignlen", "preparelen")).
setOutputCol("featuresAssem")

val scaler = new StandardScaler().
setInputCol("featuresAssem").
setOutputCol("scaledFeaturesVec").
setWithStd(true).
setWithMean(true)

val pipelineAssembler = new Pipeline().setStages(Array(myAssembler, scaler))

val scalerDF = pipelineAssembler.fit(finaldf).transform(finaldf)

// INDEXER
val myindexer = new StringIndexer().
setInputCol("newsubCategory").
setOutputCol("subCategoryIndex")
var IndexDF = myindexer.fit(scalerDF).transform(scalerDF)

// One hot encoding

val onehotfeatures = Array("staff_pick", "countryUS", "subCategoryIndex", "monthdead", "monthLaunched", "goalBin")
val onehotencoder = new OneHotEncoderEstimator().
setInputCols(Array(element)).
setOutputCols(Array(element+ "Vec")).
setDropLast(true)

/*
val encoder = onehotfeatures.flatMap{element =>
val onehotencoder = new OneHotEncoderEstimator().
setInputCols(Array(element)).
setOutputCols(Array(element+ "Vec")).
setDropLast(true)
Array(onehotencoder)
}
*/

val mypipeline = new Pipeline().setStages(encoder)
val EncodedDF = mypipeline.fit(IndexDF).transform(IndexDF)

val myfeatures = EncodedDF.columns.filter(_.contains("Vec")).toArray

val Assembler = new VectorAssembler().
setInputCols(myfeatures).
setOutputCol("modelFeatures")

val vectorIndexer = new VectorIndexer().
setInputCol("modelFeatures").
setOutputCol("indexFeatures")

val pipelineAssembler = new Pipeline().setStages(Array(Assembler, vectorIndexer))

val resultDF = pipelineAssembler.fit(EncodedDF).transform(EncodedDF)


val LR = new LogisticRegression()

val model = LR.fit(mytrain)

/* */
val trainpredict = model.transform(mytrain)

val predictions = model.transform(testset)

val evaluator = new BinaryClassificationEvaluator().
setLabelCol("label").
setRawPredictionCol("prediction").
setMetricName("areaUnderROC")

val rocTest = evaluator.evaluate(predictions)
val rocTrain = evaluator.evaluate(trainpredict)


/* Tableau */
df.registerTempTable("kickstarter")
sqlContext.sql("drop table kll482.NewKickstarter")
sqlContext.sql("create table kll482.NewKickstarter as select * from kickstarter")


/* Wrtie Json */
// json
val pathPlot = "file:///home/kll482/final_project/Kickstarter_Tech"
dfTechnology.coalesce(1).write.format("json").save(pathPlot)
