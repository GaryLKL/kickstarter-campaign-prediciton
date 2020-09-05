/* Cross Validation */
val LR = new LogisticRegression()
val paramGrid = new ParamGridBuilder().
addGrid(LR.regParam, Array(0.01, 0.05, 0.1, 0.5, 0, 1, 10, 100, 1000)).
addGrid(LR.setElasticNetParam, Array(0, 0.2, 0.4, 0.6, 0.8, 1)).
build()

val CV = new CrossValidator().
setEstimator(LR).
setEvaluator(new BinaryClassificationEvaluator).
setEstimatorParamMaps(paramGrid).
setNumFolds(10).
setParallelism(4)

//var mytrainUnique = mytrain.dropDuplicates

val CVLogModel = CV.fit(mytrain)


CVLogModel.getEstimatorParamMaps.
zip(CVLogModel.avgMetrics).
maxBy(_._2).
_1

CVLogModel.avgMetrics

val evaluator = new BinaryClassificationEvaluator().
setLabelCol("label").
setRawPredictionCol("prediction").
setMetricName("areaUnderROC")



val predictionsLog = CVLogModel.transform(testset)
val predictionstrainLog = CVLogModel.transform(mytrain)
val rocLogCVTest = evaluator.evaluate(predictionsLog) // 72.97%



val Logpredictionlabel = predictionsLog.
select($"prediction",$"label").
as[(Double, Double)].
rdd

val LogbMetrics = new BinaryClassificationMetrics(Logpredictionlabel)

val LogmMetrics = new MulticlassMetrics(Logpredictionlabel)
val Loglabels = LogmMetrics.labels

// Print out the Confusion matrix
println("Confusion Matrix:")
println(LogmMetrics.confusionMatrix)

LogmMetrics.precision

LogmMetrics.recall

LogmMetrics.fMeasure

// Precision by label
Loglabels.foreach { l =>
println(s"Precision($l) = " + LogmMetrics.precision(l))
}

// Recall by label
Loglabels.foreach { l =>
println(s"Recall($l) = " + LogmMetrics.recall(l))
}

// F-measure by label
Loglabels.foreach { l =>
println(s"F1-Score($l) = " + LogmMetrics.fMeasure(l))
}

