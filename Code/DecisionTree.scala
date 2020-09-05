val DT = new DecisionTreeClassifier()
val paramGrid = new ParamGridBuilder().
addGrid(DT.impurity, Array("entropy", "gini")).
addGrid(DT.maxDepth, Range(5, 25, 5)).
addGrid(DT.maxBins, Array(10, 20)).
addGrid(DT.minInstancesPerNode, Array(10, 25, 50, 100)).
build()

val CV = new CrossValidator().
setEstimator(DT).
setEvaluator(new BinaryClassificationEvaluator).
setEstimatorParamMaps(paramGrid).
setNumFolds(10).
setParallelism(4)


val CVDTModel = CV.fit(mytrain)


CVDTModel.getEstimatorParamMaps.
zip(CVDTModel.avgMetrics).
maxBy(_._2).
_1


val evaluator = new BinaryClassificationEvaluator().
setLabelCol("label").f
setRawPredictionCol("prediction").
setMetricName("areaUnderROC")



val predictionsDT = CVDTModel.transform(testset)
val predictiontrainDT = CVDTModel.transform(mytrain.dropDuplicates)
val rocDTCVTest = evaluator.evaluate(predictionsDT) 



val DTpredictionlabel = predictionsDT.
select($"prediction",$"label").
as[(Double, Double)].
rdd

val DTmMetrics = new MulticlassMetrics(DTpredictionlabel)
val DTlabels = DTmMetrics.labels

// Print out the Confusion matrix
println("Confusion Matrix:")
println(DTmMetrics.confusionMatrix)

DTmMetrics.precision

DTmMetrics.recall

DTmMetrics.fMeasure

// Precision by label
DTlabels.foreach { l =>
println(s"Precision($l) = " + DTmMetrics.precision(l))
}

// Recall by label
DTlabels.foreach { l =>
println(s"Recall($l) = " + DTmMetrics.recall(l))
}

// F-measure by label
DTlabels.foreach { l =>
println(s"F1-Score($l) = " + DTmMetrics.fMeasure(l))
}






