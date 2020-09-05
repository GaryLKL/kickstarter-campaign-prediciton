val RF = new RandomForestClassifier()
val paramGrid = new ParamGridBuilder().
addGrid(RF.numTrees, Array(50, 100, 150)).
addGrid(RF.maxDepth, Array(5, 10, 20)).
build()

val CV = new CrossValidator().
setEstimator(RF).
setEvaluator(new BinaryClassificationEvaluator).
setEstimatorParamMaps(paramGrid).
setNumFolds(10).
setParallelism(4)


val CVRFModel = CV.fit(mytrain)

CVRFModel.getEstimatorParamMaps.
zip(CVRFModel.avgMetrics).
maxBy(_._2).
_1

CVRFModel.avgMetrics

val evaluator = new BinaryClassificationEvaluator().
setLabelCol("label").
setRawPredictionCol("prediction").
setMetricName("areaUnderROC")



val predictionsRF = CVRFModel.transform(testset)
val rocRFCVTest = evaluator.evaluate(predictionsRF)



val RFpredictionlabel = predictionsRF.
select($"prediction",$"label").
as[(Double, Double)].
rdd

val RFmMetrics = new MulticlassMetrics(RFpredictionlabel)
val RFlabels = RFmMetrics.labels

// Print out the Confusion matrix
println("Confusion Matrix:")
println(RFmMetrics.confusionMatrix)

RFmMetrics.precision

RFmMetrics.recall

RFmMetrics.fMeasure

// Precision by label
RFlabels.foreach { l =>
println(s"Precision($l) = " + RFmMetrics.precision(l))
}

// Recall by label
RFlabels.foreach { l =>
println(s"Recall($l) = " + RFmMetrics.recall(l))
}

// F-measure by label
RFlabels.foreach { l =>
println(s"F1-Score($l) = " + RFmMetrics.fMeasure(l))
}

