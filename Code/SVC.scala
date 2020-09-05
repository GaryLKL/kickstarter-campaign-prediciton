val SVC = new LinearSVC()
val paramGrid = new ParamGridBuilder().
addGrid(SVC.setRegParam, Array(0.001, 0.01, 0.1, 0, 1, 10, 100)).
build()

val CV = new CrossValidator().
setEstimator(SVC).
setEvaluator(new BinaryClassificationEvaluator).
setEstimatorParamMaps(paramGrid).
setNumFolds(10).
setParallelism(4)


val CVSVCModel = CV.fit(mytrain)

CVSVCModel.getEstimatorParamMaps.
zip(CVSVCModel.avgMetrics).
maxBy(_._2).
_1

CVSVCModel.avgMetrics

val evaluator = new BinaryClassificationEvaluator().
setLabelCol("label").
setRawPredictionCol("prediction").
setMetricName("areaUnderROC")



val predictionsSVC = CVSVCModel.transform(testset)
val rocSVCCVTest = evaluator.evaluate(predictionsSVC)



val SVCpredictionlabel = predictionsSVC.
select($"prediction",$"label").
as[(Double, Double)].
rdd

val SVCmMetrics = new MulticlassMetrics(SVCpredictionlabel)
val SVClabels = SVCmMetrics.labels

// Print out the Confusion matrix
println("Confusion Matrix:")
println(SVCmMetrics.confusionMatrix)

SVCmMetrics.precision

SVCmMetrics.recall

SVCmMetrics.fMeasure

// Precision by label
SVClabels.foreach { l =>
println(s"Precision($l) = " + SVCmMetrics.precision(l))
}

// Recall by label
SVClabels.foreach { l =>
println(s"Recall($l) = " + SVCmMetrics.recall(l))
}

// F-measure by label
SVClabels.foreach { l =>
println(s"F1-Score($l) = " + SVCmMetrics.fMeasure(l))
}

