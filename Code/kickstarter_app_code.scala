/* Ecoding, Scaling */
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.sql.functions.rand
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import scala.math
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

/* FINAL PIPELINE*/
val myindexer = new StringIndexer().
setInputCol("newsubCategory").
setOutputCol("subCategoryIndex")

val onehotfeatures = Array("staff_pick", "countryUS", "subCategoryIndex", "monthdead", "monthLaunched", "goalBin")
val onehotfeaturesVec = Array("staff_pickVec", "countryUSVec", "subCategoryIndexVec", "monthdeadVec", "monthLaunchedVec", "goalBinVec")
val onehotencoder = new OneHotEncoderEstimator().
setInputCols(onehotfeatures).
setOutputCols(onehotfeaturesVec).
setDropLast(true)

val myassembler = (new VectorAssembler().
setInputCols(Array("staff_pickVec", "countryUSVec", "subCategoryIndexVec",
"monthdeadVec", "monthLaunchedVec", "goalBinVec", "blurblen", "namelen", "campaignlen", "preparelen")).
setOutputCol("featuresAssem"))

val myscaler = new StandardScaler().
setInputCol("featuresAssem").
setOutputCol("features").
setWithStd(true).
setWithMean(true)

val pipelineAssembler = new Pipeline().setStages(Array(myindexer, onehotencoder, myassembler, myscaler))
val resultDF = pipelineAssembler.fit(finaldf).transform(finaldf)

val Array(trainset, testset) = resultDF.randomSplit(Array(0.75, 0.25), seed = 101)

/* oversampling */
trainset.groupBy($"label").count.collect // Array([1,2759], [0,6327])

val training = trainset.union(trainset.filter($"label" === 1).orderBy(rand()).limit(2000))

val newtraing = training.filter($"label" === 0).orderBy(rand()).limit(4700)
val mytrain = newtraing.union(training.filter($"label" === 1).orderBy(rand()).limit(4700))
mytrain.groupBy($"label").count.collect

/* Model */

/* Logistic Regression */
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


/* Decision Tree */
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



/* Random Forest */

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

/* SVM */
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








