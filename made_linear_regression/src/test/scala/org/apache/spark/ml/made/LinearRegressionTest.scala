package org.apache.spark.ml.made

import breeze.linalg.DenseMatrix
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ListBuffer

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
    println("Hello, tests are started here!")
    val delta = 0.0000001
    lazy val data: DataFrame = LinearRegressionTest._data

    private def validateModel(model: LinearRegressionModel, data: DataFrame): Unit = {
        val vectors: Array[Vector] = data.select("features").collect().map(_.getAs[Vector](0))
        val predictions: Array[Double] = data.select("predictions").collect().map(_.getAs[Double](0))

        vectors.length should be(data.count())

        for (i <- 0 to vectors.length - 1) {
            (1.5 * vectors(i)(0) + 0.3 * vectors(i)(1) - 0.7) should be(predictions(i) +- delta)
        }
    }

    "Model" should "transform (predict target of) input data" in {
        val model: LinearRegressionModel = new LinearRegressionModel(
            weights = Vectors.dense(1.5, 0.3).toDense,
            bias = -0.7
        ).setInputCol("features").setOutputCol("predictions")

        val data_transformed: DataFrame = model.transform(data)

        validateModel(model, data_transformed)
    }

    "Estimator" should "approximate weights and bias with gradient descent" in {
        val estimator = new LinearRegression()
                .setInputCol("features")
                .setLabelCol("target")
                .setNumIter(1000)
                .setLambda(0.8)
                .setUseBias(true)

        val lr_model: LinearRegressionModel = new LinearRegressionModel(
            weights = Vectors.dense(1.5, 0.3).toDense,
            bias = -0.7
        ).setInputCol("features").setOutputCol("target")

        val data_transformed: DataFrame = lr_model.transform(data)
        val lr_estim = estimator.fit(data_transformed)

        lr_estim.weights(0) should be(1.5 +- delta)
        lr_estim.weights(1) should be(0.3 +- delta)
        lr_estim.bias should be(-0.7 +- delta)

    }

    "Estimator" should "should produce functional model" in {
        val estimator = new LinearRegression()
            .setInputCol("features")
            .setLabelCol("target")

        val lr_model: LinearRegressionModel = new LinearRegressionModel(
            weights = Vectors.dense(1.5, 0.3).toDense,
            bias = -0.7
        ).setInputCol("features").setOutputCol("target")

        var data_transformed: DataFrame = lr_model.transform(data)
        val model = estimator.fit(data_transformed)

        data_transformed = data_transformed.withColumnRenamed("target", "predictions")
        validateModel(model, data_transformed)
    }

    "Estimator" should "work after re-read" in {

        val pipeline = new Pipeline().setStages(Array(
            new LinearRegression()
                .setInputCol("features")
                .setLabelCol("target")
                .setNumIter(1000)
                .setLambda(0.8)
                .setUseBias(true)
        ))

        val lr_model: LinearRegressionModel = new LinearRegressionModel(
            weights = Vectors.dense(1.5, 0.3).toDense,
            bias = -0.7
        ).setInputCol("features").setOutputCol("target")

        var data_transformed: DataFrame = lr_model.transform(data)

        val model = pipeline.fit(data_transformed)

        val tmpFolder = Files.createTempDir()

        model.write.overwrite().save(tmpFolder.getAbsolutePath)
        val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

        val loaded_model = reRead.stages(0).asInstanceOf[LinearRegressionModel]

        loaded_model.bias should be(-0.7 +- delta)
        loaded_model.weights(0) should be(1.5 +- delta)
        loaded_model.weights(1) should be(0.3 +- delta)

        loaded_model.setInputCol("features").setOutputCol("predictions")
        val data_transformed_l = loaded_model.transform(data)
        validateModel(loaded_model, data_transformed_l)
    }
}

object LinearRegressionTest extends WithSpark {

    val matrix = DenseMatrix.rand[Double](100000, 2)
    var vectors = ListBuffer[List[Double]]()
    for (i <- 0 until matrix.rows) {
        val row = matrix(i, ::).t
        val vec = row.toScalaVector.toList
        vectors += vec
    }

    lazy val _data: DataFrame = {
        import sqlc.implicits._
        vectors.map((x: List[Double]) => Tuple1(Vectors.dense(x.toArray))).toDF("features")
    }
}
