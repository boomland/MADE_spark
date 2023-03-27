package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix}
import com.google.common.io.Files
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ListBuffer

object Main {
    def main(args: Array[String]): Unit = {
        // In main function I ran some example and parts of LinearRegrssion class code.
        // Please skip it

        lazy val spark = SparkSession.builder
            .appName("Simple Application")
            .master("local[4]")
            .getOrCreate()

        lazy val sqlc = spark.sqlContext

        val matrix = DenseMatrix.rand[Double](100000, 2)
        var vectors = ListBuffer[List[Double]]()
        for (i <- 0 until matrix.rows) {
            val row = matrix(i, ::).t
            val vec = row.toScalaVector.toList
            vectors += vec
        }

        lazy val data: DataFrame = {
            import sqlc.implicits._
            vectors.map((x: List[Double]) => Tuple1(Vectors.dense(x.toArray))).toDF("features")
        }

        val lr_model: LinearRegressionModel = new LinearRegressionModel(
            weights = Vectors.dense(1.5, 0.3).toDense,
            bias = -0.7
        ).setUseBias(true).setInputCol("features").setOutputCol("label").setLabelCol("label")

        val transformed_vectors: DataFrame = lr_model.transform(data)

        println("Transformed data:")
        transformed_vectors.show()
        val lr_estimator: LinearRegression = new LinearRegression().setInputCol("features").setLabelCol("label").setNumIter(1000).setLambda(0.8)
        lr_estimator.fit(transformed_vectors)


        val pipeline = new Pipeline().setStages(Array(
            new LinearRegression()
                .setInputCol("features")
                .setOutputCol("label")
                .setLabelCol("label")
        ))

        val tmpFolder = Files.createTempDir()

        pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

        val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

        val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    }
}