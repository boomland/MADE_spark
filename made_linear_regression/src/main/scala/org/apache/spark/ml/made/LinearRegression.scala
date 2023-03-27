package org.apache.spark.ml.made

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam}
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.param._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait LinearRegressionParams extends HasInputCol with HasOutputCol with HasLabelCol {
    def setInputCol(value: String) : this.type = set(inputCol, value)
    def setOutputCol(value: String): this.type = set(outputCol, value)
    def setLabelCol(str: String): this.type = set(labelCol, str)

    // do we need use for lr bias
    val useLinRegBias = new BooleanParam(parent = this, name = "useLinRegBias", doc = "Do we use bias (intercept) for training linear regression?")
    def isUseBias(): Boolean = $(useLinRegBias)
    def setUseBias(value: Boolean): this.type = set(useLinRegBias, value)

    // lambda for linear regression "fit" method
    val gdLambda = new DoubleParam(parent = this, name = "gdLambda", doc = "Coefficient for step in graident descent method")
    def getLambda(): Double = $(gdLambda)
    def setLambda(value: Double): this.type = set(gdLambda, value)

    // num iterations for linear regression "fit" method
    val gdNumIter = new IntParam(parent = this, name = "gdNumIter", doc = "Number of iterations in gradient descent method")
    def getNumIter(): Int = $(gdNumIter)
    def setNumIter(value: Int): this.type = set(gdNumIter, value)

    // set default values
    setDefault(useLinRegBias -> true)
    setDefault(gdLambda -> 1.2)
    setDefault(gdNumIter -> 100)

    // method to validate transform schema
    protected def validateAndTransformSchema(schema: StructType): StructType = {
        SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

        if (schema.fieldNames.contains($(outputCol))) {
            SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
            schema
        } else {
            SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
        }
    }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {

    def this() = this(Identifiable.randomUID("linearRegression"))

    override def fit(dataset: Dataset[_]): LinearRegressionModel = {

        // Used to converlst untyped dataframes to datasets with vectors
        implicit val encoder_vectors: Encoder[Vector] = ExpressionEncoder()
        implicit val encoder_doubles: Encoder[Double] = ExpressionEncoder()

        val vectors: Dataset[(Vector, Double)] = dataset.select(dataset($(inputCol)).as[Vector], dataset($(labelCol)).as[Double])

        val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
            vectors.first()._1.size
        )

        var weights = breeze.linalg.DenseVector.rand[Double](dim)
        val bias_rand = breeze.linalg.DenseVector.rand[Double](1)
        var bias = bias_rand(0)

        val lambda = getLambda()
        val num_objects = dataset.count()
        val step = lambda / num_objects

        val num_iterations = getNumIter()

        println("Weights: " + weights.toString())
        println("Bias: " + bias.toString())
        println("Lambda: " + lambda.toString())
        println("Num iterations: " + num_iterations.toString())
        println("Num objects: " + num_objects.toString())

        for (iter_num <- 1 to num_iterations) {
            val grads = vectors.rdd.map({
                    vect: (Vector, Double) => {
                            val x = vect._1.asBreeze
                            val y = vect._2
                            val prediction = (x dot weights) + bias
                            val diff = prediction - y
                            val vec_diff = x * diff
                            (vec_diff, diff)
                        }
            }).reduce({
                (diff_a, diff_b) => (diff_a._1 + diff_b._1, diff_a._2 + diff_b._2)
            })

            weights -= step * grads._1
            bias -= step * grads._2
        }

        println("SUMMARY:")
        println("+++++++++++++++++++++++++++++++++++++++++")
        println("Bias: " + bias.toString())
        println("Weights: " + weights.toString())
        println("+++++++++++++++++++++++++++++++++++++++++")


        copyValues(new LinearRegressionModel(
            Vectors.fromBreeze(weights), bias
        )).setParent(this)
    }

    override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val weights: DenseVector,
                                           val bias: Double) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


    private[made] def this(weights: Vector, bias: Double) =
        this(Identifiable.randomUID("LinearRegressionModel"), weights.toDense, bias)

    override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
        new LinearRegressionModel(weights, bias), extra)

    override def transform(dataset: Dataset[_]): DataFrame = {
        val transformUdf = if (isUseBias()) {
            dataset.sqlContext.udf.register(uid + "_transform", (x: DenseVector) => { (x.asBreeze dot weights.asBreeze) + bias })
        } else {
            dataset.sqlContext.udf.register(uid + "_transform", (x: DenseVector) => { x.asBreeze dot weights.asBreeze })
        }
        dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
    }

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

    override def write: MLWriter = new DefaultParamsWriter(this) {
        override protected def saveImpl(path: String): Unit = {
            super.saveImpl(path)
            val vectors: (Vector, Double) = weights.asInstanceOf[Vector] -> bias.asInstanceOf[Double]
            sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
            println("Linear model was successfully saved")
        }
    }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
    override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
        override def load(path: String): LinearRegressionModel = {
            val metadata = DefaultParamsReader.loadMetadata(path, sc)
            val vectors = sqlContext.read.parquet(path + "/vectors")

            implicit val encoder_vectors : Encoder[Vector] = ExpressionEncoder()
            implicit val encoder_doubles : Encoder[Double] = ExpressionEncoder()

            val (weights, bias) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Double]).first()

            val model = new LinearRegressionModel(weights = weights, bias = bias)
            metadata.getAndSetParams(model)
            model
        }
    }
}
