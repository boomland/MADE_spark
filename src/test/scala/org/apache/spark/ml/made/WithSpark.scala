package org.apache.spark.ml.made

import org.apache.spark.sql.SparkSession


// будем унаследоваться от этого трейта и будем
// таким образром юзать спарк
trait WithSpark {
   lazy val spark = WithSpark._spark
   lazy val sqlc = WithSpark._sqlc
}

// Companion object (singletone)
// Создаем один раз на весь запуск юниттестов
// чтобы для каждого модульного теста не создавать новый
object WithSpark {
  lazy val _spark = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  lazy val _sqlc = _spark.sqlContext
}
