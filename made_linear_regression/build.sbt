ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.17"

lazy val root = (project in file("."))
  .settings(
    name := "made_linear_regression"
  )

val sparkVersion = "3.3.2"
val BreezeVersion = "2.1.0"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "org.scalanlp" %% "breeze" % BreezeVersion ,
    "org.scalanlp" %% "breeze-viz" % BreezeVersion
)


libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.15" % "test" withSources())
//
//val javaOptsSeq = Seq(
//    "--add-opens=java.base/java.io=ALL-UNNAMED",
//    "--add-opens=java.base/java.nio=ALL-UNNAMED",
//    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
//    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
//    "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
//    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
//    "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED"
//)
//
//Test / run / fork := true
//Test / fork := true
//Test / javaOptions ++= javaOptsSeq