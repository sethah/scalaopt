name := "scalaopt"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.spire-math" %% "spire" % "0.13.0",
  "org.scalanlp" %% "breeze" % "0.12")

javaOptions += "-Xlog-implicits"
    