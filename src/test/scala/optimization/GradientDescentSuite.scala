package optimization

import util.LinearDataGenerator

class GradientDescentSuite {

  def myTest(): Unit = {
    val (data, labels, coef) = new LinearDataGenerator[Double](42).generate(false, 100, 3, 0.0, 1.0)
    println(data)
  }

}

object GradientDescentSuite {

  def main(args: Array[String]): Unit = {
    val (data, labels, coef) = new LinearDataGenerator[Double](42).generate(false, 100, 3, 0.0, 1.0)
    println(data)
  }
}
