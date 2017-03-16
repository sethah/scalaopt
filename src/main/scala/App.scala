import breeze.linalg._
import breeze.optimize.{LBFGS => BLBFGS, DiffFunction}
import loss.LeastSquaresLossFunction
import optimization._
import spire.algebra.{Field, InnerProductSpace}
import spire.implicits._
import implicits._
import util.LinearDataGenerator

object App {

  def main(args: Array[String]): Unit = {
//    val m = new DenseMatrix[Float](2, 2, Array(1.0, 2.0, 4.0, 4.0).map(_.toFloat))
//    val label = new DenseVector[Float](Array(14.0, 16.0).map(_.toFloat))
//    val lf = new LeastSquaresLossFunction(m, label)
//    val stoppingCriteria = (state: FirstOrderOptimizerState[DenseVector[Double], Double, (IndexedSeq[DenseVector[Double]], IndexedSeq[DenseVector[Double]])]) => {
//      state.iter > 4
//    }
//    val stop = (state: FirstOrderOptimizerState[_, _, _]) => {
//      state.iter > 5000
//    }
//    val stoppingCriteria =
//      new StoppingCriteria[FirstOrderOptimizerState[DenseVector[Float], Float, _]] {
//        def apply(state: FirstOrderOptimizerState[DenseVector[Float], Float, _]): Boolean = {
//          state.iter > 5
//        }
//      }
//    val (data, labels, coef) = new LinearDataGenerator[Double](42).generate(false, 2, 2, 0.0, 0.0)
    val data = new DenseMatrix(2, 2, Array(-0.42802259172575496,  0.6500079069927311,
      -0.39985491857957944,  -0.65962536952953))
    val labels = new DenseVector(Array(-0.23214907545234986, -0.1584664514815176))
    val coef = (1.0, 1.0)
    println(data)
    println(labels)
    println(coef)
    val lossFunc = new LeastSquaresLossFunction(data, labels)
//    val optimizer = new GradientDescent[DenseVector[Double], Double](0.005, stop)
//    val optimizer = new LBFGS[DenseVector[Double], Double](7)
//    val optimizer = new ConjugateGradient()
    val optimizer = new VLBFGS[DenseVector[Double], Double](7)
    val initialCoef = new DenseVector(Array.fill(2)(0.0))
    val coefs = optimizer.optimize(lossFunc, initialCoef)
//    optimizer.infiniteIterations(lossFunc, optimizer.myInitialState(lossFunc, initialCoef))
//      .take(10).foreach { state =>
//      println(state.x)
//    }
//    val coefs = optimizer.optimize(lossFunc, initialCoef)
//    println(coef)
    println(coefs)
    println("------------------------------------")

    val boptimizer = new BLBFGS[DenseVector[Double]](7)
    val bloss = new DiffFunction[DenseVector[Double]] {
      def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
        lossFunc.compute(x)
      }
    }
    val bcoef = boptimizer.minimize(bloss, initialCoef)
    println(bcoef)
//    val hist = new VLBFGS.History[DenseVector[Double], Double](3)
//    val m = 3
//    val hist = new VLBFGS.History[DenseVector[Double], Double](m, 0,
//      Array.ofDim[Double](2 * m + 1, 2 * m + 1),
//      new Array[DenseVector[Double]](m), new Array[DenseVector[Double]](m))
//    val h1 = hist.update(DenseVector(Array(5.0, -1.0)), DenseVector(Array(1.0, 2.0)), DenseVector(Array(2.0, 3.0)))
//    println()
//    h1.innerProducts.foreach(arr => println(arr.mkString(",")))
//    val h2 = h1.update(DenseVector(Array(5.0, -1.0)), DenseVector(Array(1.0, 2.0)), DenseVector(Array(2.0, 3.0)))
//    println()
//    h2.innerProducts.foreach(arr => println(arr.mkString(",")))
//    val h3 = h2.update(DenseVector(Array(5.0, -1.0)), DenseVector(Array(1.0, 2.0)), DenseVector(Array(2.0, 3.0)))
//    println()
//    h3.innerProducts.foreach(arr => println(arr.mkString(",")))
//    val grad = DenseVector(Array(5.0, -1.0))
//    val dir = h3.computeDirection(grad)
//    println(dir)
//    val pds = Array.fill(3)(DenseVector(Array(1.0, 2.0)))
//    val gds = Array.fill(3)(DenseVector(Array(2.0, 3.0)))
//    val dir2 = LBFGS.multiplyApproxInverseHessian(grad, pds, gds)
//    println(dir2)
  }

}
