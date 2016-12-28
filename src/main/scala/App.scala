import breeze.linalg._
import breeze.optimize.{LBFGS => BLBFGS, DiffFunction}
import loss.LeastSquaresLossFunction
import optimization.{LBFGS, StoppingCriteria, FirstOrderOptimizerState, GradientDescent}
import spire.algebra.{Field, InnerProductSpace}
import spire.implicits._
import implicits._
import util.LinearDataGenerator

object App {

  def main(args: Array[String]): Unit = {
    val m = new DenseMatrix[Float](2, 2, Array(1.0, 2.0, 4.0, 4.0).map(_.toFloat))
    val label = new DenseVector[Float](Array(14.0, 16.0).map(_.toFloat))
    val lf = new LeastSquaresLossFunction(m, label)
    val stoppingCriteria = (state: FirstOrderOptimizerState[DenseVector[Double], Double, (IndexedSeq[DenseVector[Double]], IndexedSeq[DenseVector[Double]])]) => {
//      println(state.params, "a;lksdfjasl;dfj")
      state.iter > 10
    }
    val stop = (state: FirstOrderOptimizerState[_, _, _]) => {
//      println(state.value)
      state.iter > 5000
    }
//    val stoppingCriteria =
//      new StoppingCriteria[FirstOrderOptimizerState[DenseVector[Float], Float, _]] {
//        def apply(state: FirstOrderOptimizerState[DenseVector[Float], Float, _]): Boolean = {
//          state.iter > 5
//        }
//      }
    val (data, labels, coef) = new LinearDataGenerator[Double](42).generate(false, 2, 2, 0.0, 0.0)
    val lossFunc = new LeastSquaresLossFunction(data, labels)
//    val optimizer = new GradientDescent[DenseVector[Double], Double](0.005, stop)
    val optimizer = new LBFGS[DenseVector[Double], Double](7, stoppingCriteria)
    val initialCoef = new DenseVector(Array.fill(2)(0.0))
    val coefs = optimizer.optimize(lossFunc, initialCoef)
    println(coefs, coef)
    println("------------------------------------")

    val boptimizer = new BLBFGS[DenseVector[Double]](7)
    val bloss = new DiffFunction[DenseVector[Double]] {
      def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
        lossFunc.compute(x)
      }
    }
    val bcoef = boptimizer.minimize(bloss, initialCoef)
    println(bcoef)

  }

}
