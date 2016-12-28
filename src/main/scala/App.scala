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
      state.iter > 4
    }
    val stop = (state: FirstOrderOptimizerState[_, _, _]) => {
      state.iter > 5000
    }
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
