package optimization

import breeze.linalg.DenseVector
import linesearch.StrongWolfe
import loss.DifferentiableFunction
import spire.algebra.{InnerProductSpace, Order, NRoot}
//import spire.implicits._

import scala.reflect.ClassTag

class ConjugateGradient extends Optimizer[DenseVector[Double],
  DifferentiableFunction[DenseVector[Double], Double]] {

  case class CGState(x: DenseVector[Double],
                     residual: DenseVector[Double],
                     direction: DenseVector[Double],
                     iter: Int,
                     converged: Boolean)  {
    lazy val rtr = residual dot residual
  }

  type State = CGState
  type History = Double

  def optimize(lossFunction: DifferentiableFunction[DenseVector[Double], Double],
               initialParameters: DenseVector[Double]): DenseVector[Double] = {
    initialParameters
  }

  def converged(state: State): Boolean = {
    state.iter > 20
  }

  def myInitialState(lossFunction: DifferentiableFunction[DenseVector[Double], Double],
    initialParams: DenseVector[Double]): State = {
    val (firstLoss, firstGrad) = lossFunction.compute(initialParams)
    CGState(initialParams, -1.0 * firstGrad, -1.0 * firstGrad, 0, false)
  }

  def infiniteIterations(
                          lossFunction: DifferentiableFunction[DenseVector[Double], Double],
                          start: State): Iterator[State] = {
    Iterator.iterate(start)(iterateOnce(lossFunction))
  }

  private def iterateOnce(lossFunction: DifferentiableFunction[DenseVector[Double], Double])(state: State): State = {
    // find alpha
    val ls = new StrongWolfe()
    val lineSearchFunction = new DifferentiableFunction[Double, Double] {
      def apply(x: Double): Double = {
        compute(x)._1
      }

      def gradientAt(x: Double): Double = compute(x)._2

      def compute(x: Double): (Double, Double) = {
        val step = x
        val (f, grad) = lossFunction.compute(state.x + state.direction * step)
        val thisGrad = grad dot state.direction
        (f, thisGrad)
      }
    }
    // TODO: why does breeze do this?
    val tmp3 = 1.0 / math.sqrt(state.direction dot state.direction)
    val initialAlpha = if (state.iter == 0) tmp3 else 1.0
    val alpha = ls.optimize(lineSearchFunction, initialAlpha)
    // update x
    val newx = state.x + alpha * state.direction
    // new residual
    val (newLoss, newGrad) = lossFunction.compute(newx)
    val newres = -1.0 * newGrad
    // calculate beta with new and old residual
    val beta = math.max(newres dot (newres - state.residual) / state.rtr, 0.0)
    // new direction
    val newdir = newres + beta * state.direction
    CGState(newx, newres, newdir, state.iter + 1, false)
  }

  def chooseDescentDirection(state: State): DenseVector[Double] = {
    // TODO: get the orthogonal search direction
    state.x
  }

  def chooseStepSize(lossFunction: DifferentiableFunction[DenseVector[Double], Double],
                     direction: DenseVector[Double], state: State): Double = {
    // TODO: compute alpha = direction^T A error / direction^T A direction
    1.0
  }

  /**
   * Return the next set of parameters from the current parameters.
   */
  def takeStep(position: DenseVector[Double], stepDirection: DenseVector[Double],
               stepSize: Double): DenseVector[Double] = {
    position + (stepDirection :* stepSize)
  }

  def initialHistory(lossFunction: DifferentiableFunction[DenseVector[Double], Double],
                     initialParams: DenseVector[Double]): History = {
    // TODO: what is history
    2.0
  }

  /**
   * Update the history for the optimizer. For LBFGS, this might be the information required to
   * compute the approximate inverse Hessian matrix.
   */
  def updateHistory(position: DenseVector[Double], gradient: DenseVector[Double],
                    value: Double, state: State): History = {
    2.0
    // TODO
  }

}
