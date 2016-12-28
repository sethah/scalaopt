package optimization

import linesearch.BacktrackingLineSearch
import loss.DifferentiableFunction
import breeze.optimize.{LBFGS => BLBFGS}
import spire.implicits._
import spire.algebra.{NRoot, Order, InnerProductSpace, Field}

import scala.reflect.ClassTag

/**
 * Classic LBFGS algorithm.
 *
 * @param m The number of iterations to keep history for.
 * @param stoppingCriteria Stopping criteria for iterative optimization.
 * @tparam T Type of parameters being optimized.
 * @tparam F The parameter's scalar type.
 */
class LBFGS[T, F: Order: ClassTag](
    m: Int,
    override val stoppingCriteria: (FirstOrderOptimizerState[T, F, (IndexedSeq[T], IndexedSeq[T])]) => Boolean)
    (implicit space: InnerProductSpace[T, F])
  extends FirstOrderOptimizer[T, F] {

  /** (deltaPosition, deltaGradient) */
  type History = (IndexedSeq[T], IndexedSeq[T])

  def initialHistory(lossFunction: DifferentiableFunction[T, F], initialParams: T): History = {
    (IndexedSeq.empty[T], IndexedSeq.empty[T])
  }

  /**
   * Add in the new deltas, remove the old ones.
   *
   * @param position Current parameters.
   * @param gradient Current gradient.
   * @param value Current loss.
   * @param state Previous optimization state.
   * @return Updated history of position and gradient deltas.
   */
  def updateHistory(
      position: T,
      gradient: T,
      value: F,
      state: State): History = {
    val (oldDeltaPositions, oldDeltaGrads) = state.history
    val newDeltaGrad = gradient - state.gradient
    val newDeltaPosition = position - state.params
    ((oldDeltaPositions :+ newDeltaPosition).take(m), (oldDeltaGrads :+ newDeltaGrad).take(m))
  }

  def takeStep(position: T, stepDirection: T, stepSize: F): T = {
    position + (stepDirection :* stepSize)
  }

  /**
   * Use a line search to determine how far to step in the given direction.
   *
   * Simple backtracking line search for now.
   */
  def chooseStepSize(lossFunction: DifferentiableFunction[T, F], direction: T, state: State): F = {
    // somewhat of a magic number
    import space._
    val beta = 0.8

    val dirNorm = direction dot direction
    var alpha = space.scalar.one
    while (lossFunction(state.params + (alpha *: direction)) >
      (lossFunction(state.params) - dirNorm * alpha / 2.0)) {
      println("updating alpha", beta, alpha)
      alpha = alpha * beta
    }
    alpha
  }

  /**
   * The descent direction is the Hessian multiplied with the gradient.
   *
   * Use the position and gradient history to approximate the Hessian and compute H^(-1)^ g.
   * TODO: explain the code that produces this multiplication.
   */
  def chooseDescentDirection(state: State): T = {
    LBFGS.multiplyApproxInverseHessian[T, F](state.gradient, state.history._1, state.history._2)
  }
}

object LBFGS {
  /**
   * Note: it is redundant to specify that F is a Field, since the innerproductspace
   * requires it. This will result in compilation errors because it won't understand where
   * to look for implicits, and the error message is useless.
   */
  def multiplyApproxInverseHessian[T, F: ClassTag](
      gradient: T,
      posDeltas: IndexedSeq[T],
      gradDeltas: IndexedSeq[T])
      (implicit space: InnerProductSpace[T, F]): T = {
    import space._
    val m = posDeltas.length
    val diag = if (posDeltas.length > 0) {
      val prevStep = posDeltas.head
      val prevGradStep = gradDeltas.head
      val sy = prevStep dot prevGradStep
      val yy = prevGradStep dot prevGradStep
//      println(sy, yy)
      // TODO: how to compare field to 0?
//      if (sy < 0 || sy.isNaN) throw new Exception("nan history")
      sy / yy
    } else {
      scalar.one
    }

    // TODO: rename, leave comments
    var dir = gradient
    val as = new Array[F](m)
    val rho = new Array[F](m)

    // TODO: in place ops
    for (i <- 0 until posDeltas.length) {
      rho(i) = (posDeltas(i) dot gradDeltas(i))
      as(i) = (posDeltas(i) dot dir) / rho(i)
      // TODO: below code...
//      if(as(i).isNaN) {
//        throw new NaNHistory
//      }
      dir = (gradDeltas(i) :* -as(i)) + dir
//      println("asdf", rho(i), as(i), posDeltas(i), gradDeltas(i))
    }

    dir = dir :* diag

    for(i <- (m - 1) to 0 by (-1)) {
      val beta = (gradDeltas(i) dot dir) / rho(i)
      dir = (posDeltas(i) :* as(i) - beta) + dir
    }
    -dir
  }
}
