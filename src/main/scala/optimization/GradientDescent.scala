package optimization

import breeze.math.Field
import spire.algebra.InnerProductSpace
import spire.implicits._

import loss.DifferentiableFunction

/**
 * Simple gradient descent implementation.
 * @param stepSize
 * @param stoppingCriteria
 * @param space We need an inner product space to be defined for type T to F because
 *              we need to be move around the parameter space.
 *
 *              TODO: does it need to be an inner product space? Only using + and - ops right now.
 *              TODO: Use the exact step size? Or implement a line search? What are the
 *              TODO: computational costs of each?
 * @tparam T
 * @tparam F
 */
class GradientDescent[T, F](
    stepSize: F,
    override val stoppingCriteria: (FirstOrderOptimizerState[T, F, _] => Boolean))
    (implicit space: InnerProductSpace[T, F])
  extends FirstOrderOptimizer[T, F] {
  type History = T

  def initialHistory(lossFunction: DifferentiableFunction[T, F], initialParams: T): History = {
    // TODO
    initialParams
  }

  def updateHistory(
      position: T,
      gradient: T,
      value: F,
      state: State): T = {
    // TODO: do we need to keep a history for gradient descent?
    gradient
  }

  def takeStep(position: T, stepDirection: T, stepSize: F): T = {
    position - (stepDirection :* stepSize)
  }

  /**
   * Constant step size for now.
   */
  def chooseStepSize(lossFunction: DifferentiableFunction[T, F], direction: T, state: State): F = {
    stepSize
  }

  /**
   * The step direction for gradient descent is simply the gradient.
   */
  def chooseDescentDirection(state: State): T = {
    state.gradient
  }
}
