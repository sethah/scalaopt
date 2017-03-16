package optimization

import loss.DifferentiableFunction
import spire.algebra.{Field, InnerProductSpace}
import spire.implicits._

/**
 *
 * @tparam T The type of parameters to be optimized.
 * @tparam F The type of loss function.
 */
trait Optimizer[T, F <: Function1[T, _]] {

  def optimize(lossFunction: F, initialParameters: T): T

}

trait FirstOrderOptimizer[T, F] extends Optimizer[T, DifferentiableFunction[T, F]] {

  // TODO: make an `IterativeOptimizer` parent trait?

  /** An abstract type alias for the history required to be tracked in the subclass optimizers. */
  type History

  type State = FirstOrderOptimizerState[T, F, History]

  /*
  Should we make a one-size-fits-all State for first order optimizers? The reason to do this is
  that we can implement some methods here that construct and return that state. If we make the state
  generic like a trait, then how can we construct and return it here? The reason not to is that
  maybe some things don't apply to all subclasses - like gd doesn't need a history...
   */

  def converged(state: State): Boolean

  def optimize(lossFunction: DifferentiableFunction[T, F], initialParameters: T): T = {

    val allIterations =
      infiniteIterations(lossFunction, initialState(lossFunction, initialParameters))
        .takeWhile(!converged(_))
    var lastIteration: State = null
    while (allIterations.hasNext) {
      lastIteration = allIterations.next()
    }
    lastIteration.params
  }

  def initialState(
                    lossFunction: DifferentiableFunction[T, F],
                    initialParams: T): State = {
    val (firstLoss, firstGradient) = lossFunction.compute(initialParams)
    FirstOrderOptimizerState(initialParams, 0, firstLoss, firstGradient,
      initialHistory(lossFunction, initialParams))
  }

  /**
   * For iterative optimizers this represents an infinite number of optimization steps.
   *
   * In practice, we take from this iterator only until convergence is achieved.
   */
  def infiniteIterations(
      lossFunction: DifferentiableFunction[T, F],
      start: State): Iterator[State] = {
    Iterator.iterate(start)(iterateOnce(lossFunction))
  }

  /**
   * Step into the next state from the previous state. This code defines the general template for
   * first order optimizers: choose a direction and a step size, update parameters, then compute
   * the loss and gradient of the loss function evaluated at the new parameters.
   *
   * @param lossFunction The differentiable loss function.
   * @param state The current optimization state.
   * @return The next optimization state.
   */
  private def iterateOnce(lossFunction: DifferentiableFunction[T, F])(state: State): State = {
    val direction = chooseDescentDirection(state)
    val stepSize = chooseStepSize(lossFunction, direction, state)
    val nextPosition = takeStep(state.params, direction, stepSize)
    val (nextLoss, nextGradient) = lossFunction.compute(nextPosition)
    val nextHistory = updateHistory(nextPosition, nextGradient, nextLoss, state)
    FirstOrderOptimizerState(nextPosition, state.iter + 1, nextLoss, nextGradient, nextHistory)
  }

  def chooseDescentDirection(state: State): T

  def chooseStepSize(lossFunction: DifferentiableFunction[T, F], direction: T, state: State): F

  /**
   * Return the next set of parameters from the current parameters.
   */
  def takeStep(position: T, stepDirection: T, stepSize: F): T

  def initialHistory(lossFunction: DifferentiableFunction[T, F], initialParams: T): History

  /**
   * Update the history for the optimizer. For LBFGS, this might be the information required to
   * compute the approximate inverse Hessian matrix.
   */
  def updateHistory(position: T, gradient: T, value: F, state: State): History

}


/**
 * Data structure holding pertinent information about the optimizer state.
 *
 * @tparam T The type of parameters being optimized.
 */
trait OptimizerState[+T] {

  def iter: Int

  def params: T

}

case class FirstOrderOptimizerState[+T, +F, +History](
    params: T,
    iter: Int,
    value: F,
    gradient: T,
    history: History) extends OptimizerState[T]

/**
 * Class representing stopping criteria for an iterative optimizer.
 *
 * @tparam State the type of optimization state required to evaluate the stopping conditions.
 */
trait StoppingCriteria[-State <: OptimizerState[_]] {

  def apply(state: State): Boolean

}

