//package optimization
//
//import loss.DifferentiableFunction
//import spire.algebra.InnerProductSpace
//
//class AdaGrad[T <: Traversable[F], F](
//    stepSize: F,
//    override val stoppingCriteria: (FirstOrderOptimizerState[T, F, _] => Boolean))
//    (implicit space: InnerProductSpace[T, F])
//  extends FirstOrderOptimizer[T, F] {
//
//  type History = Array[Double]
//
//  def chooseDescentDirection(state: State): T = {
//    state.gradient
//  }
//
//  /**
//   * Need to compute the adaptive step size here using the gradient history.
//   *
//   * Shouldn't this return a vector? Need to change abstraction for first order optimizer...
//   *
//   * Or we can do it in `takeStep`
//   *
//   * @param lossFunction
//   * @param direction
//   * @param state
//   * @return
//   */
//  def chooseStepSize(lossFunction: DifferentiableFunction[T, F], direction: T, state: State): F = {
//    stepSize
//  }
//
//  /**
//   * Return the next set of parameters from the current parameters.
//   */
//  def takeStep(position: T, stepDirection: T, stepSize: F): T = {
//    val stepVector = position
//    val epsilon = space.scalar.fromDouble(1e-10)
//    position + space.tim
//  }
//
//  def initialHistory(lossFunction: DifferentiableFunction[T, F], initialParams: T): History
//
//  /**
//   * Update the history for the optimizer. For LBFGS, this might be the information required to
//   * compute the approximate inverse Hessian matrix.
//   */
//  def updateHistory(position: T, gradient: T, value: F, state: State): History
//
//}
