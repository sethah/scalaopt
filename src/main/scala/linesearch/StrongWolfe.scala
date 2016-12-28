package linesearch

import loss.DifferentiableFunction
import spire.algebra.InnerProductSpace

class StrongWolfe[T](implicit space: InnerProductSpace[T, Double]) extends LineSearch[T] {
  import space._
  import StrongWolfe._

  // TODO: make configurable?
  val c1 = 1e-4
  val c2 = 0.9

  private val maxLineSearchIter = 10
  private val maxZoomIter = 10

  private def phi(alpha: Double, f: DifferentiableFunction[Double, Double]): Bracket = {
    val (phiVal, phiPrime) = f.compute(alpha)
    Bracket(alpha, phiVal, phiPrime)
  }

  def optimize(f: DifferentiableFunction[Double, Double], initialGuess: Double): Double = {
    // TODO
    var currentAlpha = initialGuess
    val phiZero = phi(0.0, f)
    var left = phiZero

    // TODO: rewrite using takeWhile or something similar
    for (i <- 0 until maxLineSearchIter) {
      val right = phi(currentAlpha, f)
      if (right.phi.isInfinite() || right.phi.isNaN()) {
        currentAlpha /= 2.0
      } else {
        // check if any of the three conditions is met, otherwise increase alpha and repeat
        if ((right.phi > phiZero.phi + c1 * currentAlpha * phiZero.phiPrime) ||
          (right.phi >= left.phi && i > 0)) {

        }
      }


    }

  }

  private def interpolate(left: Bracket, right: Bracket): Double = {
    // compute the alpha that minimizes a cubic interpolation between
    // left and right brackets according to Nocedal and Wright, p. 59
    val d1 = left.phiPrime + right.phiPrime - 3.0 *
      (left.phi - right.phi) / (left.alpha - right.alpha)
    val d2 = math.sqrt(d1 * d1 - left.phiPrime * right.phiPrime)
    val nextAlpha = right.alpha - (right.alpha - left.alpha) * (right.phiPrime + d2 - d1) /
      (right.phiPrime - left.phiPrime + 2 * d2)

    // A safeguard, described by Nocedal and Wright p. 58, which ensures we make
    // progress on each iteration.
    val intervalLength = right.alpha - left.alpha
    val leftBound = left.alpha + 0.1 * intervalLength
    val rightBound = left.alpha + 0.9 * intervalLength
    if (nextAlpha < leftBound) {
      leftBound
    } else if (nextAlpha > rightBound) {
      rightBound
    } else {
      nextAlpha
    }
  }

  private def zoom(left: Bracket, right: Bracket): Double = {
    val nextAlpha = interpolate(left, right)

  }

}

object StrongWolfe {
  case class Bracket(alpha: Double, phi: Double, phiPrime: Double)

  def decreaseCondition(phiZero: Bracket, currentPhi: Bracket, c1: Double): Boolean = {
    currentPhi.phi <= phiZero.phi + c1 * currentPhi.alpha * phiZero.phiPrime
  }

  def curvatureCondition(phiZero: Bracket, currentPhi: Bracket, c2: Double): Boolean = {
    math.abs(currentPhi.phiPrime) <= c2 * math.abs(phiZero.phiPrime)
  }

}
