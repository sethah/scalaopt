package linesearch

import loss.DifferentiableFunction
import spire.implicits._
import spire.algebra._

trait LineSearch {

  def optimize(f: DifferentiableFunction[Double, Double], initialGuess: Double): Double

}

//class BacktrackingLineSearch[T](implicit space: InnerProductSpace[T, Double])
//  extends LineSearch[T] {
//
//  import space._
//
//  def optimize(f: DifferentiableFunction[Double, Double], initialGuess: Double): Double = {
//    // somewhat of a magic number
//    val beta = 0.8
//
//    val dirNorm = direction dot direction
//    var alpha = space.scalar.one
//    while (f(x + (alpha *: direction)) > (f(x) - dirNorm * alpha / 2.0)) {
//      alpha = alpha * beta
//    }
//    alpha
//  }
//}

