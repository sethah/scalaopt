package optimization

import linesearch.StrongWolfe
import loss.DifferentiableFunction
import breeze.optimize.{LBFGS => BLBFGS}
import spire.implicits._
import spire.algebra.{NRoot, Order, InnerProductSpace, Field}

import scala.reflect.ClassTag

/**
 * Classic LBFGS algorithm.
 *
 * @param m The number of iterations to keep history for.
 * @tparam T Type of parameters being optimized.
 * @tparam F The parameter's scalar type.
 */
class VLBFGS[T: ClassTag, F: Field: Order: NRoot: ClassTag](m: Int)(implicit space: InnerProductSpace[T, F])
  extends FirstOrderOptimizer[T, F] {

  /** (deltaPosition, deltaGradient) */
  type History = VLBFGS.History[T, F]

  def initialHistory(lossFunction: DifferentiableFunction[T, F], initialParams: T): History = {
    new VLBFGS.History[T, F](m, 0, Array.ofDim[F](2 * m + 1, 2 * m + 1), new Array[T](m),
      new Array[T](m))
  }

  def converged(state: State): Boolean = {
    state.iter > 4
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
    val newDeltaGrad = gradient - state.gradient
    val newDeltaPosition = position - state.params
    state.history.update(gradient, newDeltaPosition, newDeltaGrad)
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
//    println(state.params, state.gradient, direction)
    val ls = new StrongWolfe()
    val lineSearchFunction = new DifferentiableFunction[Double, Double] {
      def apply(x: Double): Double = {
        compute(x)._1
      }

      def gradientAt(x: Double): Double = compute(x)._2

      def compute(x: Double): (Double, Double) = {
        val step = space.scalar.fromDouble(x)
        val (f, grad) = lossFunction.compute(state.params + direction :* step)
        val thisGrad = grad dot direction
        (f, thisGrad) match {
          case (d1: Double, d2: Double) => (d1, d2)
          case _ => throw new Exception("only double supported right now")
        }
      }
    }
    // TODO: why does breeze do this?
    val tmp1 = implicitly[NRoot[F]].sqrt((direction dot direction))
    val tmp2 = space.scalar.one
    val tmp3 = space.scalar.div(tmp2, tmp1)
    val initialAlpha = if (state.iter == 0) tmp3 else space.scalar.one
    initialAlpha match {
      case d: Double =>
        space.scalar.fromDouble(ls.optimize(lineSearchFunction, d))
      case _ => throw new Exception("only double supported right now")
    }
  }

  /**
   * The descent direction is the Hessian multiplied with the gradient.
   *
   * Use the position and gradient history to approximate the Hessian and compute H^(-1)^ g.
   * TODO: explain the code that produces this multiplication.
   */
  def chooseDescentDirection(state: State): T = {
//    LBFGS.multiplyApproxInverseHessian[T, F](state.gradient, state.history._1, state.history._2)
    state.history.computeDirection(state.gradient)
  }
}

object VLBFGS {
  case class History[T: ClassTag, F: Field: ClassTag](
                         m: Int,
                         k: Int,
                         innerProducts: Array[Array[F]],
                         posDeltas: Array[T],
                         gradDeltas: Array[T])(implicit space: InnerProductSpace[T, F]) {
    private val numBasisVectors = 2 * m + 1
    private val field = implicitly[Field[F]]

    def getBasisVector(idx: Int, pds: Array[T], gds: Array[T], gradient: T): T = {
      if (idx < m) pds(idx)
      else if (idx < 2 * m) gds(idx - m)
      else if (idx == 2 * m) gradient
      else throw new IndexOutOfBoundsException(s"Basis vector index was invalid: $idx")
    }

    private def shiftArr(elem: T, arr: Array[T]): Array[T] = {
      // TODO: using a seq or list or whatever is probably fine here since m is small
      val newArr = new Array[T](arr.length)
      for (i <- arr.length - 2 to 0 by (-1)) {
        newArr(i + 1) = arr(i)
      }
      newArr(0) = elem
      newArr
    }

    private def shiftMatrix(matrix: Array[Array[F]]): Array[Array[F]] = {
      val newMatrix = Array.ofDim[F](numBasisVectors, numBasisVectors)
      for (i <- numBasisVectors - 2 to 0 by (-1); j <- numBasisVectors - 2 to 0 by (-1)) {
        newMatrix(i + 1)(j + 1) = matrix(i)(j)
      }
      newMatrix
    }

    def update(gradient: T, posDelta: T, gradDelta: T): History[T, F] = {
      // add in the new history, shift out the old, and compute new dot products
      val newPosDeltas = shiftArr(posDelta, posDeltas)
      val newGradDeltas = shiftArr(gradDelta, gradDeltas)
      val shiftedInnerProducts = shiftMatrix(innerProducts)
      val indices = (0 to 2 * m).flatMap { i =>
        List((0, i), (math.min(m, i), math.max(m, i)), (i, 2 * m))
      }.toSet
      indices.par.foreach {
        case (i, j) =>
          val v1 = getBasisVector(i, newPosDeltas, newGradDeltas, gradient)
          val v2 = getBasisVector(j, newPosDeltas, newGradDeltas, gradient)
          // TODO: fix this nonsense
          if (v1 != null && v2 != null) {
            val dotProd = space.dot(v1, v2)
            shiftedInnerProducts(i)(j) = dotProd
            shiftedInnerProducts(j)(i) = dotProd
          }
      }
      History(m, k + 1, shiftedInnerProducts, newPosDeltas, newGradDeltas)
    }

    /**
     * Compute sum_j=1^2m+1^ delta_j * b_j
     * @param gradient
     * @return
     */
    def computeDirection(gradient: T): T = {
      val deltas = basisCoefficients
      var dir = deltas(numBasisVectors - 1) *: gradient
      var i = 0
      while (i < k) {
        dir += deltas(i) *: posDeltas(i)
        dir += deltas(i + m) *: gradDeltas(i)
        i += 1
      }
      dir
    }

    /**
     * This should not be called until k > 1
     * @return
     */
    lazy val basisCoefficients: Array[F] = {
      val deltas = Array.tabulate(numBasisVectors) { i =>
        if (i == numBasisVectors - 1) field.negate(field.one) else field.zero
      }
      if (k == 0) {
        deltas
      } else {
        val alphas = new Array[F](m)
        for (i <- 0 until k) {
          val rho = innerProducts(i)(i + m)
          val num = (0 until numBasisVectors).foldLeft(field.fromDouble(0.0)) { (acc, j) =>
            acc + innerProducts(i)(j) * deltas(j)
          }
          alphas(i) = num / rho
          deltas(m + i) = deltas(m + i) - alphas(i)
        }

        val diag = innerProducts(0)(m) / innerProducts(m)(m)
        (0 until numBasisVectors).foreach { j =>
          deltas(j) = deltas(j) * diag
        }

        for(i <- (k - 1) to 0 by (-1)) {
          // beta = yi * p / (si * yi)
          val betaDenom = innerProducts(i)(m + i)
          val betaNum = (0 until numBasisVectors).foldLeft(field.zero) { (acc, j) =>
            acc + innerProducts(m + i)(j) * deltas(j)
          }
          val beta = betaNum / betaDenom
          deltas(i) = deltas(i) + (alphas(i) - beta)
        }
        deltas
      }
    }
  }
}

