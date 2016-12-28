package loss

import breeze.linalg.operators.{OpMulMatrix, OpMulInner}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.math.{InnerProductModule => BreezeInnerProductModule}
import spire.algebra.{Field, InnerProductSpace}

import scala.reflect.ClassTag

/**
 *
 * @tparam T The type of the function's domain.
 * @tparam F The type of the functions' range.
 */
trait DifferentiableFunction[T, F]  extends Function1[T, F] {

  def apply(x: T): F

  def gradientAt(x: T): T

  def compute(x: T): (F, T)

}

trait InnerProductSpaceSyntax {
  implicit def space[F: Field: ClassTag]
  (implicit bmodule: BreezeInnerProductModule[DenseVector[F], F]): InnerProductSpace[DenseVector[F], F] =
    new InnerProductSpace[DenseVector[F], F] {

      override def scalar: Field[F] = Field[F]

      override def plus(x: DenseVector[F], y: DenseVector[F]): DenseVector[F] = {
        bmodule.addVV(x, y)
      }

      override def timesl(r: F, v: DenseVector[F]): DenseVector[F] = bmodule.mulVS(v, r)

      override def negate(x: DenseVector[F]): DenseVector[F] = x.map(e => implicitly[Field[F]].negate(e))

      override def dot(x: DenseVector[F], y: DenseVector[F]): F = {
        bmodule.dotVV(x, y)
      }

      override def zero: DenseVector[F] = new DenseVector[F](Array.empty[F])
  }
}

class LeastSquaresLossFunction[F](
    data: DenseMatrix[F],
    label: DenseVector[F])
    (implicit mmult: OpMulMatrix.Impl2[DenseMatrix[F], DenseVector[F], DenseVector[F]],
    space: BreezeInnerProductModule[DenseVector[F], F])
  extends DifferentiableFunction[DenseVector[F], F] {
  // TODO: generic working, but why we can't use implicit syntax?

  def apply(x: DenseVector[F]): F = {
      val error = space.subVV(mmult(data, x), label)
      space.dotVV(error, error)
  }

  def gradientAt(x: DenseVector[F]): DenseVector[F] = {
    val error = space.subVV(mmult(data, x), label)
    mmult(data.t, error)
  }

  def compute(x: DenseVector[F]): (F, DenseVector[F]) = {
    val yhat = mmult(data, x)
    val error = space.subVV(yhat, label)
    val (loss, grad) = (space.dotVV(error, error), mmult(data.t, error))
    (loss, grad)
  }
}
