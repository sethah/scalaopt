package util


import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions
import breeze.stats.distributions.Gaussian
import spire.algebra.Field

import scala.reflect.ClassTag

class LinearDataGenerator[F: ClassTag](seed: Long)(implicit field: Field[F]) {

  private val rng = new scala.util.Random(seed)

  // TODO: bake intercept into coefficients

  def generate(fitIntercept: Boolean, numPoints: Int, numFeatures: Int,
               epsMean: Double,
               epsVariance: Double): (DenseMatrix[F], DenseVector[F], DenseVector[F]) = {
    val means = Array.tabulate[Double](numFeatures) { i =>
      rng.nextDouble() - 0.5
    }
    val variances = Array.tabulate[Double](numFeatures) { i =>
      rng.nextDouble() * 2.0
    }
    val coefs = Array.tabulate[Double](numFeatures) { i =>
      rng.nextDouble() - 0.5
    }
    generate(fitIntercept, numPoints, means, variances, coefs, epsMean, epsVariance)
  }

  def generate(
      fitIntercept: Boolean,
      numPoints: Int,
      featureMeans: IndexedSeq[Double],
      featureVariances: IndexedSeq[Double],
      coefs: Array[Double],
      epsMean: Double,
      epsVariance: Double): (DenseMatrix[F], DenseVector[F], DenseVector[F]) = {
    val numFeatures = featureMeans.length
    val interceptCoef = if (fitIntercept) rng.nextDouble() else 0.0
    val labels = new Array[Double](numPoints)
    val data = new Array[Double](numPoints * numFeatures)
    val epsGaussian = new distributions.Gaussian(epsMean, epsVariance)
    val featureGaussians = (0 until numFeatures).map { i =>
      new Gaussian(featureMeans(i), featureVariances(i))
    }
    val coefVector = new DenseVector(coefs)
    (0 until numPoints).foreach { i =>
      val features = new DenseVector(featureGaussians.map(_.sample()).toArray)
      val label = features dot coefVector + interceptCoef + epsGaussian.sample()

      labels(i) = label
      features.foreachPair { case (j, v) =>
        data(numFeatures * i + j) = features.valueAt(j)
      }
    }
    val dataMatrix = new DenseMatrix[F](numPoints, numFeatures, data.map(field.fromDouble))
    (dataMatrix, new DenseVector[F](labels.map(field.fromDouble)),
      new DenseVector[F](coefs.map(field.fromDouble)))
  }

}
