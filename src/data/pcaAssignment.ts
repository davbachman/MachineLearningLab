import type { AssignmentSpec, ValidationResult, Vec2, Vec3 } from '../types'
import { handCalculationDatasets, pca2dDatasets, pca3dDatasets } from './pcaDatasets'
import {
  degrees,
  explainedVarianceForLine,
  explainedVarianceForPlane,
  matrixAlmostEqual,
  normalize2,
  pointsAlmostEqual,
  principalDirectionFromPoints2D,
  signInvariantAngle2,
  signInvariantAngle3,
} from '../lib/pcaMath'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { pcaCodeLab } from './codeLabQuestions'

function validate2DDirection(
  datasetId: keyof typeof pca2dDatasets,
  toleranceDegrees: number,
  minimumVarianceRatio: number,
) {
  const dataset = pca2dDatasets[datasetId]
  const optimalVariance = explainedVarianceForLine(
    dataset.points,
    dataset.answerDirection,
    dataset.scoring,
  )

  return (submission: unknown): ValidationResult => {
    const payload = submission as { direction: Vec2 }
    const angle = degrees(signInvariantAngle2(payload.direction, dataset.answerDirection))
    const varianceRatio =
      explainedVarianceForLine(dataset.points, payload.direction, dataset.scoring) / optimalVariance

    return {
      correct: angle <= toleranceDegrees && varianceRatio >= minimumVarianceRatio,
      message:
        angle <= toleranceDegrees && varianceRatio >= minimumVarianceRatio
          ? 'Correct! Good Job!'
          : dataset.scoring === 'secondMoment'
            ? 'This is the uncentered case. Rotate toward the line that keeps the projected points far from the origin on average, not the line that merely makes them look most spread out.'
            : 'That direction still leaves variance on the table. Rotate toward the longest spread of projected points.',
    }
  }
}

function validatePlane(
  datasetId: keyof typeof pca3dDatasets,
  toleranceDegrees: number,
  minimumVarianceRatio: number,
) {
  const dataset = pca3dDatasets[datasetId]
  const optimalVariance = explainedVarianceForPlane(
    dataset.points,
    dataset.answerFirst,
    dataset.answerSecond,
  )

  return (submission: unknown): ValidationResult => {
    const payload = submission as { first: Vec3; second: Vec3 }
    const firstAngle = degrees(signInvariantAngle3(payload.first, dataset.answerFirst))
    const retainedVarianceRatio =
      explainedVarianceForPlane(dataset.points, payload.first, payload.second) / optimalVariance
    const planeIsGood = retainedVarianceRatio >= minimumVarianceRatio
    const firstDirectionIsGood = firstAngle <= toleranceDegrees

    return {
      correct: firstDirectionIsGood && planeIsGood,
      message:
        firstDirectionIsGood && planeIsGood
          ? 'Correct! Good Job!'
          : planeIsGood
            ? 'You found essentially the right plane, but the red first direction is not aligned with the first principal direction yet.'
            : 'The plane is close, but you can still capture more variance by rotating the first direction and then rolling the second around it.',
    }
  }
}

function validateTableEntry(datasetId: keyof typeof handCalculationDatasets) {
  const dataset = handCalculationDatasets[datasetId]
  const targetDirection = normalize2(dataset.firstDirection)
  const sampleCovariance = dataset.covariance.map((row) =>
    row.map((value) => (value * dataset.rows.length) / (dataset.rows.length - 1)),
  )
  const scatterMatrix = dataset.covariance.map((row) =>
    row.map((value) => value * dataset.rows.length),
  )

  return (submission: unknown): ValidationResult => {
    const payload = submission as {
      step: 'centeredData' | 'covariance' | 'direction'
      values: number[][] | number[]
    }

    if (payload.step === 'centeredData') {
      return {
        correct: matrixAlmostEqual(dataset.centeredRows, payload.values as number[][], 0.05),
        message: 'Check the mean of each column again, then subtract it from every entry.',
      }
    }

    if (payload.step === 'covariance') {
      return {
        correct:
          matrixAlmostEqual(dataset.covariance, payload.values as number[][], 0.05) ||
          matrixAlmostEqual(sampleCovariance, payload.values as number[][], 0.05) ||
          matrixAlmostEqual(scatterMatrix, payload.values as number[][], 0.05),
        message: 'Use the centered data from the previous step to compute the covariance matrix.',
      }
    }

    const rawDirection = payload.values as number[]
    const magnitude = Math.hypot(rawDirection[0] ?? 0, rawDirection[1] ?? 0)
    if (magnitude < 1e-9) {
      return {
        correct: false,
        message: 'Enter a non-zero direction vector for the first principal direction.',
      }
    }

    const normalizedDirection = normalize2([rawDirection[0], rawDirection[1]])
    return {
      correct: pointsAlmostEqual(targetDirection, normalizedDirection, 0.08) || pointsAlmostEqual(
        targetDirection,
        [-normalizedDirection[0], -normalizedDirection[1]],
        0.08,
      ),
      message: 'The first principal direction should line up with the largest eigenvector of the covariance matrix.',
    }
  }
}

export const pcaAssignment: AssignmentSpec = {
  id: 'pca',
  version: 6,
  title: 'Principal Component Analysis',
  topic: 'Dimensionality Reduction',
  description:
    'Work through centered and uncentered PCA, outlier sensitivity, scaling choices, and one hand-calculation problem before exporting your JSON submission.',
  published: true,
  questions: [
    {
      id: 'pca-uncentered',
      kind: 'pca2dLine',
      title: 'Uncentered PCA',
      prompt:
        'Rotate the line until the projected variance is as large as possible. This is the PCA objective.',
      instructions:
        'The line always passes through the origin. Watch the right-hand view with the origin fixed in place: the uncentered PCA direction can look wrong if you are expecting the widest visible spread.',
      datasetId: 'uncentered',
      minimumVarianceRatio: 0.985,
      initialAngleDeg: -18,
      tolerance: 12,
      hintSchedule: [2, 4],
      hints: [
        'Do not re-center the projected points mentally. The origin stays fixed in the middle of the right-hand view.',
        'Because the data are not centered, the direction chosen by uncentered PCA follows large average squared distance from the origin rather than the visually longest spread of the cluster.',
      ],
      validator: validate2DDirection('uncentered', 12, 0.985),
      reveal: {
        explanation:
          'Without centering, the objective is dominated by how far projections sit from the origin. That pulls the chosen line away from the cloud’s natural elongation, which is why centering matters before PCA.',
      },
      successCopy: 'Correct! Good Job!',
    },
    {
      id: 'pca-centered',
      kind: 'pca2dLine',
      title: 'Centered PCA',
      prompt:
        'Rotate the line until the projected variance is as large as possible. This is the PCA objective.',
      datasetId: 'centered',
      minimumVarianceRatio: 0.99,
      initialAngleDeg: 132,
      tolerance: 10,
      hints: [
        'Centering removes the mean offset, so only the shape of the cloud matters now.',
        'The best line should follow the long diagonal direction of the points.',
      ],
      validator: validate2DDirection('centered', 10, 0.99),
      reveal: {
        explanation:
          'Once the mean is removed, PCA finds the direction of largest spread relative to the centered cloud. That matches the long axis you can see directly.',
      },
      successCopy: 'Correct! Good Job!',
    },
    {
      id: 'pca-near-circular',
      kind: 'pca2dLine',
      title: 'Nearly Circular Cloud',
      prompt:
        'This centered cloud is much closer to circular. Rotate the line until the projected points spread out as much as possible.',
      instructions:
        'The best direction still exists, but the answer is less obvious because the two axis spreads are closer together.',
      datasetId: 'nearCircular',
      minimumVarianceRatio: 0.995,
      initialAngleDeg: 86,
      tolerance: 8,
      hintSchedule: [2, 5],
      hints: [
        'Because the cloud is close to circular, small angle changes make smaller variance changes than before.',
        'Look for the slightly longer diameter rather than a dramatic long axis.',
      ],
      validator: validate2DDirection('nearCircular', 8, 0.995),
      reveal: {
        explanation:
          'Even when the cloud looks almost round, PCA still picks the direction with the slightly larger variance. The difference is smaller, so the optimum is harder to see.',
      },
      successCopy: 'Correct! Good Job!',
    },
    {
      id: 'pca-outlier',
      kind: 'pca2dLine',
      title: 'Outlier Sensitivity',
      prompt:
        'Rotate the line until the projected points spread out as much as possible, then compare the revealed answer with the ghost reference.',
      instructions:
        'One extreme point has been added to this centered cloud. PCA still follows variance, even when a single point changes that variance a lot.',
      datasetId: 'outlierSensitive',
      minimumVarianceRatio: 0.985,
      initialAngleDeg: 24,
      tolerance: 12,
      hintSchedule: [2, 4],
      hints: [
        'An outlier can drag the best direction toward itself because it contributes a lot of squared distance.',
        'When the answer is revealed, compare it with the ghost line that shows what the cloud wanted before the outlier was added.',
      ],
      validator: validate2DDirection('outlierSensitive', 12, 0.985),
      reveal: {
        explanation:
          'PCA is sensitive to outliers because variance squares large deviations. The ghost line shows how the dominant direction would look without the extreme point.',
      },
      successCopy: 'Correct! Good Job!',
    },
    {
      id: 'pca-standardize',
      kind: 'multipleChoice',
      title: 'House Prices and Bedrooms',
      prompt:
        'Answer both questions about running PCA on centered house-price and bedroom data.',
      instructions:
        'First predict what centered-but-unscaled PCA will do, then choose how to make the representation sensitive to all features.',
      datasetId: 'housePrices',
      parts: [
        {
          id: 'pca-behavior',
          prompt:
            "Suppose you are predicting a particular house's value. You have information about housing prices in the area and the number of bedrooms for each. If you center the data and run PCA, what will happen?",
          correctOptionId: 'price-dominates',
          options: [
            {
              id: 'equal-weight',
              title: 'PCA will weight price and bedrooms equally',
              description: 'Centering alone does not force equal influence across features.',
            },
            {
              id: 'price-dominates',
              title: 'PCA will mostly follow housing price',
              description: 'Because price is on a much larger numerical scale, its variance dominates the principal direction.',
            },
            {
              id: 'bedrooms-dominate',
              title: 'PCA will mostly follow bedrooms',
              description: 'Bedroom count has smaller numerical scale here, so it will not dominate the covariance.',
            },
          ],
        },
        {
          id: 'pca-fix',
          prompt: 'What can you do so that predictions will be sensitive to all features?',
          correctOptionId: 'standardize',
          options: [
            {
              id: 'recenter',
              title: 'Center the data again',
              description: 'Repeated centering does not change the scale imbalance between price and bedrooms.',
            },
            {
              id: 'change-denominator',
              title: 'Use 1/(n-1) instead of 1/n',
              description: 'Changing the covariance denominator does not fix feature-scale dominance.',
            },
            {
              id: 'standardize',
              title: 'Standardize each feature after centering',
              description: 'Put price and bedrooms on comparable scales before PCA so both can matter.',
            },
          ],
        },
      ],
      tolerance: 0,
      hints: [
        'Centering changes the mean but not the units, so the larger-scale feature still contributes much larger squared values.',
        'To make PCA sensitive to all features, you need a preprocessing step that changes scale, not just location.',
      ],
      validator: validateMultipleChoiceSelections(
        {
          'pca-behavior': 'price-dominates',
          'pca-fix': 'standardize',
        },
        'Centering alone does not stop the larger-scale price feature from dominating. Try again.',
      ),
      reveal: {
        explanation:
          'With centered but unscaled inputs, PCA is driven mostly by house price because its numerical variation is so much larger than bedroom count. Standardizing after centering makes both features contribute on comparable footing.',
      },
      successCopy: 'Correct! Good Job!',
    },
    {
      id: 'pca-3d-plane',
      kind: 'pca3dPlane',
      title: 'A Two-Dimensional PCA Plane',
      prompt:
        'Rotate the first principal direction with your mouse or trackpad, then use the left and right arrow keys to rotate the second direction around it until the projected points spread out as much as possible.',
      instructions:
        'The left panel shows the centered 3D cloud and the candidate projection plane. The right panel shows the orthogonal view of that plane with the projected points.',
      datasetId: 'centered3d',
      minimumVarianceRatio: 0.985,
      initialAzimuthDeg: 34,
      initialElevationDeg: 11,
      initialRollDeg: 35,
      tolerance: 18,
      hintSchedule: [2, 5],
      hints: [
        'Start by rotating the red first-direction arrow toward the longest spread in the point cloud.',
        'Then use the arrow keys to roll the second direction so the right-hand plane view captures as much spread as possible.',
      ],
      validator: validatePlane('centered3d', 18, 0.985),
      reveal: {
        explanation:
          'The best PCA plane is spanned by the top two eigenvectors of the covariance matrix. The right-hand view makes it easier to judge how much of the 3D variance your plane retains.',
      },
      successCopy: 'Correct! Good Job!',
    },
    {
      id: 'pca-hand-calculation',
      kind: 'tableEntry',
      title: 'Work the Numbers by Hand',
      prompt:
        'Center the data, enter the covariance matrix, and then enter a first principal direction for the four 2D points below.',
      instructions:
        'The direction can be any non-zero vector pointing along the first principal direction.',
      datasetId: 'workedExample',
      substeps: ['Center the data', 'Enter the covariance matrix', 'Enter the first principal direction'],
      tolerance: 0.05,
      hintSchedule: [2, 4],
      hints: [
        'First compute the mean of each column, then subtract that mean from every row.',
        'Once the covariance matrix is filled in, the first principal direction points along its largest eigenvector.',
      ],
      validator: validateTableEntry('workedExample'),
      reveal: {
        explanation:
          'Because the centered points all lie on the same diagonal line, the covariance matrix has one dominant direction. Any non-zero multiple of that direction represents the same principal axis.',
      },
      successCopy: 'Correct! Good Job!',
    },
    pcaCodeLab,
  ],
}

export const pcaReferenceDirection = principalDirectionFromPoints2D(
  handCalculationDatasets.workedExample.rows as Vec2[],
  'covariance',
)
