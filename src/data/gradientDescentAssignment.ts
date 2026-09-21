import type { AssignmentSpec, GradientDescentQuestionSpec, MultipleChoiceQuestionSpec } from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { descentSteps, descentTargetLoss, descentTrajectory, isDescentLearningRate } from '../lib/gradientDescent'
import { gradientDescentNotebookLab } from './regressionCodeLabQuestions'

function question(input: Omit<MultipleChoiceQuestionSpec, 'kind' | 'validator' | 'hintSchedule'>): MultipleChoiceQuestionSpec {
  return { ...input, kind: 'multipleChoice', hintSchedule: [2, 4],
    validator: validateMultipleChoiceSelections(Object.fromEntries(input.parts.map(p => [p.id, p.correctOptionId])), 'Use the plotted loss and the update rule to reconsider your selections.') }
}
const option = (id: string, title: string) => ({ id, title, description: '' })

export const learningRateQuestion: GradientDescentQuestionSpec = {
  id: 'gradient-learning-rate', kind: 'gradientDescent', initialLearningRate: 0.01,
  title: 'Find a Learning Rate That Converges',
  prompt: `Adjust the learning rate so that ${descentSteps} gradient steps bring the loss to ${descentTargetLoss.toFixed(2)} or below.`,
  instructions: 'The contour map uses the same loss as Question 2. Every slider change restarts from S = (−2, 1) and runs exactly 20 simultaneous updates. Use the step viewer to inspect the path. Choose one fixed learning rate, then check it; several rates work.',
  hintSchedule: [2, 4], hints: [
    'A tiny learning rate may move downhill but make too little progress within the step budget.',
    'If the path bounces farther and farther across the valley, lower the learning rate. Try a value between the slow and unstable cases.',
  ],
  validator: submission => {
    const rate = submission && typeof submission === 'object' && 'learningRate' in submission ? submission.learningRate : null
    if (!isDescentLearningRate(rate)) return { correct: false, message: 'Choose a learning rate from 0.01 to 0.30 in steps of 0.01.' }
    const trajectory = descentTrajectory(rate)
    const loss = trajectory[descentSteps].loss
    const correct = loss <= descentTargetLoss + 1e-10
    return { correct, message: correct
      ? 'Good work! This fixed learning rate reaches the target within 20 steps.'
      : loss > trajectory[0].loss ? 'The final loss exceeds the starting loss. Reduce the learning rate and inspect the path.'
        : 'The loss is still above the target after 20 steps. Compare a different rate and watch how quickly the path settles.' }
  },
  reveal: { explanation: 'Learning rate controls how far each step travels along the negative gradient. A small rate can be stable but slow; a large rate can repeatedly overshoot and diverge. Crossing the valley is not itself a failure: shrinking oscillations can still converge. The global minimum of this bowl is 1 at (1, −1).' },
}

export const gradientDescentAssignment: AssignmentSpec = {
  id: 'gradient-descent', version: 2, displayNumber: 6, published: true,
  title: 'Gradient Descent', topic: 'Optimization',
  description: 'Read loss landscapes, follow downhill directions, tune a learning rate, and distinguish convergence from getting stuck before tracing the notebook implementation.',
  questions: [
    question({
      id: 'gradient-downhill', title: 'Read the Slope, Choose the Direction',
      prompt: 'The horizontal axis is a model parameter w; the vertical axis is the loss, not a data feature or prediction.',
      instructions: 'Gradient descent subtracts learning rate × slope from w. Use the curve and tangent lines to reason about a small positive learning rate; no derivative calculation is required.',
      datasetId: 'gradientSlopes',
      parts: [{ id: 'directions', prompt: 'Which way should w move from A and from C to decrease the loss?', correctOptionId: 'toward', options: [
        option('both-right', 'Right from both A and C.'), option('toward', 'Right from A; left from C.'), option('away', 'Left from A; right from C.'), option('both-left', 'Left from both A and C.'),
      ] }, { id: 'distance', prompt: 'With the same learning rate, which point produces the larger first change in w: A or C?', correctOptionId: 'A', options: [
        option('same', 'The same change: learning rate fixes the step length.'), option('C', 'C, because its w-coordinate is larger.'), option('A', 'A, because its slope has greater magnitude.'),
      ] }],
      hints: ['At A the curve falls as w increases; at C it rises as w increases.', 'The size of the parameter change is learning rate × absolute slope.'],
      reveal: { explanation: 'The slope is negative at A and positive at C, so subtracting it moves right from A and left from C. A has the steeper tangent and therefore the larger step for the same learning rate. The learning rate scales the gradient; it is not a fixed distance traveled.' },
    }),
    question({
      id: 'gradient-contours', title: 'Find Downhill on a Contour Map',
      prompt: 'Each contour joins parameter pairs with the same loss. Lower contour labels mean better model fits.',
      instructions: 'Both axes are model parameters, a and b, drawn with equal unit scales. The four arrows from S have equal length. Choose the direction of greatest initial loss decrease, not the endpoint with the lowest loss after a long jump.',
      datasetId: 'gradientContours',
      parts: [{ id: 'direction', prompt: 'Which arrow points along the negative gradient at S?', correctOptionId: 'D', options: ['A','B','C','D'].map(id => option(id, `Arrow ${id}`)) },
      { id: 'reason', prompt: 'Why doesn’t the steepest downhill direction necessarily point straight at the center of the ellipses?', correctOptionId: 'local-slope', options: [
        option('random', 'The gradient direction is randomly chosen at each step.'),
        option('local-slope', 'The gradient follows local steepness, perpendicular to a contour; the loss bends more sharply in one direction.'),
        option('center', 'It must point at the center; the arrow diagram cannot be correct.'),
      ] }],
      hints: ['A downhill arrow crosses contours toward smaller labels. The steepest one crosses the nearby contour at a right angle.', 'Heading directly toward the minimizer and heading in the locally steepest direction coincide for circular contours, not for every ellipse.'],
      reveal: { explanation: 'At S = (−2, 1), the gradient is (−6, 16), so the negative gradient points along (6, −16): arrow D. Arrow A aims directly at the minimizer but is not the steepest initial descent direction. B points uphill; C is tangent to the contour.' },
    }),
    learningRateQuestion,
    question({
      id: 'gradient-convergence', title: 'Diagnose the Optimization Run',
      prompt: 'Three full-gradient runs use the same loss and starting point, but different fixed learning rates.',
      instructions: 'Compare the loss curves and the numerical checkpoints. These are training losses across optimization steps, not training-versus-validation curves. Then consider why scaling the input features can help optimization.',
      datasetId: 'gradientRuns',
      parts: [{ id: 'diagnosis', prompt: 'Which diagnosis fits the three runs?', correctOptionId: 'slow-stable-unstable', options: [
        option('slow-stable-unstable', 'A: stable but slow; B: near the minimum; C: unstable, so more steps alone will not fix it.'),
        option('all-more', 'All three just need more steps with the same learning rate.'),
        option('overfit', 'C is overfitting because its training loss increases.'),
      ] }, { id: 'scaling', prompt: 'When very different feature scales create a narrow loss valley, how can standardizing the features help?', correctOptionId: 'balance', options: [
        option('lower-best', 'It guarantees a lower minimum training error for the same unrestricted linear model.'),
        option('zero', 'It makes the gradient zero at the starting point.'),
        option('balance', 'It can reduce the imbalance between parameter directions, so one learning rate makes useful progress without excessive zigzagging.'),
      ] }],
      hints: ['A keeps decreasing, B settles near 1, and C eventually grows. Distinguish slow progress from instability.', 'Scaling changes the coordinates of the optimization problem, not which lines an unrestricted linear model can represent.'],
      reveal: { explanation: 'A needs a larger stable rate or more steps; B has essentially converged; C needs a smaller rate. Standardization can improve the geometry for gradient descent, but does not guarantee perfect conditioning. For the same unregularized linear least-squares problem, converged gradient descent and the normal equations minimize the same objective; scaling does not create a better possible fit.' },
    }),
    question({
      id: 'gradient-local-minima', title: 'Does Stopping Mean You Found the Best Fit?',
      prompt: 'This loss landscape has two valleys. It is deliberately unlike the single convex bowl of linear least squares.',
      instructions: 'Assume exact gradients and sufficiently small steps that do not jump across the hill. A and C are valley bottoms; B is the hilltop, with exactly zero slope.',
      datasetId: 'gradientMinima',
      parts: [{ id: 'basin', prompt: 'Starting at S, where will gradient descent settle?', correctOptionId: 'A', options: [
        option('C', 'C, because gradient descent always finds the lowest valley.'), option('B', 'B, because it lies between the valleys.'), option('A', 'A, even though C has a lower loss.'),
      ] }, { id: 'stationary', prompt: 'If initialized exactly at B, what happens under the ordinary gradient update?', correctOptionId: 'stay', options: [
        option('random', 'It randomly chooses one of the valleys.'), option('stay', 'It stays at B: zero gradient does not, by itself, prove a minimum.'), option('global', 'It moves straight to C because C has the lowest loss.'),
      ] }],
      hints: ['From S, follow the curve downhill using small steps. Reaching the other valley would require going uphill first.', 'The update subtracts learning rate × gradient. What happens when the gradient is exactly zero?'],
      reveal: { explanation: 'Small steps from S lead to the local minimum A, not the global minimum C. At B the exact gradient is zero, so the ordinary update cannot move despite B being a local maximum. Nonconvex models can depend on initialization; a small gradient alone is not proof of a globally best solution. In contrast, every local minimum of a convex linear least-squares loss is global.' },
    }),
    gradientDescentNotebookLab,
  ],
}
