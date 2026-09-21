import type { AssignmentSpec, MultipleChoiceQuestionSpec } from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { softmaxNotebookLab } from './classificationCodeLabQuestions'

export interface SoftmaxVisualDataset {
  id: string
  kind: 'softmaxVisual'
  mode: 'probabilities' | 'explorer' | 'boundary' | 'loss' | 'updates'
  caption: string
}
export const softmaxVisualDatasets: Record<string, SoftmaxVisualDataset> = {
  softmaxProbabilities: {id:'softmaxProbabilities',kind:'softmaxVisual',mode:'probabilities',caption:'Each panel describes one observation. Bars are probabilities of mutually exclusive classes A, B, and C; they sum to 1 within a panel.'},
  softmaxExplorer: {id:'softmaxExplorer',kind:'softmaxVisual',mode:'explorer',caption:'For one fixed observation, the original class scores are (1.2, 0.6, −0.4). Adjust their common shift and positive spread multiplier. The probability bars are recomputed with softmax.'},
  softmaxBoundary: {id:'softmaxBoundary',kind:'softmaxVisual',mode:'boundary',caption:'Scores depend on two input features: score A = x₁, score B = x₂, score C = 0. The dashed diagonal marks equal A and B scores. P = (1, 1), Q = (−1, −1).'},
  softmaxLoss: {id:'softmaxLoss',kind:'softmaxVisual',mode:'loss',caption:'The observed class is B. U and V are two models’ predictions for this same observation. Class order is always A, B, C.'},
  softmaxUpdates: {id:'softmaxUpdates',kind:'softmaxVisual',mode:'updates',caption:'The observed class is B, but the model gives probabilities (0.50, 0.30, 0.20). Treat the three scores as adjustable values for this one observation.'},
}
const option = (id: string, title: string) => ({id,title,description:''})
function question(input: Omit<MultipleChoiceQuestionSpec,'kind'|'validator'|'hintSchedule'>): MultipleChoiceQuestionSpec {
  return {...input,kind:'multipleChoice',hintSchedule:[2,4],validator:validateMultipleChoiceSelections(Object.fromEntries(input.parts.map(p=>[p.id,p.correctOptionId])),'Compare the displayed scores, probabilities, and stated prediction rule.')}
}

export const softmaxAssignment: AssignmentSpec = {
  id:'softmax',version:2,displayNumber:10,published:true,title:'Softmax',topic:'Multiclass Classification',
  description:'Explore competing class probabilities, score shifts, multiclass boundaries, and cross-entropy learning before tracing the notebook implementation.',
  questions:[
    question({
      id:'softmax-competing-classes',title:'Choose Among Competing Classes',
      prompt:'A multiclass classifier predicts the class with the greatest probability. It does not require that probability to exceed 0.5.',
      instructions:'Use the two probability panels. Each observation belongs to exactly one of the three classes. This is not three independent yes/no decisions.',
      datasetId:'softmaxProbabilities',
      parts:[{id:'prediction',prompt:'Which class does the model predict for P, and what probability does it assign to that class?',correctOptionId:'A38',options:[
        option('none','No class, because every probability is below 0.5.'),option('A38','Class A, with probability 0.38.'),option('all','All three classes, because every probability is positive.'),
      ]},{id:'meaning',prompt:'For Q, what is the model’s probability that the class is not B?',correctOptionId:'35',options:[
        option('25','0.25'),option('65','0.65'),option('35','0.35'),
      ]}],
      hints:['Choose the tallest bar for one observation, not the tallest bar across observations.','Not B combines the mutually exclusive alternatives A and C.'],
      reveal:{explanation:'P is classified as A even though 0.38 is less than 0.5. For Q, the probability of not B is 0.10 + 0.25 = 0.35. Softmax probabilities compete within each observation and sum to 1.'},
    }),
    question({
      id:'softmax-score-transformations',title:'Change Scores, Watch Probabilities',
      prompt:'Use the controls to distinguish an overall score offset from a change in the gaps between scores.',
      instructions:'First keep the spread multiplier at 1 and vary the common shift. Then set shift to 0 and compare spread 1 with spread 2. These controls explore fixed scores; they do not retrain a model.',
      datasetId:'softmaxExplorer',
      parts:[{id:'shift',prompt:'Adding the same number to all three scores has what effect?',correctOptionId:'unchanged',options:[
        option('increase','All three probabilities increase.'),option('unchanged','All three probabilities stay the same.'),option('uniform','The probabilities become equal.'),
      ]},{id:'spread',prompt:'For these scores, increasing the spread multiplier from 1 to 2 does what?',correctOptionId:'sharper',options:[
        option('switch','Switches the predicted class from A to B.'),option('flatter','Keeps A as the prediction but lowers its probability.'),option('sharper','Keeps A as the prediction and raises its probability.'),
      ]}],
      hints:['A common offset changes absolute scores but not their differences.','With a positive multiplier, the highest score remains highest; examine how much probability the lower-scoring classes lose.'],
      reveal:{explanation:'A common shift multiplies every exponential by the same factor, which cancels in normalization. A larger positive spread preserves score ranking but makes this distribution more concentrated on A. Greater confidence is not evidence that the prediction is more accurate.'},
    }),
    question({
      id:'softmax-boundary-competition',title:'Which Score Tie Is a Decision Boundary?',
      prompt:'Two equal class scores form a decision boundary between those classes only where neither is beaten by another class.',
      instructions:'Inspect points P and Q on the A–B score-tie line. Here an A–B decision boundary means A and B tie for the highest score. An exact tie need not be assigned a class to answer this question.',
      datasetId:'softmaxBoundary',
      parts:[{id:'boundary',prompt:'Which marked point lies on an actual A–B decision boundary?',correctOptionId:'P',options:[
        option('both','Both P and Q.'),option('Q','Only Q.'),option('P','Only P.'),option('neither','Neither point.'),
      ]},{id:'Q',prompt:'What class is predicted at Q?',correctOptionId:'C',options:[option('A','A'),option('B','B'),option('C','C')]}],
      hints:['At P the A and B scores are both 1; at Q they are both −1. Compare each pair with C’s score.','Softmax preserves the ordering of scores, so the largest score also gives the largest probability.'],
      reveal:{explanation:'At P, scores are (1, 1, 0), so A and B tie for the lead. At Q, scores are (−1, −1, 0), so C wins. The negative part of the A–B score-tie line runs inside C’s region, not between the A and B regions.'},
    }),
    question({
      id:'softmax-target-loss',title:'Which Probability Does Cross-Entropy Use?',
      prompt:'Categorical cross-entropy for one observation is −ln(probability assigned to the observed class).',
      instructions:'The correct class is B in both panels. The logarithm is natural. You can compare the losses without evaluating logarithms numerically.',
      datasetId:'softmaxLoss',
      parts:[{id:'target',prompt:'Which one-hot target represents class B in the displayed class order?',correctOptionId:'B',options:[option('A','(1, 0, 0)'),option('B','(0, 1, 0)'),option('C','(0, 0, 1)')]},
      {id:'loss',prompt:'Which model has lower cross-entropy on this observation?',correctOptionId:'same',options:[
        option('U','U, because its largest incorrect-class probability is smaller.'),option('same','They tie: both assign probability 0.60 to B.'),option('V','V, because it assigns less probability to C.'),
      ]}],
      hints:['One-hot targets mark the observed class, not the model’s most confident prediction.','Redistributing probability between incorrect classes does not change the probability assigned to B here.'],
      reveal:{explanation:'The target is (0, 1, 0), so the loss selects the B probability. Both losses are −ln(0.60), approximately 0.511. The probabilities of incorrect classes affect normalization, but once B’s probability is fixed, their redistribution leaves this observation’s loss unchanged.'},
    }),
    question({
      id:'softmax-learning-signal',title:'Turn a Probability Error into a Learning Signal',
      prompt:'For cross-entropy with softmax, the derivative with respect to each class score is its predicted probability minus its one-hot target.',
      instructions:'Starting from probabilities (0.50, 0.30, 0.20) for classes A, B, C and observed class B, consider a gradient-descent step with a small positive learning rate on the three scores themselves. Parameter updates in a shared model combine signals across observations.',
      datasetId:'softmaxUpdates',
      parts:[{id:'directions',prompt:'Which changes to the scores does this gradient step make?',correctOptionId:'raise-B',options:[
        option('raise-all','Raise all three scores equally.'),option('raise-A','Raise A’s score and lower B’s and C’s scores.'),option('raise-B','Raise B’s score and lower A’s and C’s scores.'),
      ]},{id:'stop',prompt:'If B later has the highest probability but its probability is still below 1, is this observation’s score gradient necessarily zero?',correctOptionId:'no',options:[
        option('yes','Yes. Training stops pushing once the predicted class is correct.'),option('no','No. Cross-entropy still pushes probability toward the observed class.'),option('threshold','It is zero as soon as B’s probability exceeds 0.5.'),
      ]}],
      hints:['Subtract target (0, 1, 0) from probabilities (0.50, 0.30, 0.20), then remember that gradient descent subtracts the result.','Correct classification only depends on the largest probability; cross-entropy also depends on its value.'],
      reveal:{explanation:'The score derivatives are (0.50, −0.70, 0.20), so a small descent step lowers A and C and raises B. Even when B already ranks first, a probability below 1 still produces a learning signal. This is why unchanged accuracy does not imply unchanged cross-entropy.'},
    }),
    softmaxNotebookLab,
  ],
}
