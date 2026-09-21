import { neuralFoundationsAssignments } from './neuralFoundationsAssignments'
import { sequenceFoundationsAssignments } from './sequenceFoundationsAssignments'
import { transformerInterpretabilityAssignments } from './transformerInterpretabilityAssignments'

/** Homework 11–25; the existing 0–10 assignment versions remain unchanged. */
export const advancedAssignments = [
  ...neuralFoundationsAssignments,
  ...sequenceFoundationsAssignments,
  ...transformerInterpretabilityAssignments,
]

export const advancedNotebookFilenames: Record<number, string> = {
  11: '11MLPs.ipynb',
  12: '12AutoGrad.ipynb',
  13: '13PyTorch.ipynb',
  14: '14Optimizers.ipynb',
  15: '15Convolutions.ipynb',
  16: '16CNNs.ipynb',
  17: '17AutoEncoders.ipynb',
  18: '18Tokenization.ipynb',
  19: '19Attention.ipynb',
  20: '20CausalAttention.ipynb',
  21: '21PicoGPT.ipynb',
  22: '22TrainingPicoGPT.ipynb',
  23: '23TransformerActivations.ipynb',
  24: '24SparseAutoencoders.ipynb',
  25: '25FeatureSteering.ipynb',
}
