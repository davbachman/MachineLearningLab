import { defineCodeLabQuestion, type CodeLabQuestionSpec, type CodeLabStageSpec } from '../lib/codeLab'

function stage(id: string, title: string, prompt: string, label: string, correct: string, choices: [string, string][], successCopy: string, kind: CodeLabStageSpec['kind'] = 'executionTrace'): CodeLabStageSpec {
  return { id, title, prompt, kind, fields: [{ id: 'answer', label, correctOptionId: correct, options: choices.map(([id, label]) => ({ id, label })) }], successCopy }
}
function lab(input: Omit<CodeLabQuestionSpec, 'kind' | 'language' | 'validator' | 'hintSchedule' | 'variantId'>): CodeLabQuestionSpec {
  return defineCodeLabQuestion({ ...input, kind: 'codeLab', language: 'python', variantId: `${input.id}-v1`, hintSchedule: [2, 4] })
}

const mlps = lab({
  id: 'mlps-notebook-lab', title: 'Notebook Lab: Trace a Three-Layer Network',
  prompt: 'Trace the Linear, ReLU, and Sequential classes from Homework2026/11MLPs.ipynb on a tiny fixed batch.',
  instructions: 'The relevant class implementations below are copied from the notebook. The fixture intentionally overwrites initialized weights and biases, so no answer depends on random draws. Execute the fixture once and then all invocation statements in order. Each checkpoint is independently scored.',
  code: `import numpy as np

class Linear():
    '''Fully connected linear layer class'''
    def __init__(self, input_size, output_size):
        np.random.seed(1) #Don't use in practice! This is just to make sure we all get the same answers
        self.kernel = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size) #Standard initialization of weights
        self.bias = np.zeros(output_size) #Standard initialization of intercept

    def __call__(self,input):
        return input@self.kernel+self.bias

class ReLU():
    '''ReLU layer class'''
    def __init__(self):
        pass #No init function necessary

    def __call__(self,input):
        self.input=input
        return np.maximum(input,0)

class Sequential():
    def __init__(self,layerlist):
        self.layerlist=layerlist

    def __call__(self,input):
        for layer in self.layerlist:
            input=layer(input)
        return input`,
  fixtureTitle: 'Two observations and explicit parameters',
  fixture: `X = np.array([[1., -1.], [0., 2.]])
layer1 = Linear(2, 3)
layer1.kernel = np.array([[1., -2., 0.], [2., 1., -1.]])
layer1.bias = np.array([0., 1., 1.])
layer2 = ReLU()
layer3 = Linear(3, 1)
layer3.kernel = np.array([[1.], [-1.], [2.]])
layer3.bias = np.array([0.5])
network = Sequential([layer1, layer2, layer3])`,
  invocationTitle: 'Run these deterministic forward passes',
  invocation: `preactivation = layer1(X)
activation = layer2(preactivation)
prediction = network(X)
without_relu = Sequential([layer1, layer3])(X)`,
  stages: [
    stage('kernel-shape', '1. Match the matrix dimensions', 'Inspect the fixture and Linear.__call__.', 'layer1.kernel.shape', 'two-three', [['three-two', '(3, 2)'], ['two-three', '(2, 3)'], ['two-two', '(2, 2)']], 'The notebook stores its kernel as input features × output features, so (2, 2) @ (2, 3) produces (2, 3).'),
    stage('affine', '2. Include the bias', 'Trace only the first observation through layer1.', 'preactivation[0].tolist()', 'negative', [['negative', '[-1.0, -2.0, 2.0]'], ['clipped', '[0.0, 0.0, 2.0]'], ['no-bias', '[-1.0, -3.0, 1.0]']], 'The weighted row is [−1, −3, 1]; adding [0, 1, 1] gives [−1, −2, 2].'),
    stage('activation', '3. Rectify one observation', 'Trace the second observation through layer1 and layer2.', 'activation[1].tolist()', 'four-three-zero', [['unclipped', '[4.0, 3.0, -1.0]'], ['four-two-zero', '[4.0, 2.0, 0.0]'], ['four-three-zero', '[4.0, 3.0, 0.0]']], 'The second preactivation row [4, 3, −1] becomes [4, 3, 0]. ReLU changes values but preserves dimensions.'),
    stage('prediction', '4. Finish the network', 'Trace the final dense layer for both observations.', 'prediction.tolist()', 'correct', [['flat', '[4.5, 1.5]'], ['correct', '[[4.5], [1.5]]'], ['bias-missing', '[[4.0], [1.0]]']], 'The final weighted sums are 4 and 1; the 0.5 bias makes a two-row, one-column prediction matrix.'),
    stage('remove-relu', '5. Remove the activation', 'The last invocation uses the same parameters but omits layer2.', 'float(without_relu[0, 0])', 'five-half', [['four-half', '4.5'], ['five-half', '5.5'], ['zero', '0.0']], 'Without ReLU, the negative hidden values contribute: (−1)×1 + (−2)×(−1) + 2×2 + 0.5 = 5.5.', 'diagnoseMutation'),
  ], hints: ['Each output column is a dot product followed by its own bias.', 'ReLU acts before the final layer, not after the final prediction.'],
  reveal: { explanation: 'The notebook composes callable objects in list order. Its kernel orientation is input × output, and its initialized parameters are overwritten only for this deterministic lab.' },
})

const autograd = lab({
  id: 'autograd-notebook-lab', title: 'Notebook Lab: Follow Stored Gradients',
  prompt: 'Trace the scalar tensor implementation in Homework2026/12AutoGrad.ipynb, including its behavior when backward is called twice on the same graph.',
  instructions: 'The excerpt reproduces the constructor and methods needed for this fixture; the notebook’s other arithmetic methods are omitted. Run all invocation statements in order. The second call intentionally retains intermediate gradients; it is not equivalent to two fresh graphs. Each saved variable is a number copied at that moment.',
  code: `import numpy as np

class tensor():
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    def __mul__(self, other): #self * other
        out = tensor(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def log(self): # self.log()
        out = tensor(np.log(self.data), (self,))
        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other): # self**n
        out = tensor(self.data**other, (self,))
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def backward(self): #Implementation of backprop
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()`,
  fixtureTitle: 'Fresh scalar values', fixture: `a = tensor(2)
b = tensor(3)
d = a * (b**2)`,
  invocationTitle: 'Save results before and after each backward call',
  invocation: `forward = d.data
d.backward()
first_a = a.grad
first_b = b.grad
d.backward()
second_b = b.grad

a.grad = 0
b.grad = 0
d = a * (b**2)  # rebuild, as the notebook instructs
d.backward()
rebuilt_b = b.grad

u = tensor(4)
h = (u**0.5).log()
h.backward()
chain_grad = u.grad`,
  stages: [
    stage('forward', '1. Evaluate the forward expression', 'Use a = 2 and b = 3.', 'forward', 'eighteen', [['twelve', '12'], ['eighteen', '18'], ['thirty-six', '36']], 'The graph computes 2×3² = 18 before any gradients are propagated.'),
    stage('first-a', '2. Follow the derivative to a', 'Inspect the multiplication backward rule on the first call.', 'first_a', 'nine', [['six', '6'], ['twelve', '12'], ['nine', '9']], 'The multiplication contributes b² = 9 to a.grad.'),
    stage('first-b', '3. Follow the derivative to b', 'The derivative passes through both multiplication and the square.', 'first_b', 'twelve', [['twelve', '12'], ['six', '6'], ['nine', '9']], 'The product gives the square node gradient 2; the square gives b a further factor 2b = 6, producing 12.'),
    stage('repeat', '4. Reuse the uncleared graph', 'Read += carefully: the intermediate square node also keeps its gradient between calls.', 'second_b', 'thirty-six', [['twenty-four', '24'], ['thirty-six', '36'], ['twelve', '12']], 'On the second call the square node gradient grows from 2 to 4. It adds 6×4 = 24 to b.grad, which was already 12, yielding 36. This is specific to the uncleared reused graph.'),
    stage('rebuild', '5. Reset leaves and rebuild', 'The invocation clears a.grad and b.grad and creates a new square node.', 'rebuilt_b', 'twelve', [['zero', '0'], ['thirty-six', '36'], ['twelve', '12']], 'A rebuilt graph has fresh intermediate gradients, so clearing the leaves and rebuilding restores the first-pass derivative 12.'),
    stage('chain', '6. Trace a different composition', 'Use the displayed power and log methods for the fresh u tensor.', 'chain_grad', 'eighth', [['eighth', '0.125'], ['quarter', '0.25'], ['half', '0.5']], 'The logarithm contributes 1/2 at √4 = 2; the square-root derivative is 1/4. Their product is 1/8.'),
  ], hints: ['Every gradient begins at zero, but the root is explicitly set to 1 on each backward call.', 'Repeated use of this graph retains intermediate gradients as well as leaf gradients.'],
  reveal: { explanation: 'Autodifferentiation composes local derivatives. This implementation uses accumulation, so reusing an uncleared graph gives b.grad = 36 on the second pass, not merely twice the original 12. The notebook’s reset-and-rebuild pattern restores a fresh derivative.' },
})

const pytorch = lab({
  id: 'pytorch-notebook-lab', title: 'Notebook Lab: Audit Shapes Before Training',
  prompt: 'Inspect the network and loss conventions shown in Homework2026/13PyTorch.ipynb without relying on trained weights or reported final accuracy.',
  instructions: 'The code panel reproduces the corrected notebook’s introductory architecture and relevant cars/Iris lines. The cars and Iris excerpts are commented here only because their dataset and trained-model setup is omitted. Unlike the corrected notebook, the fixed two-row diagnostic deliberately uses a flat target of shape (2,), a common mistake that broadcasts against predictions of shape (2, 1). Diagnose this intentional counterexample as written; its broadcasting warning is expected. No training or data download is needed.',
  code: `import torch
from torch.nn import Linear, ReLU, Sequential

network=Sequential(
    Linear(2,3),
    ReLU(),
    Linear(3,1)
)

# Corrected notebook excerpts; dataset/model setup is omitted here:
# mpg=torch.tensor(cars.mpg.to_numpy(),dtype=torch.float32).reshape(-1,1)
# MSEloss=torch.nn.MSELoss()(prediction,mpg)
# predictions=torch.argmax(iris_net(X),dim=1)`,
  fixtureTitle: 'Fixed shape checks and an intentionally flat diagnostic target',
  fixture: `network = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))
X = torch.zeros(15, 2)
prediction = torch.tensor([[1.], [3.]])
mpg = torch.tensor([2., 0.])
scores = torch.tensor([[2., -1., 0.], [0., 3., 1.]])`,
  invocationTitle: 'Shapes and losses, not random prediction values',
  invocation: `weight_shape = tuple(network[0].weight.shape)
parameter_count = sum(p.numel() for p in network.parameters())
output_shape = tuple(network(X).shape)
error_shape = tuple((prediction - mpg).shape)
broadcast_mse = torch.nn.MSELoss()(prediction, mpg).item()
class_indices = torch.argmax(scores, dim=1).tolist()`,
  stages: [
    stage('weight', '1. Inspect a PyTorch weight matrix', 'PyTorch Linear stores one row per output feature, unlike the custom kernel in Homework 11.', 'weight_shape', 'three-two', [['two-three', '(2, 3)'], ['three-two', '(3, 2)'], ['fifteen-three', '(15, 3)']], 'Linear(2, 3) stores weights in output × input order, (3, 2); its forward operation handles the corresponding transpose.'),
    stage('count', '2. Count parameters', 'Count both weight matrices and both bias vectors.', 'parameter_count', 'thirteen', [['nine', '9'], ['ten', '10'], ['thirteen', '13']], 'The first layer has 6 weights and 3 biases; the final layer has 3 weights and 1 bias, totaling 13.'),
    stage('output', '3. Preserve the batch axis', 'The random initialization can change values, but not this shape.', 'output_shape', 'fifteen-one', [['fifteen-one', '(15, 1)'], ['one-fifteen', '(1, 15)'], ['fifteen', '(15,)']], 'Fifteen input observations each receive one output, so the last dimension remains present.'),
    stage('broadcast', '4. Diagnose target broadcasting', 'Trailing dimensions align: (2, 1) minus (2,) expands to a matrix.', 'error_shape', 'two-two', [['two-one', '(2, 1)'], ['two-two', '(2, 2)'], ['error', 'A shape error prevents subtraction.']], 'The target vector broadcasts across each prediction row, producing every prediction–target pair rather than just matching observations.'),
    stage('mse', '5. Trace the loss actually computed', 'Trace the displayed mismatch as written; do not reshape mpg for this checkpoint.', 'broadcast_mse', 'three', [['five', '5.0'], ['ten', '10.0'], ['three', '3.0']], 'Squared errors are [[1, 1], [1, 9]], so the default mean is 3. Matching targets would instead give 5, but that is not the expression executed here.'),
    stage('classes', '6. Select a class for each row', 'This diagnostic uses fixed raw scores rather than iris_net training output.', 'class_indices', 'zero-one', [['zero-one', '[0, 1]'], ['zero-zero', '[0, 0]'], ['one-two', '[1, 2]']], 'dim=1 selects the largest score within each observation row; the class indices are zero-based.'),
  ], hints: ['A PyTorch dense weight matrix uses (output features, input features).', 'Write the entire 2 × 2 subtraction matrix before taking its mean squared value.'],
  reveal: { explanation: 'Shape checks are deterministic even with randomly initialized weights. The corrected notebook keeps the cars target column-shaped so predictions and targets match observation by observation. This lab deliberately supplies a flat target instead to expose a common shape mistake: broadcasting silently changes the objective into an all-pairs comparison.' },
})

const optimizers = lab({
  id: 'optimizers-notebook-lab', title: 'Notebook Lab: Trace the Digits Training Pipeline',
  prompt: 'Trace preprocessing, mini-batch boundaries, and evaluation mode from Homework2026/14Optimizers.ipynb with deterministic small diagnostic inputs.',
  instructions: 'The code panel quotes the relevant supplied network and operations. The original full training loop is shown only as a commented excerpt; do not train a model or download data. The fixture fixes N = 1437 and an ordered permutation so every answer is independent of split randomness and learned parameters. The actual supplied network has dropout only after its two hidden layers, not after its output layer.',
  code: `import numpy as np
import torch
from torch import nn
from torch.optim import Adam

# Notebook preprocessing, before the train/test split:
# X=(X-8)/8 #normalization--each pixel was in range 0-16

digitsNN=nn.Sequential(
    nn.Linear(64,32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Dropout(0.1),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.BatchNorm1d(16),
    nn.Dropout(0.1),
    nn.Linear(16,10),
)
optimizer=Adam(digitsNN.parameters(),lr=0.01)

# Relevant training-loop lines, commented here to avoid training:
# for i in range(0, N, batch_size):
#     batch_indices = indices[i:i+batch_size]
#     batch_X = Xtrain[batch_indices]
#     batch_y = ytrain[batch_indices]
#     optimizer.zero_grad()
#     CEloss=nn.CrossEntropyLoss()(digitsNN(batch_X),batch_y)
#     CEloss.backward()
#     optimizer.step()

# Evaluation lines from the notebook:
# digitsNN.eval()
# with torch.no_grad():
#   y_pred=digitsNN(Xtest) #generate predictions for the test set.`,
  fixtureTitle: 'Fixed sizes and a diagnostic test batch',
  fixture: `raw_pixels = np.array([0., 8., 16.])
N = 1437
batch_size = 100
indices = torch.arange(N)
Xtest = torch.zeros(37, 64)`,
  invocationTitle: 'Run each diagnostic in order',
  invocation: `normalized = (raw_pixels - 8) / 8
batches = [indices[i:i+batch_size] for i in range(0, N, batch_size)]
batch_count = len(batches)
last_size = len(batches[-1])
digitsNN.train()
with torch.no_grad():
    dropout_before_eval = digitsNN[3].training
digitsNN.eval()
with torch.no_grad():
    y_pred = digitsNN(Xtest)
output_shape = tuple(y_pred.shape)
dropout_after_eval = digitsNN[3].training`,
  stages: [
    stage('normalize', '1. Map the pixel scale', 'Apply the notebook’s affine normalization to the fixed pixel values.', 'normalized.tolist()', 'symmetric', [['symmetric', '[-1.0, 0.0, 1.0]'], ['unit', '[0.0, 0.5, 1.0]'], ['shift', '[-8.0, 0.0, 8.0]']], 'Subtracting 8 centers the original [0, 16] range and dividing by 8 maps it to [−1, 1].'),
    stage('batches', '2. Count optimizer updates per epoch', 'Each complete or short batch reaches one optimizer.step() in the supplied loop.', 'batch_count', 'fifteen', [['fourteen', '14'], ['fifteen', '15'], ['one', '1']], 'range(0, 1437, 100) starts batches at 0 through 1400, for 15 updates per epoch.'),
    stage('tail', '3. Keep the remainder', 'Trace the slice starting at index 1400.', 'last_size', 'thirty-seven', [['hundred', '100'], ['zero', '0'], ['thirty-seven', '37']], 'The last slice contains indices 1400 through 1436: 37 observations. It is neither padded nor dropped.'),
    stage('recording', '4. Separate no-grad from layer mode', 'The invocation calls train() before entering the first no_grad context.', 'dropout_before_eval', 'true', [['false', 'False'], ['true', 'True'], ['none', 'None']], 'The context disables gradient recording but does not change a module’s training flag.', 'diagnoseMutation'),
    stage('shape', '5. Read the evaluation output dimensions', 'Use the actual final Linear(16,10) layer, without assuming learned score values.', 'output_shape', 'thirty-seven-ten', [['thirty-seven-one', '(37, 1)'], ['ten-thirty-seven', '(10, 37)'], ['thirty-seven-ten', '(37, 10)']], 'Each of the 37 rows receives ten raw class scores, one for each digit.'),
    stage('eval', '6. Check the mode switch', 'The invocation has now called digitsNN.eval().', 'dropout_after_eval', 'false', [['false', 'False'], ['true', 'True'], ['zero-probability', '0.1']], 'eval() sets the training flag to False throughout the model, so dropout passes activations through and batch normalization uses running statistics.'),
  ], hints: ['Count a final short batch whenever N is not divisible by batch_size.', 'no_grad and eval have separate responsibilities.'],
  reveal: { explanation: 'The deterministic questions follow preprocessing and control flow, not a stochastic training outcome. The original loop clears gradients, computes a loss, backpropagates, then steps the optimizer once per mini-batch.' },
})

const convolutions = lab({
  id: 'convolutions-notebook-lab', title: 'Notebook Lab: Trace the Image Windows',
  prompt: 'Trace the supplied Conv and MaxPool functions in Homework2026/15Convolutions.ipynb using a small numerical image instead of downloading face photos.',
  instructions: 'These functions reproduce the notebook implementation. Conv slides the kernel without flipping it (cross-correlation in strict mathematical terminology). MaxPool uses non-overlapping complete blocks and discards leftover rows and columns. The fixture does not alter either function.',
  code: `import numpy as np

def Conv(image,kernel):
  newdimx=image.shape[0]-kernel.shape[0]+1
  newdimy=image.shape[1]-kernel.shape[1]+1
  newimage=np.zeros((newdimx,newdimy))
  for i in range(newdimx):
    for j in range(newdimy):
      newimage[i,j]=np.sum(kernel*image[i:i+kernel.shape[0],j:j+kernel.shape[1]])
  return newimage

def MaxPool(image,pool_size):
  newimage=np.zeros((image.shape[0]//pool_size[0],image.shape[1]//pool_size[1]))
  for i in range(image.shape[0]//pool_size[0]):
    for j in range(image.shape[1]//pool_size[1]):
      newimage[i,j]=np.max(image[i*pool_size[0]:(i+1)*pool_size[0],j*pool_size[1]:(j+1)*pool_size[1]])
  return newimage`,
  fixtureTitle: 'A 3 × 5 image and an asymmetric filter',
  fixture: `image = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15]])
kernel = np.array([[1, 2], [0, -1]])`,
  invocationTitle: 'Use the supplied functions exactly as written',
  invocation: `response = Conv(image, kernel)
pooled_image = MaxPool(image, (2, 2))
pooled_response = MaxPool(response, (2, 2))
flipped_response = Conv(image, kernel[::-1, ::-1])
changed = image.copy()
changed[-1, :] = 1000
pooled_changed = MaxPool(changed, (2, 2))`,
  stages: [
    stage('shape', '1. Count valid filter positions', 'A 2 × 2 kernel is slid over a 3 × 5 image.', 'response.shape', 'two-four', [['three-five', '(3, 5)'], ['two-four', '(2, 4)'], ['one-two', '(1, 2)']], 'The valid output dimensions are 3−2+1 = 2 and 5−2+1 = 4.'),
    stage('response', '2. Trace the first window', 'Use the unflipped kernel shown in the fixture.', 'float(response[0, 0])', 'minus-two', [['eighteen', '18.0'], ['two', '2.0'], ['minus-two', '-2.0']], 'The first response is 1×1 + 2×2 + 0×6 − 1×7 = −2.'),
    stage('pool', '3. Pool the original image', 'Each 2 × 2 block is replaced by its maximum; incomplete borders are discarded.', 'pooled_image.tolist()', 'seven-nine', [['seven-nine', '[[7.0, 9.0]]'], ['twelve-fourteen', '[[12.0, 14.0]]'], ['seven-nine-ten', '[[7.0, 9.0, 10.0]]']], 'The two complete top-row blocks have maxima 7 and 9. The last row and last column are not pooled.'),
    stage('composition', '4. Pool the response map', 'First complete every Conv window, then pool that response.', 'pooled_response.tolist()', 'ten-fourteen', [['two-four', '[[2.0, 4.0]]'], ['ten-fourteen', '[[10.0, 14.0]]'], ['seven-nine', '[[7.0, 9.0]]']], 'The response is [[−2, 0, 2, 4], [8, 10, 12, 14]], whose two complete pooling blocks have maxima 10 and 14.'),
    stage('flip', '5. Diagnose flipping the kernel', 'The invocation explicitly reverses both kernel axes before calling the unchanged Conv function.', 'float(flipped_response[0, 0])', 'eighteen', [['minus-two', '-2.0'], ['two', '2.0'], ['eighteen', '18.0']], 'The flipped kernel is [[−1, 0], [2, 1]], so the first response is −1 + 0 + 12 + 7 = 18. The supplied unflipped convention matters.', 'diagnoseMutation'),
    stage('discard', '6. Change only the discarded row', 'The final image row is replaced by very large values before pooling.', 'pooled_changed.tolist()', 'unchanged', [['unchanged', '[[7.0, 9.0]]'], ['large', '[[1000.0, 1000.0]]'], ['extra-row', '[[7.0, 9.0], [1000.0, 1000.0]]']], 'Floor division permits only one row of complete pooling blocks, so no value from the final image row is read.'),
  ], hints: ['Keep the kernel orientation exactly as shown unless the invocation explicitly flips it.', 'Integer floor division in MaxPool determines which edge pixels are never visited.'],
  reveal: { explanation: 'The notebook’s Conv is valid stride-1 cross-correlation; MaxPool uses stride equal to its pool dimensions. Both output sizes and boundary behavior follow directly from their loops.' },
})

export const neuralFoundationsNotebookLabs = { mlps, autograd, pytorch, optimizers, convolutions }
