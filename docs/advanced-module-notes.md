# Homework 11–25: module and notebook alignment

These modules follow the `ml-assignment-design` skill: self-contained conceptual warm-ups, meaningful interactive exploration, then notebook-specific code comprehension. They preserve the instructor's implementations. The notebook files were initially unchanged; Homework 13 was subsequently corrected with the instructor's approval, as detailed below. Every module records raw answers and supports partial JSON submission, retries, and direct sidebar navigation. No subjective evaluation of generated images or text is required.

The notebook lab is not a request to retrain a model to reproduce one arbitrary number. Its small checkpoints concern deterministic supplied examples, shapes, operations, and controlled experiments. Fixed educational examples are identified as such rather than presented as results from an instructor checkpoint. Exploratory controls are not themselves submitted answers; the question's fixed reference conditions determine the scored choices.

## Coverage

| Homework | Main conceptual progression |
| --- | --- |
| 11 | Composition of layers, nonlinear activations, shared parameters |
| 12 | Chain rule, contributions along multiple paths, gradient accumulation |
| 13 | Batch/feature dimensions, aligned regression targets, classification logits |
| 14 | Optimizer state, shuffled mini-batches, training versus evaluation |
| 15 | Local filters, weight sharing, pooling and boundary behavior |
| 16 | CNN spatial/channel dimensions, learned filters, evaluation behavior |
| 17 | Denoising targets, lossy representations, reconstruction |
| 18 | Tokenization, vocabulary/embedding indices, next-token targets |
| 19 | Query/key matching, value retrieval, normalized attention |
| 20 | Causal masks, multiple heads, prefix invariance and position |
| 21 | Residual blocks, normalization, complete causal model data flow |
| 22 | Story-level splits, shifted batches, controlled sampling |
| 23 | Activation caches, logit lens, association versus intervention |
| 24 | Sparse reconstruction, L1 tradeoffs, decoder normalization |
| 25 | Coefficient interventions, held-out evaluation, matched controls and quality costs |

## Existing notebook issues to review separately

These observations do not prevent the app's fixed-fixture labs from being graded. Other than the approved Homework 13 corrections, the notebooks have not been edited.

- **13PyTorch — corrected:** The completed examples now run rather than remaining commented out. The introductory loss uses its defined `target`, and the cars target is reshaped to `(N, 1)` to match the predictions. Cars features are standardized for stable, effective gradient descent; the target remains in MPG units. Manual updates use `torch.no_grad()` rather than `.data`. Stale outputs and hard-coded results have been removed, and the prose now matches the existing two-hidden-layer Iris architecture. The app's fixed flat-target example remains as an explicitly labeled common-mistake diagnostic, not as a claim about the corrected notebook. Its numerical answers and version-1 grading keys are unchanged.
- **14Optimizers:** The prose asks for normalization/dropout after every linear layer, but the code correctly leaves the final classification layer as raw logits. Its last cell refers to the obsolete `homework16gradescope.ipynb` workflow.
- **15Convolutions:** The from-scratch `Conv` function does not flip the kernel (cross-correlation, as commonly used by CNN layers). `MaxPool` uses floor division and drops incomplete edge windows. These are explicit conventions, not silently corrected errors.
- **16CNNs:** The final instruction refers to the obsolete `homework18gradescope.py` file. App submissions instead use this module's JSON export.
- **17AutoEncoders:** The reshape hardcodes 1,288 images. Training uses inverted dropout, which scales retained pixels, whereas the manual noisy-image example only masks pixels. The encoder reduces spatial resolution but expands the total number of activations: 64 × 12 × 9 = 6,912, compared with 1 × 48 × 36 = 1,728 input values. The lab therefore does not describe this architecture as a smaller latent vector. The old screenshot/PDF submission instruction is not the new app's submission format. Its trained output is not guaranteed to perfectly restore every missing pixel.
- **18–25:** Some checkpoint dictionary variable names retain their previous homework numbers. Lab references use the current filenames and section labels, not those old numbers. The fast GPT/SAE reference files are present in the repository; the optional TinyStories reference pair is not. No app answer depends on which pair is installed or on the identity of a feature learned in a student run.

## Publishing and grading

New modules use version 1. Existing assignment versions and answer keys are unchanged. `npm run gradescope:test` rebuilds and verifies all packages, including the fifteen new packages listed in `gradescope/README.md`. Upload each ZIP separately to Gradescope when releasing the corresponding assignment. A GitHub push is not a Gradescope upload.

For a source-backed notebook smoke check, run `python3 scripts/verify_advanced_notebook_traces.py` in an environment with NumPy and PyTorch. It executes selected definitions from the actual notebooks, the complete small notebooks 18–21, and fixed fixtures for later interventions. It never trains a model, downloads a dataset, or changes a notebook/checkpoint. Full training and generated text quality are deliberately outside these deterministic checks. App tests separately verify answer keys, controls, export behavior, and grading.

Run `node scripts/verify_advanced_app_labs.mjs /path/to/python` to execute the exact code, fixture, and invocation displayed in every new lab and compare literal-valued answers. The Homework 13 app lab's deliberately incorrect flat-target fixture emits a PyTorch broadcasting warning; the corrected notebook itself does not use that mismatch. Set `ML_NOTEBOOK_PYTHON=/path/to/python` when running `npm run test:run` to also enable the source-backed Python checks in the module tests.
