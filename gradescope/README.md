# Gradescope autograders

Each published app assignment has its own Gradescope package. The package grades the JSON file downloaded from the app; students do not submit code.

## Build the packages

From the repository root, run:

```sh
npm run gradescope:build
```

This bundles the grader as `gradescope/build/grader.cjs`, asks that bundle for the current published-assignment metadata, and writes one upload-ready ZIP per assignment to `gradescope/dist/`. To rebuild only the ZIPs from an existing bundle, run:

```sh
python3 gradescope/scripts/build_packages.py
```

The builder and verifier use only the Python standard library. Run all grader and package checks with:

```sh
npm run gradescope:test
```

To run just the package verifier directly:

```sh
python3 gradescope/scripts/verify_packages.py
```

In addition to checking ZIP layout and executable bits, the verifier extracts each package, runs it against a malformed local submission using the path overrides below, and confirms that it writes a zero-score `results.json`.

The generated `gradescope/build/` and `gradescope/dist/` directories are disposable build output. Do not hand-edit their contents.

## Set up each Gradescope assignment

For every assignment:

1. Create a Gradescope [**Programming Assignment**](https://guides.gradescope.com/hc/en-us/articles/22254107840909-Creating-a-Programming-Assignment) and make its autograder worth 100 points. Programming Assignments require an institutional Gradescope license.
2. Open **Configure Autograder**, upload the corresponding `gradescope/dist/<assignment-id>.zip`, and select **Update Autograder**.
3. The grader splits those 100 points equally among the assignment's questions. A question with independently gradable parts, trace fields, or interaction phases divides its share equally among those checks. The reported total is capped at exactly 100.
4. Submit a JSON export from the app as an instructor test submission and confirm that the displayed question-level results total 100 points.

The ZIP has the four files Gradescope needs at its root: executable `setup.sh` and `run_autograder`, plus `grader.cjs` and that assignment's `assignment.json`. `setup.sh` uses the system Node installation when present and installs `nodejs` with `apt-get` only when Node is missing.

## Student submission instructions

Students can use the app's **Download JSON** control at any time and upload the resulting `<assignment-id>-submission.json` file to the matching Gradescope assignment. Incomplete exports are accepted and receive credit for the recorded answers that are correct. An answer currently edited in the interface is recorded only after the student checks or submits it. Students should upload the JSON file itself, not a screenshot, notebook, ZIP, or renamed text file.

The runner first looks for that app-generated filename. It also accepts `submission.json`, or a single JSON file in the submission directory, which makes instructor testing easier.

## Versions and updates

Every exported submission contains an assignment ID and version. Every autograder ZIP contains the current answer key and any explicitly supported historical keys. Unsupported versions are rejected rather than interpreted against a different answer key.

The K-means package accepts both version 6 (the original initialization) and version 7 (the easier initialization), using the appropriate Lloyd sequence for each. Upload the rebuilt `gradescope/dist/kmeans.zip` and regrade existing submissions after updating the grader. Students do not need to redo version-6 work or change the version in their JSON. Pushing the app to GitHub does not update the package already installed on Gradescope.

Whenever a published assignment's version changes, run `npm run gradescope:build` again and upload the newly generated ZIP to that Gradescope assignment. Existing ZIP filenames stay stable, so the new upload replaces the old autograder. Do the same after changing validation logic even if the visible questions did not change.

The rebuilt Linear Regression assignment (version 2) contains three questions: interactive RSS line fitting, interactive 3D plane fitting, and the expanded notebook lab. Upload the rebuilt `gradescope/dist/linear-regression.zip` when publishing it. Both fitting graders recompute RSS from the submitted coefficients and accept values within 5% of the least-squares minimum; they do not trust a submitted RSS readout. Plane residuals are response-axis errors at fixed feature values, not perpendicular distances. No earlier Linear Regression versions are supported because students had not started the assignment when it was rebuilt.

The Linear Regression notebook lab uses variant `linear-regression-notebook-v3`, with nine single-answer checkpoints. Each calculation can be checked and recorded separately; the grader awards partial credit per answer, including for an unfinished lab.

## Combined polynomial regression and overfitting assignment

Homework 5 now uses `Homework2026/5PolynomialRegression.ipynb` and the app assignment **Polynomial Regression and Overfitting** (ID `polynomial-regression`, version 3). Upload `gradescope/dist/polynomial-regression.zip` for the combined seven-question assignment. Questions 2 and 3 are degree-slider exercises minimizing training and validation MSE respectively. Their graders recompute the best degree from fixed training-only fits and held-out observations; submitted MSE readouts are not trusted. The two notebook labs grade each checkpoint independently for partial credit. The former separate Overfitting assignment is no longer published, and its obsolete ZIP is no longer generated. Version-1 and version-2 polynomial submissions have different questions and are rejected rather than graded against the new questions.

K-NN is Homework 3 with notebooks `3aKNN.ipynb` and `3bKNN.ipynb`; Linear Regression is Homework 4; later assignments are numbered consecutively. Renumbering alone does not change existing assignment IDs or their answer-key versions. See `Homework2026/README.md` for the full mapping.

The combined notebook and Python answer traces can be checked with `python3 scripts/verify_polynomial_notebook.py` after installing NumPy, pandas, matplotlib, scikit-learn, nbformat, and the app's npm dependencies. This executes the notebook and checks the actual app answer choices against its classes.

## Conceptual gradient descent assignment

Gradient Descent (Homework 6, ID `gradient-descent`, version 2) has five conceptual/visual warm-ups followed by the unchanged notebook lab. The learning-rate exercise records the chosen rate, final parameters, and final loss. The grader recomputes all 20 updates from the fixed starting point and accepts any available rate that reaches loss ≤ 1.05; submitted loss, parameters, and step counts cannot override that check. Multiple-choice parts and notebook fields receive independent partial credit.

Upload the rebuilt `gradescope/dist/gradient-descent.zip` when publishing these changes. It also retains the original version-1 answer key for students who already exported the previous five-question assignment. Do not change the version tag in a student's JSON.

## Local runner paths

On Gradescope, the runner uses the official paths:

- `/autograder/source`
- `/autograder/submission`
- `/autograder/results/results.json`

For local tests, paths can be overridden with these environment variables:

- `AUTOGRADER_ROOT`
- `AUTOGRADER_SOURCE_DIR`
- `AUTOGRADER_SUBMISSION_DIR`
- `AUTOGRADER_RESULTS_DIR`
- `AUTOGRADER_NODE_BIN`
- `AUTOGRADER_GRADER_FILE`
- `AUTOGRADER_ASSIGNMENT_FILE`
- `AUTOGRADER_SUBMISSION_FILE`
- `AUTOGRADER_RESULTS_FILE`

## Security model

These packages are designed for deterministic grading, not answer-key secrecy or tamper resistance. The app and grader can be inspected in the public repository, and a motivated student can inspect a downloaded autograder if they obtain it. Gradescope should be treated as the authoritative place where submitted JSON is scored, but this system does not prove that a student completed the work without outside help.
