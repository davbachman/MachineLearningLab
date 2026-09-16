"""Run the combined notebook and independently verify the app's Python traces.

Requires NumPy, pandas, matplotlib, scikit-learn, nbformat, and npm dependencies.
Run from the repository root. Optional --figures DIRECTORY saves plot previews.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--figures", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / "Homework2026/5PolynomialRegression.ipynb", as_version=4)
nbformat.validate(notebook)
scope = {}
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        exec(compile(cell.source, f"notebook cell {index}", "exec"), scope)
assert np.isfinite(scope["metrics"]).all()
assert (np.diff(scope["metrics"][:, 0]) < 1e-5).all()
assert scope["best_degree"] == 4
assert len(plt.get_fignums()) == 2
for ax in plt.figure(1).axes:
    assert (np.diff(ax.lines[0].get_xdata()) >= 0).all()
assert scope["LinearRegression"].__module__ != "sklearn.linear_model._base"

# Load the actual TypeScript answer definitions, not a second hand-copied key.
node = """
import { createServer } from 'vite';
const server = await createServer({ server: { middlewareMode: true }, logLevel: 'silent' });
try {
 const labs = await server.ssrLoadModule('/src/data/regressionCodeLabQuestions.ts');
 const visuals = await server.ssrLoadModule('/src/data/polynomialVisualData.ts');
 const degrees = await server.ssrLoadModule('/src/data/polynomialDegreeData.ts');
 process.stdout.write(JSON.stringify({ feature: labs.polynomialRegressionNotebookLab,
   evaluation: labs.polynomialEvaluationNotebookLab, visuals, degrees }));
} finally { await server.close(); }
"""
payload = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", node], cwd=root, text=True))
assert np.allclose(scope["metrics"], np.array(payload["visuals"]["polynomialErrorRows"])[:, 1:], atol=0.0051)
points = np.array(payload["visuals"]["polynomialExamplePoints"])
errors = {}
for model in payload["visuals"]["polynomialExampleModels"]:
    fitted = np.polynomial.polynomial.polyfit(points[:, 0], points[:, 1], model["degree"])
    assert np.allclose(fitted, model["coefficients"], atol=1e-5)
    errors[model["name"]] = np.mean((np.polynomial.polynomial.polyval(points[:, 0], fitted) - points[:, 1])**2)
assert errors["C"] < errors["A"] < errors["B"]

# Independently refit every slider model without touching the held-out observations.
degree_data = payload["degrees"]
train = np.array(degree_data["trainingPoints"])
validation = np.array(degree_data["validationPoints"])
slider_errors = []
for degree, coefficients in enumerate(degree_data["degreeCoefficients"], start=1):
    fitted = np.polynomial.polynomial.polyfit(train[:, 0], train[:, 1], degree)
    assert np.allclose(fitted, coefficients, atol=1e-9)
    slider_errors.append([np.mean((np.polynomial.polynomial.polyval(data[:, 0], fitted) - data[:, 1])**2)
                          for data in (train, validation)])
    curve = np.polynomial.polynomial.polyval(np.linspace(-2, 2, 401), fitted)
    assert curve.min() >= 0 and curve.max() <= 5, "Fixed plot axes must include the entire curve"
assert np.array(slider_errors).argmin(axis=0).tolist() == [7, 1]

def chosen(stage):
    field = stage["fields"][0]
    return next(o["label"] for o in field["options"] if o["id"] == field["correctOptionId"])

def check_numeric(stage, value):
    for option in stage["fields"][0]["options"]:
        candidate = ast.literal_eval(option["label"])
        matches = np.shape(candidate) == np.shape(value) and np.allclose(candidate, value, atol=1e-8)
        assert matches == (option["id"] == stage["fields"][0]["correctOptionId"]), (stage["id"], option)

feature = payload["feature"]
env = {}
exec(feature["code"], env)
exec(feature["fixture"], env)
exec(feature["invocation"], env)
for stage in feature["stages"][:3]:
    check_numeric(stage, eval(stage["fields"][0]["label"], env))
# The lab must show the exact classes used by the notebook (labels/comments aside).
class_nodes = {n.name: ast.dump(n, include_attributes=False)
               for c in notebook.cells if c.cell_type == "code"
               for n in ast.parse(c.source).body if isinstance(n, ast.ClassDef)}
for n in ast.parse(feature["code"]).body:
    if isinstance(n, ast.ClassDef):
        assert ast.dump(n, include_attributes=False) == class_nodes[n.name], n.name
mutant = {}
exec(feature["code"].replace("X**(i+1)", "X**i"), mutant)
exec(feature["fixture"], mutant)
exec(feature["invocation"], mutant)
check_numeric(feature["stages"][3], mutant["features"].tolist())
test_field = feature["stages"][4]["fields"][0]
for option in test_field["options"]:
    statement = option["description"].strip('`')
    exec(statement, env)
    try:
        exec(statement, mutant)
        distinguishes = False
    except AssertionError:
        distinguishes = True
    assert distinguishes == (option["id"] == test_field["correctOptionId"])

evaluation = payload["evaluation"]
env = {"np": np, "PolynomialFeatures": scope["PolynomialFeatures"], "LinearRegression": scope["LinearRegression"]}
exec(evaluation["fixture"], env)
exec(evaluation["code"], env)
for stage in evaluation["stages"][:8]:
    check_numeric(stage, eval(stage["fields"][0]["label"], env))
mutant = env.copy()
exec(evaluation["code"].replace("MSEval=((val_predictions-yval)**2).mean()", "MSEval=((val_predictions-yval)**2).sum()"), mutant)
check_numeric(evaluation["stages"][8], mutant["MSEval"])
for option in evaluation["stages"][9]["fields"][0]["options"]:
    exec(option["label"], env)
    try:
        exec(option["label"], mutant)
        distinguishes = False
    except AssertionError:
        distinguishes = True
    assert distinguishes == (option["id"] == evaluation["stages"][9]["fields"][0]["correctOptionId"])

if args.figures:
    args.figures.mkdir(parents=True, exist_ok=True)
    for number in plt.get_fignums():
        plt.figure(number).savefig(args.figures / f"polynomial-{number}.png")
print("PASS: full notebook execution, independent class/answer checks, mutations, plots, and MSE chart data.")
print("Validation MSE by degree:", scope["metrics"][:, 1])
