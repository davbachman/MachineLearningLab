"""Execute small source-backed checks for Homework 11–25, without training/downloads.

Run with Python, NumPy and PyTorch installed. Notebooks/checkpoints are read-only;
only selected definitions and tiny fixed fixtures run, never notebook training cells.
The app's answer keys and partial-credit rules have separate Vitest tests.
"""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "Homework2026"
torch.set_num_threads(1)


def cells(filename):
    return ["".join(c["source"]) for c in json.loads((COURSE / filename).read_text())["cells"]]


def execute(source, scope):
    with redirect_stdout(io.StringIO()):
        exec(compile(source, "<notebook trace>", "exec"), scope)


def definitions(source, scope):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    with redirect_stdout(io.StringIO()):
        exec(compile(tree, "<notebook definitions>", "exec"), scope)


def scope():
    return {"__name__": "__main__", "np": np, "torch": torch, "nn": nn}


def verify():
    n = cells("11MLPs.ipynb")
    env = scope()
    for index in (4, 5, 6, 8):
        execute(n[index], env)
    assert env["network"](env["X"]).shape == (15, 1)
    assert sum(layer.kernel.size + layer.bias.size for layer in (env["layer1"], env["layer3"])) == 13
    print("11: exact NumPy layer composition, output shape, parameter count")

    n = cells("12AutoGrad.ipynb")
    env = scope()
    execute(n[3], env)
    execute(n[5], env)
    assert (env["a"].grad, env["b"].grad) == (9, 12)
    execute(n[8], env)
    assert (env["a"].grad, env["b"].grad) == (18, 36)
    execute(n[10], env)
    assert (env["a"].grad, env["b"].grad) == (9, 12)
    execute(n[15], env)
    assert env["h_u"] == 0.125
    print("12: fresh and reused graphs, clearing/rebuilding, log-root derivative")

    n = cells("13PyTorch.ipynb")
    env = scope() | {"Linear": nn.Linear, "ReLU": nn.ReLU, "Sequential": nn.Sequential}
    execute(n[4], env)
    assert env["network"](torch.zeros(15, 2)).shape == (15, 1)
    # Execute the actual loss/target assignments, without training or downloading cars.
    def named_assignment(source, name):
        nodes = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == name
                         for target in node.targets)]
        assert len(nodes) == 1, (name, len(nodes))
        return ast.unparse(nodes[0])

    env.update(prediction=torch.tensor([[1.], [3.]]), target=torch.tensor([[2.], [0.]]))
    assert "mpg" not in env
    execute(named_assignment(n[12], "MSEloss"), env)
    assert env["MSEloss"].item() == 5  # The intro uses target, not an undefined mpg.
    env["cars"] = SimpleNamespace(mpg=SimpleNamespace(to_numpy=lambda: np.array([2., 0.])))
    execute(named_assignment(n[14], "mpg"), env)
    assert env["mpg"].shape == env["prediction"].shape == (2, 1)
    execute(named_assignment(n[18], "MSEloss"), env)
    assert env["MSEloss"].item() == 5
    print("13: active network, defined intro target, and correctly paired cars targets")

    env = scope()
    execute(cells("14Optimizers.ipynb")[18], env)
    assert env["digitsNN"](torch.zeros(4, 64)).shape == (4, 10)
    assert isinstance(env["digitsNN"][-1], nn.Linear)
    assert [len(range(12)[i:i + 5]) for i in range(0, 12, 5)] == [5, 5, 2]
    print("14: digits architecture returns logits; final partial mini-batch retained")

    env = scope()
    n = cells("15Convolutions.ipynb")
    execute(n[9], env)
    execute(n[17], env)
    image = np.array([[1, 2, 0, 1], [3, 4, 1, 0], [0, 2, 5, 1], [1, 0, 2, 3]])
    response = env["Conv"](image, np.array([[-1, 1], [-1, 1]]))
    assert response.shape == (3, 3) and response[0, 0] == 2
    pooled = env["MaxPool"](np.array([[1, 5, 2, 4, 99], [3, 0, 6, 1, 99], [99]*5]), (2, 2))
    assert pooled.tolist() == [[5., 6.]]
    print("15: exact Conv orientation and incomplete pooling-border behavior")

    env = scope()
    execute(cells("16CNNs.ipynb")[10], env)
    model = env["model"].eval()
    with torch.no_grad():
        assert model[:12](torch.zeros(2, 1, 50, 37)).shape == (2, 128, 6, 4)
        assert model(torch.zeros(2, 1, 50, 37)).shape == (2, 7)
    print("16: exact CNN pooled dimensions, 3072 flatten width, seven logits")

    env = scope()
    execute(cells("17AutoEncoders.ipynb")[14], env)
    env["autoencoder"].eval()
    with torch.no_grad():
        x = torch.zeros(2, 1, 48, 36)
        assert env["encoder"](x).shape == (2, 64, 12, 9)
        assert env["autoencoder"](x).shape == x.shape
    print("17: exact encoder/decoder dimensions, not a scalar undercomplete bottleneck")

    for filename, assertions in [
        ("18Tokenization.ipynb", lambda e: e["input_vectors"].shape == (1, 12, 8)
         and torch.equal(e["inputs"][1:], e["targets"][:-1]) and e["stoi"]["<bos>"] == 2),
        ("19Attention.ipynb", lambda e: e["bridge_outputs_match"] and e["custom_parameter_count"] == 58
         and e["learned_weights"].shape == (1, 4, 4) and e["learned_output"].shape == (1, 4, 2)),
        ("20CausalAttention.ipynb", lambda e: e["attention_weights"].shape == (3, 2, 6, 6)
         and e["causal_weights"][0].tolist() == [1., 0., 0., 0.] and e["prefix_difference"].item() < 1e-6),
        ("21PicoGPT.ipynb", lambda e: e["logits"].shape == (3, 12, 20) and e["weights_are_tied"]
         and e["generated_ids"].shape == (1, 11) and e["complete_model_prefix_difference"].item() < 1e-6),
    ]:
        env = scope()
        notebook = json.loads((COURSE / filename).read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                execute("".join(cell["source"]), env)
        assert assertions(env), filename
        print(f"{filename[:2]}: complete small notebook runs; deterministic structural assertions pass")

    sys.path.insert(0, str(COURSE))
    from pico_gpt import get_batch
    x, y = get_batch(torch.arange(30), batch_size=3, block_size=4,
                     generator=torch.Generator().manual_seed(158))
    assert x.shape == y.shape == (3, 4) and torch.equal(x + 1, y)
    print("22: course helper's actual random-window and shifted-target implementation")

    env = scope() | {
        "tokenizer": SimpleNamespace(stoi={".": 7}), "target_layer": 0,
        "targets": torch.tensor([[7, 0, 7, 0]]),
        "cache": {"blocks": [{"residual_post": torch.tensor([[[2., 1.], [0., 1.], [4., 1.], [0., 1.]]])}]},
    }
    execute(cells("23TransformerActivations.ipynb")[11], env)
    assert env["period_direction"].tolist() == [1., 0.]
    print("23: actual label-conditioned mean difference and direction normalization")

    env = scope()
    definitions(cells("24SparseAutoencoders.ipynb")[6], env)
    sae = env["SparseAutoencoder"](2, 8)
    reconstruction, features = sae(torch.zeros(3, 2))
    assert reconstruction.shape == (3, 2) and features.shape == (3, 8)
    assert torch.all(features >= 0) and torch.allclose(sae.decoder_directions.norm(dim=1), torch.ones(8))
    print("24: actual SAE returns overcomplete nonnegative codes and unit decoder rows")

    env = scope() | {
        "sae": SimpleNamespace(encode=lambda residual: residual[..., :1] * 0 + 2),
        "selected_feature": 0, "target_layer": 1,
        "feature_direction": torch.tensor([.6, .8]),
    }
    definitions(cells("25FeatureSteering.ipynb")[9], env)
    residual = torch.tensor([[[3., 4.]]])
    assert torch.allclose(env["ablate_feature"](1, residual), torch.tensor([[[1.8, 2.4]]]))
    assert torch.allclose(env["amplify_feature"](1, residual), torch.tensor([[[4.2, 5.6]]]))
    assert torch.equal(env["amplify_feature"](0, residual), residual)
    env.update({"WORD_PATTERN": re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?"),
                "tokenizer": SimpleNamespace(itos=["<bos>", "word", ".", "<eos>"])})
    definitions(cells("25FeatureSteering.ipynb")[18], env)
    statistics = env["generation_statistics"](torch.tensor([[0, 1, 1, 2, 3]]), 1)
    assert statistics["generated_tokens"] == 4 and statistics["words_per_period"] == 2
    assert statistics["periods_per_100_tokens"] == 33.33
    print("25: actual intervention signs, layer guard, and generated-token statistics")


if __name__ == "__main__":
    verify()
