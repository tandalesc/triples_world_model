#!/usr/bin/env python3
"""Visualize TextDynamicsModel architecture as a block diagram.

Generates a Graphviz diagram showing the compressor → dynamics → expander
pipeline with internal structure, data flow, and conditioning pathways.

Usage:
  uv run python scripts/visualize_architecture.py --run-dir results/mixed_chain_v16
  uv run python scripts/visualize_architecture.py --run-dir results/mixed_chain_v16 --format svg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def build_graph(model):
    try:
        import graphviz
    except ImportError:
        print("Install graphviz: uv add graphviz")
        sys.exit(1)

    g = graphviz.Digraph("TextDynamicsModel", format="png")
    g.attr(rankdir="TB", fontname="Helvetica", fontsize="11",
           bgcolor="white", pad="0.5")
    g.attr("node", fontname="Helvetica", fontsize="10", style="filled",
           shape="box", margin="0.15,0.08")
    g.attr("edge", fontname="Helvetica", fontsize="9")

    comp = model.text_compressor
    dyn = model.dynamics
    exp = model.text_expander
    d = model.config.d_model
    max_t = model.config.max_triples
    max_tok = model.max_text_tokens
    bn_d = model._bottleneck_dim

    comp_total, comp_train = count_params(comp)
    dyn_total, dyn_train = count_params(dyn)
    exp_total, exp_train = count_params(exp)
    mode_params = sum(p.numel() for p in [model.mode_emb.weight, model.mode_role_emb.weight])

    # --- Input ---
    g.node("input", f"BPE Text\n(B, T) T≤{max_tok}",
           fillcolor="#E8F4FD", shape="oval")

    # --- Shared Embedding ---
    emb_total, _ = count_params(model.shared_token_emb)
    g.node("emb", f"Shared Token Embedding\n(frozen, {format_params(emb_total)} params)\n{model.tokenizer.vocab_size} vocab → {d}d",
           fillcolor="#F0F0F0")

    # --- Compressor ---
    with g.subgraph(name="cluster_comp") as c:
        c.attr(label=f"Compressor ({format_params(comp_train)} trainable)",
               style="rounded,filled", fillcolor="#E8F5E9", color="#4CAF50")
        c.node("comp_self_attn", f"Self-Attention Encoder\n{model._text_compressor_layers}L, {d}d, {model.config.n_heads}H",
               fillcolor="#C8E6C9")
        c.node("comp_pool", f"Learned Pool Query\n→ {max_t}×3 bottleneck positions",
               fillcolor="#C8E6C9")

    # --- Bottleneck ---
    g.node("bottleneck", f"Bottleneck\n(B, {max_t}×3, {bn_d}d)",
           fillcolor="#FFF9C4", shape="box", penwidth="2")

    # --- Mode Conditioning ---
    g.node("mode", f"Mode Triple\n(advance/query/identity)\n{format_params(mode_params)} params",
           fillcolor="#F3E5F5", shape="oval")

    # --- Dynamics ---
    with g.subgraph(name="cluster_dyn") as c:
        c.attr(label=f"Dynamics Core ({format_params(dyn_train)} trainable)",
               style="rounded,filled", fillcolor="#E3F2FD", color="#2196F3")
        c.node("dyn_cat", f"Concat [mode | bottleneck]\n(B, 3+{max_t}×3, {bn_d}d)",
               fillcolor="#BBDEFB")
        c.node("dyn_transformer", f"Transformer\n{model._dynamics_layers}L, {bn_d}d, {max(1, bn_d//16)}H\n(zero-init residual gate)",
               fillcolor="#BBDEFB")
        c.node("dyn_strip", f"Strip mode → delta\n+ input residual",
               fillcolor="#BBDEFB")

    # --- Post-dynamics bottleneck ---
    g.node("bn_post", f"Post-Dynamics Bottleneck\n(B, {max_t}×3, {bn_d}d)",
           fillcolor="#FFF9C4", shape="box", penwidth="2")

    # --- Expander ---
    with g.subgraph(name="cluster_exp") as c:
        c.attr(label=f"Diffusion Expander ({format_params(exp_train)} trainable)",
               style="rounded,filled", fillcolor="#FBE9E7", color="#FF5722")

        c.node("exp_pool", f"Attention Pool → cond\n(B, {d}d)",
               fillcolor="#FFCCBC")
        c.node("exp_memory", f"Memory Proj + LayerNorm\n→ cross-attn keys/values\n(B, {max_t}×3, {d}d)",
               fillcolor="#FFCCBC")
        c.node("exp_noise", f"Noise + Position Emb\n(B, T, {d}d)\nt ~ importance sampling",
               fillcolor="#FFCCBC")
        c.node("exp_denoiser", f"adaLN-Zero Denoiser\n{model._text_expander_layers}L, {d}d, {model.config.n_heads}H\n+ cross-attention to memory",
               fillcolor="#FFCCBC")
        c.node("exp_decode", f"NN Decode\n→ nearest token embedding",
               fillcolor="#FFCCBC")
        c.node("exp_length", f"Length Head\ncond → token count",
               fillcolor="#FFCCBC")

    # --- Conditioning signals ---
    g.node("timestep", f"Timestep t\n→ sinusoidal embed",
           fillcolor="#F3E5F5", shape="oval")

    # --- Output ---
    g.node("output", f"Output Text\n(B, T) BPE tokens",
           fillcolor="#E8F4FD", shape="oval")

    # --- Edges ---
    # Input flow
    g.edge("input", "emb")
    g.edge("emb", "comp_self_attn")
    g.edge("comp_self_attn", "comp_pool")
    g.edge("comp_pool", "bottleneck")

    # Dynamics flow
    g.edge("bottleneck", "dyn_cat")
    g.edge("mode", "dyn_cat")
    g.edge("dyn_cat", "dyn_transformer")
    g.edge("dyn_transformer", "dyn_strip")
    g.edge("dyn_strip", "bn_post")
    g.edge("bottleneck", "dyn_strip", style="dashed", label="residual",
           color="#888888")

    # Chain unrolling: post-dynamics bottleneck feeds back into dynamics
    g.edge("bn_post", "dyn_cat", label="chain unroll\n(N-1 steps)",
           style="dotted", color="#2196F3", constraint="false")

    # Expander flow
    g.edge("bn_post", "exp_pool", label="pool")
    g.edge("bn_post", "exp_memory", label="project")
    g.edge("bn_post", "exp_length")
    g.edge("exp_noise", "exp_denoiser", label="input x")
    g.edge("exp_pool", "exp_denoiser", label="adaLN cond",
           style="dashed", color="#FF5722")
    g.edge("exp_memory", "exp_denoiser", label="cross-attn",
           style="dashed", color="#FF5722")
    g.edge("timestep", "exp_denoiser", label="adaLN time",
           style="dashed", color="#9C27B0")
    g.edge("exp_denoiser", "exp_decode")
    g.edge("exp_length", "exp_decode", style="dashed", label="truncate")
    g.edge("exp_decode", "output")

    # Iterative refinement loop
    g.edge("exp_denoiser", "exp_noise", label="re-noise\n(multi-step)",
           style="dotted", color="#666666", constraint="false")

    return g


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="Model checkpoint directory")
    group.add_argument("--config", help="JSON config file (no checkpoint needed)")
    parser.add_argument("--output", default=None, help="Output path (without extension)")
    parser.add_argument("--format", default="png", choices=["png", "svg", "pdf"])
    args = parser.parse_args()

    import torch
    from twm.text_dynamics_model import TextDynamicsModel
    from twm.config import ModelConfig
    from twm.domain_bpe import DomainBPETokenizer

    if args.run_dir:
        model = TextDynamicsModel.load(args.run_dir, device="cpu")
    else:
        with open(args.config) as f:
            cfg = json.load(f)
        model_config = ModelConfig(
            d_model=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("dynamics_layers", 2),
            d_ff=cfg.get("d_ff", 256),
            max_triples=cfg.get("max_triples", 8),
        )
        tokenizer = DomainBPETokenizer.from_pretrained(
            cfg.get("tokenizer_pretrained", "gpt2"),
            max_length=cfg.get("max_text_tokens", 128),
        )
        model = TextDynamicsModel(
            config=model_config, domain_tokenizer=tokenizer,
            text_compressor_layers=cfg.get("compressor_layers", 3),
            text_expander_layers=cfg.get("expander_layers", 3),
            dynamics_layers=cfg.get("dynamics_layers", 2),
            max_text_tokens=cfg.get("max_text_tokens", 128),
        )
    model.eval()

    total, trainable = count_params(model)
    print(f"Model: {format_params(total)} total, {format_params(trainable)} trainable")

    g = build_graph(model)
    g.format = args.format

    out_path = args.output or "architecture"
    try:
        g.render(out_path, cleanup=True)
        print(f"Saved: {out_path}.{args.format}")
    except Exception:
        # Save DOT source if graphviz binary not available
        dot_path = out_path + ".dot"
        with open(dot_path, "w") as f:
            f.write(g.source)
        print(f"Graphviz binary not found. Saved DOT source: {dot_path}")
        print(f"Render with: dot -T{args.format} {dot_path} -o {out_path}.{args.format}")
        print(f"Or paste into https://dreampuf.github.io/GraphvizOnline/")


if __name__ == "__main__":
    main()
