# Theoretical Foundations

Core principles underlying TWM's design.

## W-Space

A world state of N triples lives in Sym^N(R^d) — the symmetric product quotiented by permutation invariance. The transformer respects this geometry by construction. Valid states occupy a low-dimensional submanifold **W** within this space. The advance function is a near-identity map on W (biased by residual connections).

## The Bus

W is the universal interface. Encoders project in, decoders project out, dynamics operate within. No component talks to any other except through W. Analogous to Perceiver IO's shared latent bottleneck, but with explicit relational structure.

See also: [Perceiver IO, GATO comparison](references.md#positioning)

## Homoiconicity

Mode tokens, meta-triples, and state triples are all the same type of object in W. The transformer processes them identically through attention. This is what makes mode conditioning work without architecture changes — `(#mode, type, advance)` is just another triple.

## Compositional Closure

TWM(data) outputs triples. TWM(data) takes triples as input. TWMs compose freely: TWM_1(TWM_2(data)). No translation layers needed between stages.

## Hierarchy via Data, Not Architecture

Unlike GLOM (hierarchy in architecture), TWM puts hierarchy in data through meta-triples: `(arm, part_of, person)`, `(person, abstracts_as, agent)`. The flat transformer learns hierarchical propagation from these structural annotations.

See also: [GLOM comparison](references.md#positioning)

## Why Small Models Work

Structural priors (triple decomposition, set symmetry, residual dynamics, shared advance function) carry knowledge that parameters carry in unstructured models. The combinatorial advantage of compositional representation grows exponentially with entity/attribute count. This is why Mini (178K) matches Base (4.5M) on context-dependent reasoning.

The architecture doesn't depend on the specific (entity, attribute, value) format — it depends on the properties: structured atoms, unordered sets, shared continuous space, independently interpretable. Triples are the current instantiation.

## Diffusion in W-Space

Three geometric insights from Sprint 3 (see [problem/solution log](sprint3_diffusion_decoder.md)):

1. **Discrete masking has a task discontinuity.** At mask_ratio < 1.0 the model denoises; at 1.0 it must generate from nothing. Continuous Gaussian noise eliminates this. (Sprint 3, problems #1-2)

2. **Denoising and conditioning spaces must be geometrically separated.** If identical, the denoiser bypasses conditioning via proximity shortcut. Optimal: frozen BPE embeddings for denoising targets, learned W-space for conditioning, connected through cross-attention/adaLN. (Sprint 3, problems #3, #6, #7)

3. **Position must route through noise-free pathways.** Positional embeddings added to noisy inputs get buried. Route position through adaLN alongside content conditioning. (Sprint 3, problem #12)

## Continuous M*

The mode-conditioned transformations form a smooth manifold M*. Nearby mode tokens → similar transformations. Trained anchors (advance, identity, instruct) are discrete points; the space between them may yield meaningful interpolations. Empirically untested but architecturally supported by attention's smooth dependence on inputs.

## Von Neumann Analogy

W as memory, mode-conditioned transformations as instruction set, controller selects mode tokens → programmable processor over relational state. Variable-width state (grouping/expansion operators) pushes toward Turing completeness. Speculative but directionally useful framing.
