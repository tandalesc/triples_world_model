# References

Papers and systems referenced during TWM development.

## Architecture

| Reference | Year | What We Used |
|-----------|------|-------------|
| DDPM (Ho et al.) | 2020 | Continuous noise framework `x_t = sqrt(α)*x_0 + sqrt(1-α)*ε`. Adapted cosine schedule with `alpha_min` clamping for token embedding space. x0-prediction over ε-prediction. |
| DiT (Peebles & Xie) | 2023 | adaLN-Zero conditioning. Reshapes activation geometry before computation (vs FiLM which only transforms output). Zero-init gate for stable training ramp-up. |
| LLaDA (Nie et al.) | 2025 | Inspired initial masked discrete approach. Revealed fundamental limitation: discrete masking has a task discontinuity at mask_ratio=1.0. |
| CFG (Ho & Salimans) | 2022 | Conditioning dropout during training. Turned out unnecessary with compressor/expander — joint training produces strong enough conditioning natively. |
| FiLM (Perez et al.) | 2018 | Tried and rejected. Mean-pooling triples into (γ,β) discards relational structure. Wrong tool for transformers. |

## Positioning

| Reference | Year | Relationship to TWM |
|-----------|------|-------------|
| JEPA (LeCun) | 2022 | Closest philosophical match. Both: world model in latent space, decoding is separate. Difference: JEPA's latent is opaque; TWM's is structured triples. |
| Perceiver IO (Jaegle et al.) | 2021 | Closest architectural match. Shared latent bottleneck with modality-specific encoders/decoders. Difference: Perceiver's latent is unstructured; TWM's has relational structure. |
| GATO (Reed et al.) | 2022 | Serializing all modalities into one token stream destroys structure. Mediocre at everything. |
| DreamFusion (Poole et al.) | 2022 | Diffusion as renderer, not as world model. TWM inverts the typical architecture. |
| GLOM (Hinton) | 2021 | Hierarchy in architecture (columns/levels) vs TWM's hierarchy in data (meta-triples). |
| VAEs (Kingma & Welling) | 2014 | Compressor/expander is a deterministic-encoder VAE. Posterior collapse = conditioning collapse. Same problem, same class of solution. |
| ECS (Unity DOTS, Bevy) | — | TWM is "ECS where the systems are learned from data." Directly legible to game devs. |
| Production Rules (CLIPS, Rete) | — | Same structure (bag of facts + rules), but rules are learned not authored. No rule explosion. |

## Embeddings & Data

| Reference | Year | What We Used |
|-----------|------|-------------|
| GloVe (Pennington et al.) | 2014 | Original token embeddings. Real words with distinct vectors essential — compound tokens ("person_a") get near-identical vectors. |
| Sentence-Transformers (Reimers & Gurevych) | 2019 | BPE embedding initialization. Must freeze — trainable embeddings collapse under continuous noise. |
| T5 (Raffel et al.) | 2020 | Subword vocabulary gives compositional generalization. Led to domain-specific BPE. |
| ATOMIC 2020 (Hwang et al.) | 2021 | Primary diffusion decoder test bed. Open-vocab, inherently ambiguous. 10K subset. |
| WebNLG (Gardent et al.) | 2017 | Validated compressor/expander at 100% exact match on concrete entities. |
| ProPara (Dalvi et al.) / OpenPI (Tandon et al.) | 2018/2020 | Location + attribute triples for original TWM training set. |
| REBEL (Huguet Cabot & Navigli) | 2021 | Potential data scaling source — seq2seq triple extraction from text. |

## Scene Representation (Future)

| Reference | Year | Connection |
|-----------|------|-----------|
| 3D Gaussian Splatting (Kerbl et al.) | 2023 | Explicit primitives for scenes, like triples for world state. TWM could drive Gaussians as a visual decoder. |
| NeRF (Mildenhall et al.) | 2020 | Implicit→explicit transition parallels TWM's philosophy. |

## Cognitive Science

| Reference | Year | Connection |
|-----------|------|-----------|
| Hofstadter (Fluid Concepts) | 1995 | Analogy = same concept in different structural roles. Motivates unified (not split) embedding space. |
| Lakoff (Conceptual Metaphor) | 1980 | Metaphor maps entities→relations, not just entity→entity. Split embeddings hardcode ontology; unified enables fluidity. |
