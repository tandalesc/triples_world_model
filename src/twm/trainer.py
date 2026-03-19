"""Training orchestrator for config-driven experiments.

Usage:
    config = TrainingConfig.load("configs/v18_mini64.json")
    trainer = Trainer(config)
    trainer.run()
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from .config import ModelConfig
from .domain_bpe import DomainBPETokenizer
from .text_model import TextWorldModel
from .text_dynamics_model import TextDynamicsModel
from .text_dataset import TextDataset
from .text_pair_dataset import TextPairDataset
from .training_config import TrainingConfig, StageConfig, PhaseConfig
from .training_losses import sample_timestep, compute_diffusion_loss
from .training_eval import assess, print_samples, format_metrics, diagnose_mode_attention, save_latent_snapshot


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Config-driven training orchestrator."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = _resolve_device(config.device)
        self.tokenizer = DomainBPETokenizer.load(
            config.tokenizer_path, max_length=config.max_text_tokens
        )
        self.model = self._build_model()
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Save config for reproducibility
        self.config.save(self.out_dir / "training_config.json")

    def _build_model(self) -> torch.nn.Module:
        model_config = self.config.build_model_config()
        c = self.config
        dyn_layers = c.dynamics_layers if c.dynamics_layers is not None else model_config.n_layers

        compressor_kwargs = {}
        if hasattr(c, 'compressor_type'):
            compressor_kwargs['compressor_type'] = c.compressor_type
            compressor_kwargs['compressor_denoise_steps'] = c.compressor_denoise_steps
            compressor_kwargs['compressor_random_k'] = c.compressor_random_k
            compressor_kwargs['compressor_k_min'] = c.compressor_k_min

        if c.model_type == "io":
            model = TextWorldModel(
                config=model_config, domain_tokenizer=self.tokenizer,
                text_compressor_layers=c.text_compressor_layers,
                text_expander_layers=c.text_expander_layers,
                max_text_tokens=c.max_text_tokens,
                dropout=c.dropout, alpha_min=c.alpha_min,
                vae=c.vae,
                **compressor_kwargs,
            )
        else:
            model = TextDynamicsModel(
                config=model_config, domain_tokenizer=self.tokenizer,
                text_compressor_layers=c.text_compressor_layers,
                text_expander_layers=c.text_expander_layers,
                dynamics_layers=dyn_layers,
                max_text_tokens=c.max_text_tokens,
                dropout=c.dropout, alpha_min=c.alpha_min,
                vae=c.vae,
                **compressor_kwargs,
            )
        model.init_embeddings()
        return model.to(self.device)

    def run(self):
        """Run all stages in sequence."""
        print(f"=== Config-driven Training ===")
        print(f"Device: {self.device}")
        print(f"Model: {self.config.model_type} ({self.config.profile})")
        print(f"Total: {self.model.param_count():,} params")

        for stage in self.config.stages:
            self._run_stage(stage)

    def _run_stage(self, stage: StageConfig):
        print(f"\n{'#'*60}")
        print(f"# Stage: {stage.name}")
        print(f"{'#'*60}")

        data_dir = Path(self.config.data_dir)

        # Load pretrained weights
        pretrained = self._resolve_pretrained(stage)
        if pretrained:
            self._load_pretrained(pretrained)

        # Apply freezes — auto-freeze dynamics during IO stages (unless joint training)
        freeze = list(stage.freeze)
        if stage.dataset == "identity" and hasattr(self.model, "dynamics"):
            if "dynamics" not in freeze and not stage.joint:
                freeze.append("dynamics")

        # Default unfreeze: length_head when expander is frozen (it must adapt
        # to transformed bottlenecks from dynamics). Override with explicit
        # unfreeze=[] to keep it frozen (e.g. uniform-length data).
        unfreeze = stage.unfreeze
        if unfreeze is None:
            unfreeze = ["length_head"] if "expander" in freeze else []

        self._apply_freeze(freeze, unfreeze)

        # Load dataset
        train_ds, ds_for_assessment = self._load_datasets(stage, data_dir)

        # Print stage info
        trainable_count = self.model.trainable_param_count()
        frozen_parts = freeze if freeze else ["none"]
        unfrozen_parts = unfreeze if unfreeze else []
        print(f"  Frozen: {', '.join(frozen_parts)}")
        if unfrozen_parts:
            print(f"  Unfrozen overrides: {', '.join(unfrozen_parts)}")
        print(f"  Trainable: {trainable_count:,} params")
        print(f"  Train: {len(train_ds)} examples")
        print(f"  Assessment: {len(ds_for_assessment)} examples")

        # Run each phase
        for pi, phase in enumerate(stage.phases):
            phase_name = f"{stage.name}_phase{pi + 1}"
            # Rebuild optimizer between phases (fresh momentum)
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            lr = phase.lr if phase.lr is not None else stage.lr
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=stage.weight_decay)
            self._run_phase(phase_name, phase, optimizer, train_ds, ds_for_assessment, stage)

    def _run_phase(self, name: str, phase: PhaseConfig, optimizer,
                   train_ds, ds_for_assessment, stage: StageConfig):
        phase_dir = self.out_dir / name
        phase_dir.mkdir(parents=True, exist_ok=True)
        c = self.config
        is_dynamics = stage.dataset in ("qa", "mode_warmup") or stage.joint

        metric_key = phase.metric  # "tok_acc" or "exact"

        print(f"\n{'='*60}")
        t_min_anneal = phase.t_min_end is not None and phase.t_min_end != phase.t_min
        if t_min_anneal:
            print(f"Phase: {name}  t in [{phase.t_min}→{phase.t_min_end}, {phase.t_max}]  bias={phase.bias_power}  metric={metric_key}")
        else:
            print(f"Phase: {name}  t in [{phase.t_min}, {phase.t_max}]  bias={phase.bias_power}  metric={metric_key}")
        print(f"{'='*60}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(phase.epochs, 1)
        )

        n_train = len(train_ds)
        best_metric = -1.0
        best_epoch = 0
        no_improve = 0
        history = []
        pca_basis = None

        # Initial assessment
        init_m = assess(self.model, ds_for_assessment, self.device, self.tokenizer,
                        n_examples=64, n_steps=c.denoise_steps)
        print(f"  init: {format_metrics(init_m)}", flush=True)

        for epoch in range(1, phase.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_ce = 0.0
            epoch_bn = 0.0
            epoch_bn_e = 0.0
            epoch_bn_a = 0.0
            epoch_bn_v = 0.0
            epoch_role = 0.0
            epoch_kl = 0.0
            epoch_spec = 0.0
            n_batches = 0

            # β annealing: linear ramp from 0 to kl_weight over kl_anneal_epochs
            if c.kl_weight > 0 and c.kl_anneal_epochs > 0:
                eff_kl_weight = c.kl_weight * min(1.0, epoch / c.kl_anneal_epochs)
            else:
                eff_kl_weight = c.kl_weight
            perm = torch.randperm(n_train)

            for start in range(0, n_train - c.batch_size + 1, c.batch_size):
                idx = perm[start:start + c.batch_size]
                B = idx.shape[0]
                # Anneal t_min if t_min_end is set
                if t_min_anneal:
                    progress = (epoch - 1) / max(phase.epochs - 1, 1)
                    cur_t_min = phase.t_min + (phase.t_min_end - phase.t_min) * progress
                else:
                    cur_t_min = phase.t_min
                timestep = sample_timestep(B, self.device, cur_t_min, phase.t_max, phase.bias_power)

                if is_dynamics:
                    # Dynamics path: route through dynamics core with mode conditioning.
                    # For joint training on identity data, use same text as input/output
                    # with mode=0 (identity) so dynamics gradients flow back to compressor.
                    if stage.joint and stage.dataset == "identity":
                        in_ids = train_ds._text_token_ids[idx]
                        in_pad = train_ds._text_pad_mask[idx]
                        out_ids = in_ids
                        out_pad = in_pad
                        out_len = train_ds._text_lengths[idx]
                        batch_modes = torch.zeros(B, dtype=torch.long)
                    else:
                        in_ids = train_ds._input_token_ids[idx]
                        in_pad = train_ds._input_pad_mask[idx]
                        out_ids = train_ds._output_token_ids[idx]
                        out_pad = train_ds._output_pad_mask[idx]
                        out_len = train_ds._output_lengths[idx]
                        batch_modes = train_ds._modes[idx]
                    loss, batch_m = compute_diffusion_loss(
                        self.model,
                        in_ids, in_pad, out_ids, out_pad,
                        out_len, self.device, timestep,
                        mode_ids=batch_modes,
                        aux_ce_weight=c.aux_ce_weight, length_weight=c.length_weight,
                        bottleneck_weight=c.bottleneck_weight,
                        role_prior_weight=c.role_prior_weight,
                        bn_role_weights=tuple(c.bn_role_weights) if c.bn_role_weights else None,
                        detach_dynamics_expander=c.detach_dynamics_expander,
                        kl_weight=eff_kl_weight,
                        spectral_weight=c.spectral_weight,
                    )
                else:
                    loss, batch_m = compute_diffusion_loss(
                        self.model,
                        train_ds._text_token_ids[idx], train_ds._text_pad_mask[idx],
                        None, None,
                        train_ds._text_lengths[idx], self.device, timestep,
                        mode_ids=None,
                        aux_ce_weight=c.aux_ce_weight, length_weight=c.length_weight,
                        role_prior_weight=c.role_prior_weight,
                        kl_weight=eff_kl_weight,
                        spectral_weight=c.spectral_weight,
                    )

                optimizer.zero_grad()
                loss.backward()
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                epoch_mse += batch_m.get("mse", 0.0)
                epoch_ce += batch_m.get("ce", 0.0)
                epoch_bn += batch_m.get("bn_loss", 0.0)
                epoch_bn_e += batch_m.get("bn_e", 0.0)
                epoch_bn_a += batch_m.get("bn_a", 0.0)
                epoch_bn_v += batch_m.get("bn_v", 0.0)
                epoch_role += batch_m.get("role_loss", 0.0)
                epoch_kl += batch_m.get("kl", 0.0)
                epoch_spec += batch_m.get("spec", 0.0)
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_mse = epoch_mse / max(n_batches, 1)
            avg_ce = epoch_ce / max(n_batches, 1)
            avg_bn = epoch_bn / max(n_batches, 1)
            avg_bn_e = epoch_bn_e / max(n_batches, 1)
            avg_bn_a = epoch_bn_a / max(n_batches, 1)
            avg_bn_v = epoch_bn_v / max(n_batches, 1)
            avg_role = epoch_role / max(n_batches, 1)
            avg_kl = epoch_kl / max(n_batches, 1)
            avg_spec = epoch_spec / max(n_batches, 1)

            if epoch % c.log_every == 0 or epoch == 1:
                self.model.eval()
                gen_m = assess(self.model, ds_for_assessment, self.device, self.tokenizer,
                               n_examples=64, n_steps=c.denoise_steps)
                gen_cache = gen_m.pop("_gen", None)

                log = f"Epoch {epoch:4d} | loss {avg_loss:.4f} mse={avg_mse:.4f} ce={avg_ce:.4f}"
                if c.bottleneck_weight > 0 or c.bn_role_weights:
                    log += f" bn={avg_bn:.4f}"
                if c.bn_role_weights:
                    log += f" (e={avg_bn_e:.4f} a={avg_bn_a:.4f} v={avg_bn_v:.4f})"
                if c.role_prior_weight > 0:
                    log += f" role={avg_role:.4f}"
                if c.kl_weight > 0:
                    log += f" kl={avg_kl:.4f} (β={eff_kl_weight:.4f})"
                if c.spectral_weight > 0:
                    log += f" spec={avg_spec:.4f}"
                log += f" | {format_metrics(gen_m)}"

                cur_metric = gen_m[metric_key]
                if cur_metric > best_metric:
                    best_metric = cur_metric
                    best_epoch = epoch
                    no_improve = 0
                    torch.save(self.model.state_dict(), phase_dir / "model_best.pt")
                    log += " *"
                else:
                    no_improve += c.log_every

                print(log, flush=True)
                history.append({"epoch": epoch, "loss": avg_loss, **gen_m})

                if epoch == 1 or epoch % c.diagnostic_every == 0:
                    print_samples(self.model, ds_for_assessment, self.device, self.tokenizer,
                                  n=5, n_steps=c.denoise_steps, gen_cache=gen_cache)
                    # Mode-attention diagnostic for mode_warmup stages
                    if stage.dataset == "mode_warmup":
                        diagnose_mode_attention(self.model, ds_for_assessment, self.device)

                if phase.patience > 0 and no_improve >= phase.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(best {metric_key}={best_metric:.4f} at epoch {best_epoch})")
                    break

            if c.snapshot_every > 0 and (epoch == 1 or epoch % c.snapshot_every == 0):
                pca_basis = save_latent_snapshot(
                    self.model, ds_for_assessment, self.device,
                    epoch, name, phase_dir, pca_basis=pca_basis
                )

        torch.save(self.model.state_dict(), phase_dir / "model_final.pt")
        with open(phase_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n{name} done. Best {metric_key}: {best_metric:.4f} at epoch {best_epoch}")

    def _load_datasets(self, stage: StageConfig, data_dir: Path):
        c = self.config
        max_ex = stage.max_examples if stage.max_examples is not None else c.max_examples
        if stage.dataset == "identity":
            train_ds = TextDataset(
                data_dir / "identity_train.jsonl", self.tokenizer,
                max_text_tokens=c.max_text_tokens, max_examples=max_ex,
            )
            assessment_path = data_dir / "identity_test.jsonl"
            ds_for_assessment = train_ds
            if assessment_path.exists():
                ds_for_assessment = TextDataset(
                    assessment_path, self.tokenizer, max_text_tokens=c.max_text_tokens
                )
        elif stage.dataset == "mode_warmup":
            train_ds = TextPairDataset(
                data_dir / "mode_warmup_train.jsonl", self.tokenizer,
                max_text_tokens=c.max_text_tokens, max_examples=max_ex,
            )
            n_id = (train_ds._modes == 0).sum().item()
            n_rev = (train_ds._modes == 2).sum().item()
            print(f"  Dataset: {len(train_ds)} ({n_id} identity, {n_rev} reverse)")
            assessment_path = data_dir / "mode_warmup_test.jsonl"
            ds_for_assessment = train_ds
            if assessment_path.exists():
                ds_for_assessment = TextPairDataset(
                    assessment_path, self.tokenizer, max_text_tokens=c.max_text_tokens
                )
        else:  # "qa", "qa_balanced", or other pair datasets
            train_file = data_dir / f"{stage.dataset}_train.jsonl"
            train_ds = TextPairDataset(
                train_file, self.tokenizer,
                max_text_tokens=c.max_text_tokens, max_examples=max_ex,
            )
            n_id = (train_ds._modes == 0).sum().item()
            n_qa = (train_ds._modes == 1).sum().item()
            print(f"  Dataset: {len(train_ds)} ({n_id} identity, {n_qa} Q&A)")
            assessment_path = data_dir / "qa_test.jsonl"
            ds_for_assessment = train_ds
            if assessment_path.exists():
                ds_for_assessment = TextPairDataset(
                    assessment_path, self.tokenizer, max_text_tokens=c.max_text_tokens
                )
        return train_ds, ds_for_assessment

    def _apply_freeze(self, components: list[str], unfreeze: list[str] | None = None):
        """Freeze named model components, then selectively unfreeze overrides.

        Args:
            components: top-level modules to freeze ("compressor", "expander", "dynamics")
            unfreeze: sub-components to unfreeze after freezing ("length_head").
                      Useful when freezing "expander" but the length head must
                      adapt to transformed bottlenecks from dynamics.
        """
        # Unfreeze all (except shared_token_emb which is always frozen)
        for name, p in self.model.named_parameters():
            if "shared_token_emb" not in name:
                p.requires_grad = True

        freeze_map = {
            "compressor": getattr(self.model, "text_compressor", None),
            "expander": getattr(self.model, "text_expander", None),
            "dynamics": getattr(self.model, "dynamics", None),
            "embeddings": getattr(self.model, "shared_token_emb", None),
        }
        # Additional modules/parameters associated with "dynamics"
        dynamics_extras = []
        for attr in ("mode_emb", "mode_role_emb"):
            obj = getattr(self.model, attr, None)
            if obj is not None:
                dynamics_extras.append(obj)

        for comp_name in components:
            module = freeze_map.get(comp_name)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
            if comp_name == "dynamics":
                for obj in dynamics_extras:
                    if isinstance(obj, nn.Parameter):
                        obj.requires_grad = False
                    elif isinstance(obj, nn.Module):
                        for p in obj.parameters():
                            p.requires_grad = False

        # Selectively unfreeze sub-components
        unfreeze_map = {
            "length_head": None,
        }
        # Resolve length_head location
        expander = getattr(self.model, "text_expander", None)
        if expander is not None:
            unfreeze_map["length_head"] = getattr(expander, "length_head", None)

        for name in (unfreeze or []):
            module = unfreeze_map.get(name)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

    def _resolve_pretrained(self, stage: StageConfig) -> str | None:
        """Resolve pretrained checkpoint path."""
        if stage.pretrained:
            return stage.pretrained

        # Auto-detect from previous stage's best checkpoint
        stages = self.config.stages
        stage_idx = next((i for i, s in enumerate(stages) if s is stage), -1)
        if stage_idx <= 0:
            return None

        # Look for the last phase's best checkpoint from the previous stage
        prev = stages[stage_idx - 1]
        n_phases = len(prev.phases)
        prev_phase_name = f"{prev.name}_phase{n_phases}"
        ckpt = self.out_dir / prev_phase_name / "model_best.pt"
        if ckpt.exists():
            return str(ckpt)
        return None

    def _load_pretrained(self, path: str):
        """Load pretrained weights with shape-compatible partial loading."""
        print(f"  Loading pretrained: {path}")
        state = torch.load(path, map_location=self.device, weights_only=True)
        model_state = self.model.state_dict()
        loaded = []
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded.append(k.split(".")[0])
        self.model.load_state_dict(model_state)
        print(f"  Loaded: {sorted(set(loaded))}")
