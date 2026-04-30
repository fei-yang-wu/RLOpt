# IPMD + VQ-VAE Skill Codebook for Hierarchical Imitation

Technical report. Branch `vqvae-impl` in both `RLOpt/` and `IsaacLab-Imitation/`.

## 1. Goal

Replace the current single-step deterministic latent encoder in IPMD with a
**windowed VQ-VAE** so the latent command becomes a **discrete skill code**
held across many env steps. This sets up a hierarchical control stack:

- **Low-level (LL) policy** — current IPMD actor; conditions on a code embedding.
- **High-level (HL) planner** — separate module (future work) that picks codes
  from the codebook every `code_period` steps. Standard categorical policy /
  autoregressive transformer over skill tokens.

The new latent learner (`patch_vqvae`) is a drop-in alternative to the existing
`patch_autoencoder` and lives behind a config flag.

## 2. Background

### 2.1 Why a window

Single-frame posterior conflates pose with skill. A window of expert states
forces the encoder to compress *temporally extended behavior* (subskill,
phrase). The IsaacLab-Imitation env already exposes window observations via
`expert_window/expert_motion`, `expert_window/expert_anchor_pos_b`,
`expert_window/expert_anchor_ori_b` (built by
`_build_expert_window_terms` in `imitation_rl_env.py`). Window length =
`latent_patch_past_steps + 1 + latent_patch_future_steps`.

### 2.2 Why discrete latents (VQ over VAE)

For HL planning a categorical action space simplifies everything: BC = cross-
entropy on tokens, RL = categorical policy, transformers/LLMs slot in
naturally. Continuous latent regression is harder and less composable.

### 2.3 Quantizer choices implemented

| Quantizer | Codebook params | Collapse risk | Aux loss | Capacity |
|---|---|---|---|---|
| **FSQ** (default) | none (fixed grid) | none | none | ∏ levels |
| **VQ-EMA** | learned + EMA | medium | commitment | K |
| **Gumbel-Softmax** | learned embedding + logits | medium | KL-to-uniform + entropy | K |

- **FSQ** (Mentzer 2023): per-dim scalar quantization on a fixed grid. No
  codebook, no aux loss, no collapse. Output dim is small (5 by default ⇒
  12 800 effective codes); we project to `latent_dim` with a small linear.
- **VQ-EMA** (van den Oord + DALL·E line): nearest-neighbor on a learned
  codebook, EMA cluster updates, commitment loss, k-means init, dead-code
  revival.
- **Gumbel-Softmax**: encoder emits logits over the codebook, sample with
  Gumbel noise, optional straight-through hard forward. Provides true reparam
  gradients; supports KL-to-uniform regularizer + marginal-usage entropy bonus
  to prevent collapse. Familiar to the lab; useful as a backup / ablation.

### 2.4 Why a code period

A code per env step turns the HL planner into a high-frequency controller and
loses the "skill" abstraction. We hold one sampled code per env for
`code_period` steps; the encoder's window summarizes the upcoming `K` steps,
and the HL planner replans only at chunk boundaries (~0.6 s @ 50 Hz with
period 30). Aligns with classical *options*.

## 3. Architecture

```
expert window x_{t-P:t}         ┐
   ↓ encoder (MLP)              │ training only
   z_e ∈ R^d                    │
   ↓ quantizer (FSQ|VQ|Gumbel)  │
   z_q, code_id                 │
   ├──► decoder ──► x̂           │ recon loss → MSE(x̂, x)
   └──► action_decoder ──► â    │ optional MSE(â, expert_action)
   ↓ project (linear if needed) │
   latent_dim embedding         ┘  ↑ also published at rollout time

rollout (collector):
   posterior obs (expert_window/*) ──► encode ──► quantize ──► code embedding
   held for `code_period` env steps per env, then re-encode + re-quantize.
```

### 3.1 Pluggable in IPMD

- All work lives in the `BaseLatentLearner` interface. IPMD code path
  unchanged except the posterior branch in `_inject_latent_command`, which now
  dispatches to `infer_collector_latents` if the learner exposes it
  (`patch_vqvae` does, `patch_autoencoder` does not). All other learners
  unaffected.

## 4. Files changed

### 4.1 RLOpt (worktree `RLOpt-vqvae`, branch `vqvae-impl`)

- `rlopt/agent/imitation/latent_learning.py`
  - `FSQQuantizer`, `EMAVQQuantizer`, `GumbelQuantizer` (new modules).
  - `PatchVQVAELatentLearner` (new learner), registered key `"patch_vqvae"`.
  - Diagnostic metrics: codebook perplexity, dead-code count, recon MSE,
    action-recon MSE, commit loss, gumbel KL, gumbel τ, code usage entropy,
    revived-codes count.
  - `infer_collector_latents` for hold-across-steps (per-env countdown,
    renew on done OR expiry).

- `rlopt/agent/ipmd/ipmd.py`
  - `IPMDLatentLearningConfig` extended (see §6 for config reference).
  - `_inject_latent_command` posterior path uses
    `learner.infer_collector_latents(td)` when present.

### 4.2 IsaacLab-Imitation (worktree `IsaacLab-Imitation-vqvae`, branch `vqvae-impl`)

- `tasks/manager_based/imitation/config/g1/imitation_g1_latent_vqvae_env_cfg.py`
  — `ImitationG1LatentVQVAEEnvCfg` extends the latent G1 env with
  `latent_patch_past_steps = 8`, `latent_patch_future_steps = 0`.
  All other observation groups unchanged; `expert_window/*` already exists in
  `G1ObservationCfg.ExpertWindowCfg` and is auto-synced via
  `_sync_expert_window_observation_params`.

- `tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_vqvae_cfg.py`
  — `G1ImitationLatentRLOptIPMDVQVAEConfig`. Defaults:
  - `method = "patch_vqvae"`, `quantizer = "fsq"`, `fsq_levels = [8,8,8,5,5]`.
  - Posterior input keys = `expert_window/expert_motion`,
    `expert_window/expert_anchor_pos_b`, `expert_window/expert_anchor_ori_b`.
  - `recon_coeff = 1.0`, `action_recon_coeff = 0.5`,
    `code_period = 30`, `latent_steps_min = latent_steps_max = 30`,
    `command_source = "posterior"`, `latent_dim = 64`,
    `train_posterior_through_policy = False`.
  - Probe enabled.

- `tasks/manager_based/imitation/config/g1/__init__.py`
  — registers task ID **`Isaac-Imitation-G1-Latent-VQVAE-v0`** with
  `rlopt_ipmd_vqvae_cfg_entry_point` (and aliases `rlopt_cfg_entry_point` /
  `rlopt_ipmd_cfg_entry_point` to the same config so `--algo IPMD` works).

## 5. Usage

Smoke-test (uses `train.py`, since direct `import rlopt` pulls IsaacLab/IsaacSim
through torchrl's gym wrapper):

```bash
cd /home/fwu91/Documents/Research/SkillLearning/IsaacLab-Imitation-vqvae

conda run -n SkillLearning-vqvae python scripts/rlopt/train.py \
    --task Isaac-Imitation-G1-Latent-VQVAE-v0 \
    --algo IPMD \
    --max_iterations 1 --num_envs 32 --headless \
    env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

Switch quantizer via Hydra override:

```bash
... ipmd.latent_learning.quantizer=gumbel \
    ipmd.latent_learning.gumbel_kl_to_uniform_coeff=0.01 \
    ipmd.latent_learning.code_usage_entropy_coeff=0.01

... ipmd.latent_learning.quantizer=vq_ema \
    ipmd.latent_learning.codebook_size=512 \
    ipmd.latent_learning.commitment_coeff=0.25 \
    ipmd.latent_learning.dead_code_reset_iters=1000
```

Lint:

```bash
conda run -n SkillLearning-vqvae ruff check RLOpt-vqvae IsaacLab-Imitation-vqvae
conda run -n SkillLearning-vqvae ruff format --check RLOpt-vqvae IsaacLab-Imitation-vqvae
```

## 6. Config reference (`ipmd.latent_learning.*`)

### Common
| Field | Default | Notes |
|---|---|---|
| `method` | `"patch_autoencoder"` | Set to `"patch_vqvae"` to use VQVAE. |
| `posterior_input_keys` | `[]` | Should point to `expert_window/*` keys for windowed encoding. |
| `patch_past_steps` / `patch_future_steps` | `0` / `0` | Algorithm-side hint; env supplies the actual window via `latent_patch_past_steps`. Keep them aligned. |
| `encoder_hidden_dims` | `[256, 256]` | MLP encoder hidden widths. |
| `decoder_hidden_dims` | `[256, 256]` | MLP decoder hidden widths. |
| `lr` | `3e-4` | Latent-learner optimizer lr. |
| `grad_clip_norm` | `1.0` | Latent-learner grad clip. |
| `recon_coeff` | `0.0` | Window reconstruction MSE weight. Set > 0 for VQ-VAE. |
| `weight_decay_coeff` | `0.0` | Optional L2 on encoder/decoder. |

### VQ-VAE specific
| Field | Default | Notes |
|---|---|---|
| `quantizer` | `"fsq"` | One of `"fsq" \| "vq_ema" \| "gumbel"`. |
| `fsq_levels` | `[8,8,8,5,5]` | Effective codebook size = ∏ levels (default = 12 800). |
| `codebook_size` | `512` | Used by `vq_ema` and `gumbel`. |
| `codebook_embed_dim` | `None` | None → use `latent_dim`. |
| `commitment_coeff` | `0.25` | β in van den Oord. `vq_ema` only. |
| `ema_decay` | `0.99` | EMA codebook update decay. `vq_ema` only. |
| `dead_code_reset_iters` | `0` | 0 disables. `vq_ema` only. |
| `gumbel_tau_start` / `gumbel_tau_end` | `1.0` / `0.3` | Linear anneal. |
| `gumbel_tau_anneal_iters` | `200_000` | Update calls until τ=end. |
| `gumbel_hard` | `True` | ST hard forward, soft backward. |
| `gumbel_kl_to_uniform_coeff` | `0.0` | KL(q‖U) regularizer. |
| `code_usage_entropy_coeff` | `0.0` | Marginal usage entropy bonus. |
| `action_recon_coeff` | `0.0` | If > 0, decode `expert_action` from `z_q`. |
| `code_period` | `30` | Env steps per held code. |
| `latent_dropout_to_random_code_prob` | `0.0` | Random-code substitution during PPO updates. **Not yet wired** (see §8). |

## 7. Diagnostics logged

Per `iterate()` call:

- `train/ipmd/vqvae_total_loss`
- `train/ipmd/vqvae_recon_mse`
- `train/ipmd/vqvae_action_recon_mse`
- `train/ipmd/vqvae_commit_loss` (vq_ema)
- `train/ipmd/vqvae_gumbel_kl_uniform`, `vqvae_gumbel_tau` (gumbel)
- `train/ipmd/vqvae_code_usage_entropy`
- `train/ipmd/vqvae_codebook_perplexity` — **primary collapse indicator**
- `train/ipmd/vqvae_dead_codes`, `vqvae_codes_revived`
- `train/ipmd/vqvae_codebook_size`
- `train/ipmd/vqvae_weight_decay`

Plus the existing IPMD reward and PPO metrics. The optional `FeatureProbe`
(reused from `policy_kl_bottleneck`) measures how much of the reward-input
state the latent linearly preserves.

## 8. Known caveats / not-yet-wired

1. **`latent_dropout_to_random_code_prob` is parsed but not applied** — the
   PPO update path does not currently swap inferred `z_q` with a random
   codebook embedding. Easy follow-up: add a hook in
   `_backward_ppo_terms` / posterior recompute that, with prob `p`, replaces
   the latent with a uniformly-sampled codebook entry. Especially important
   for `vq_ema`/`gumbel`; trivial for FSQ (sample random grid point).
2. **No sequence encoder yet** — the encoder is a flatten + MLP over the
   window. Ablation: causal 1-D TCN or small GRU/Transformer over `(B, T, F)`.
   Add `encoder_type` in cfg.
3. **No HL planner module** — out of scope for this PR. Recommended next
   piece: tokenize expert trajectories (offline) + autoregressive transformer
   over codes + BC pretrain. Place under `rlopt/agent/imitation/hl_planner/`.
4. **Random-code training during BC** — `bc_coef > 0` will pull policy toward
   expert action under expert-encoded code only. Add code-dropout to BC for
   robustness.
5. **No FSQ output regularization** — encoder pre-quantization can drift far
   from the bounded grid. If saturation seen in `vqvae_recon_mse`, add a
   small weight on `||z_e||²` or a tanh inside the encoder head.
6. **`latent_steps_min == latent_steps_max == code_period`** in the new agent
   cfg — keeps the `LatentCommandController` resample cadence in lockstep
   with the learner's hold logic. If you change `code_period`, change the
   range too.
7. **Compatibility** — `patch_autoencoder` (current default) untouched.
   Existing tasks unaffected. New task ID is purely additive.

## 9. Future work (priority order)

1. **Wire random-code dropout during PPO updates** (§8.1). 30-line change,
   high payoff for downstream HL.
2. **Tokenize expert trajectories offline** — script
   `scripts/rlopt/tokenize_expert.py` reads expert dataset, runs the trained
   encoder + quantizer, dumps per-chunk `code_id` to disk. Feeds HL planner.
3. **HL planner v0**: autoregressive transformer over code tokens, BC + PPO
   on top. Frozen LL.
4. **Sequence encoder** (causal TCN / small GRU). Should help when
   `code_period` grows past ~60.
5. **RVQ fallback** if FSQ + EMA-VQ both hit recon ceiling. Drop-in
   alternative quantizer.
6. **Joint LL+HL fine-tune** via option-critic / HIRO. Risky; only after
   v0 HL is solid.

## 10. References

- van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation
  Learning* (VQ-VAE), NeurIPS 2017.
- Razavi et al. *Generating Diverse High-Fidelity Images with VQ-VAE-2*,
  NeurIPS 2019.
- Mentzer et al. *Finite Scalar Quantization: VQ-VAE Made Simple*, ICLR 2024.
- Jang, Gu, Poole. *Categorical Reparameterization with Gumbel-Softmax*,
  ICLR 2017.
- Maddison, Mnih, Teh. *The Concrete Distribution: A Continuous
  Relaxation of Discrete Random Variables*, ICLR 2017.
- Zeghidour et al. *SoundStream: An End-to-End Neural Audio Codec* (RVQ),
  TASLP 2022.
- Peng et al. *ASE: Adversarial Skill Embeddings for Physically Simulated
  Characters*, SIGGRAPH 2022 — closest skill-codebook precedent for our
  domain (continuous latents though).
- Quest / TAP / PRISE / H-GAP — recent BC + token-VQ skill papers worth
  comparing HL planner choices against.
