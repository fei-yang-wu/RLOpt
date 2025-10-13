from torchrl.modules import MLP
from torchrl.modules import ProbabilisticActor, TanhNormal
import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.data import BoundedContinuous, CompositeSpec
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.manual_seed(42)

# Ensure deterministic behavior (slower but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# (Optional) for PyTorch 2.0+, enforce deterministic algorithms
torch.use_deterministic_algorithms(True, warn_only=True)


in_keys = ["observation"]
action_spec = BoundedContinuous(
    shape=torch.Size([6]),
    low=torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], device="cuda:0"),
    high=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device="cuda:0"),
    device="cuda:0",
    dtype=torch.float32,
    domain="continuous",
)
actor_net = MLP(
    num_cells=[256, 256],
    out_features=12,
    activation_class=torch.nn.ReLU,
    device="cuda:0",
)

dist_class = TanhNormal
dist_kwargs = {
    "low": torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], device="cuda:0"),
    "high": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device="cuda:0"),
    "tanh_loc": False,
}

actor_extractor = NormalParamExtractor(
    scale_mapping=f"biased_softplus_1.0",
    scale_lb=0.1,
).to("cuda:0")
actor_net = torch.nn.Sequential(actor_net, actor_extractor)

in_keys_actor = in_keys
actor_module = TensorDictModule(
    actor_net,
    in_keys=in_keys_actor,
    out_keys=[
        "loc",
        "scale",
    ],
)
actor = ProbabilisticActor(
    spec=action_spec,
    in_keys=["loc", "scale"],
    module=actor_module,
    distribution_class=dist_class,
    distribution_kwargs=dist_kwargs,
    default_interaction_type=InteractionType.RANDOM,
    return_log_prob=False,
)

print(actor)
print("\n\n\n")

for name, param in actor.named_parameters():
    print(f"{name}: {param.data}")
