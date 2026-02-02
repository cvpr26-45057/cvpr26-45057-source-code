
import torch
from models.reltr import build_reltr
from util.misc import NestedTensor

class Args:
    # Model config
    backbone = 'dynamic_swin_t'
    lr_backbone = 1e-5
    masks = False
    return_interm_layers = True
    dilation = False
    hidden_dim = 256
    position_embedding = 'sine'
    enc_layers = 2
    dec_layers = 2
    dim_feedforward = 1024
    dropout = 0.1
    nheads = 8
    num_queries = 100
    pre_norm = False
    
    # Dataset config (using PID defaults)
    dataset = 'pid'
    dataset_file = 'pid'
    num_classes = 13
    num_rel_classes = 3
    num_entities = 100
    num_triplets = 200
    
    # Loss config
    aux_loss = True
    bbox_loss_coef = 5.0
    giou_loss_coef = 2.0
    rel_loss_coef = 1.0
    eos_coef = 0.1
    
    # Matcher config
    set_cost_class = 1.0
    set_cost_bbox = 5.0
    set_cost_giou = 2.0

    # Dynamic config
    ratio_weight = 2.0
    distill_weight = 0.5
    sparse_ratio = [0.5, 0.4, 0.3]
    pruning_loc = [1, 1, 5]
    enable_reconstruction = True # Enable for testing
    recon_loss_coef = 1.0
    
    device = 'cpu' # Run smoke test on CPU for speed/simplicity if no GPU

print("Building RelTR with Dynamic Swin...")
args = Args()
model, criterion, postprocessors = build_reltr(args)
print(f"ratio_weight= {args.ratio_weight} distill_weight {args.distill_weight}")

# Mock Input
x = torch.randn(2, 3, 224, 224)
mask = torch.zeros((2, 224, 224), dtype=torch.bool)
samples = NestedTensor(x, mask)

# Mock Targets
targets = [
    {
        'labels': torch.tensor([1, 2], dtype=torch.long),
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]),
        'rel_annotations': torch.tensor([[0, 1, 1]]),
        'iscrowd': torch.tensor([0, 0])
    },
    {
        'labels': torch.tensor([1], dtype=torch.long),
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        'rel_annotations': torch.tensor([[0, 0, 0]]), # Dummy self loop or whatever
        'iscrowd': torch.tensor([0])
    }
]

print("Model built.")
print("Forward Pass...")
model.train()
outputs = model(samples)
print(f"Output Keys: {outputs.keys()}")

if 'decisions' in outputs:
    print(f"Decisions found: {len(outputs['decisions'])}")
else:
    print("WARNING: Decisions not found in output!")

print("Loss Calculation...")
# Pass inputs so DistillDiffPruningLoss can function (even if no teacher for now)
loss_dict = criterion(outputs, targets, inputs=samples)
print(f"Loss Dict Keys: {loss_dict.keys()}")

if 'loss_sparsity' in loss_dict:
    print(f"Loss Sparsity: {loss_dict['loss_sparsity']}")
    print(f"Sparsity Weight: {criterion.weight_dict.get('loss_sparsity', 'Not Found')}")
else:
    print("WARNING: loss_sparsity NOT found in loss dict!")

print("Test Complete.")
