import torch
import torch.nn as nn

from fcn import FCN
from fcn_skip import FCNSkip

NUM_CLASSES = 21

def build(model_csl, seed = 0):
    torch.manual_seed(seed)
    return model_csl(num_classes=NUM_CLASSES)

def same_weights(a, b):
    return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))

def main():
    fcn = build(FCN).eval()
    skip = build(FCNSkip).eval()

    print("\n1. shared layers\n")
    for name in ["conv1", "conv2", "conv3", "classifier"]:
        print(f"{name:<11} identical weights: {same_weights(getattr(fcn, name), getattr(skip, name))}")

    captured = {}
    skip.score_pool2.register_forward_hook(lambda m, inp, out: captured.update(out=out))
    skip.score_pool2.register_full_backward_hook(
        lambda m, grad_in, grad_out: captured.update(grad_in=grad_in[0], grad_out=grad_out[0]))

    torch.manual_seed(1)
    x = torch.randn(2, 3, 256, 256)

    print("\n2. outputs before training\n")
    with torch.no_grad():
        out_fcn = fcn(x)
        out_skip = skip(x)
    diff = (out_fcn - out_skip).abs()
    agree = (out_fcn.argmax(dim=1) == out_skip.argmax(dim=1)).float().mean().item()
    print(f"skip branch output, max |value|: {captured['out'].abs().max().item():.3g}")
    print(f"logits std (FCN):                {out_fcn.std().item():.4f}")
    print(f"max |FCN - FCNSkip|:             {diff.max().item():.4f}")
    print(f"mean |FCN - FCNSkip|:            {diff.mean().item():.4f}")
    print(f"pixels with the same argmax:     {agree:.1%}")

    print("\n3. can the zero-initialized branch learn?")
    skip.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(skip.parameters(), lr=0.1)
    target = torch.randint(0, NUM_CLASSES, (2, 256, 256))

    for step in (1, 2):
        optimizer.zero_grad()
        loss = loss_fn(skip(x), target)
        loss.backward()
        w = skip.score_pool2.weight
        print(f"\nstep {step}:")
        print(f"|W| max before update:           {w.abs().max().item():.3g}")
        print(f"grad wrt W, norm:                {w.grad.norm().item():.3g}")
        print(f"grad wrt b, norm:                {skip.score_pool2.bias.grad.norm().item():.3g}")
        print(f"grad coming from the loss, norm: {captured['grad_out'].norm().item():.3g}")
        print(f"grad passed back to p2, norm:    {captured['grad_in'].norm().item():.3g}")
        optimizer.step()


if __name__ == "__main__":
    main()