import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from core import build_model, cifar100_loaders, ensure_parent, evaluate, set_seed, train_one_epoch


def run_training(args):
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    train_loader, test_loader = cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )
    model = build_model(args.model, num_classes=100).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    best_test_acc = 0.0
    final_train_loss = 0.0
    final_test_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args.use_amp
        )
        scheduler.step()
        test_acc, test_loss = evaluate(model, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)
        final_train_loss = train_loss
        final_test_acc = test_acc

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
        }
        history.append(record)
        if epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"test_acc={test_acc:.2f} best={best_test_acc:.2f}"
            )

    result = {
        "model": args.model,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "momentum": args.momentum,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_test_acc": best_test_acc,
        "final_test_acc": final_test_acc,
        "final_train_loss": final_train_loss,
    }

    if args.output:
        ensure_parent(args.output)
        payload = {"config": vars(args), "result": result, "history": history}
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-100 model with SGD/SGDM.")
    parser.add_argument("--model", choices=["resnet18", "resnet50", "vgg16"], default="resnet18")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
