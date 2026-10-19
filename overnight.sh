#!/bin/bash
# Overnight encoder comparison on the pod: ConvNeXt-Tiny, then ConvNeXt-Small if Tiny trained normally, or
# Tiny again at a lower learning rate if it didn't. Checkpoints and logs are copied to /workspace after each run
# and on exit; the pod is stopped only once the checkpoints are there.
# Run with: mkdir -p logs && nohup ./overnight.sh > logs/overnight.log 2>&1 &

DEST=/workspace/overnight_$(date +%Y%m%d)
mkdir -p "$DEST" logs

save() {
  cp -v data/unet_convnext_*.pt logs/*.log "$DEST"/ 2>/dev/null
  ls -la "$DEST"
}

finish() {
  save
  if ls "$DEST"/unet_convnext_*.pt > /dev/null 2>&1; then
    echo "results saved to $DEST, stopping pod"
    runpodctl stop pod "$RUNPOD_POD_ID"
  else
    echo "no checkpoints in $DEST, leaving the pod running"
  fi
}
trap finish EXIT

COMMON="--model unet --size 384 --class-weight-power 0.25 --lighting --batch 32 --steps 50000 --eval-every 5000
  --val-batches 50 --channels-last --compile --areas houten utrecht leidsche_rijn amsterdam nijmegen veere
  --test-areas amersfoort"

python3 center_patch.py $COMMON --encoder tu-convnext_tiny --out data/unet_convnext_tiny.pt 2>&1 \
  | tee logs/convnext_tiny.log
save

VAL=$(python3 -c "import torch; print(torch.load('data/unet_convnext_tiny.pt', map_location='cpu')['val_miou'])" \
  2>/dev/null || echo 0)
echo "convnext_tiny final val mIoU: $VAL"

if python3 -c "import sys; sys.exit(0 if float('$VAL') >= 0.50 else 1)"; then
  python3 center_patch.py $COMMON --encoder tu-convnext_small --out data/unet_convnext_small.pt 2>&1 \
    | tee logs/convnext_small.log
else
  python3 center_patch.py $COMMON --encoder tu-convnext_tiny --lr 5e-4 --out data/unet_convnext_tiny_lr5e-4.pt 2>&1 \
    | tee logs/convnext_tiny_lr5e-4.log
fi
