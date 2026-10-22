#!/bin/bash
# Road-only model on the US areas (areas.py US_AREAS) on the pod: fetch photos and labels, then train with the
# dense NL settings. Fetching is resumable, so rerunning skips what is already there. Needs ~142 GB in data/.
# The checkpoint and log are copied to /workspace at the end; the pod is left running.
# Run with: mkdir -p logs && nohup ./us_road.sh > logs/us_road.log 2>&1 &

set -o pipefail
DEST=/workspace/us_road_$(date +%Y%m%d)
mkdir -p "$DEST" logs data

TRAIN="denver_downtown denver_highlands denver_wash_park denver_park_hill lakewood aurora littleton arvada
  highlands_ranch bennett dc_capitol_hill_2021 dc_capitol_hill_2025 dc_downtown_2021 dc_downtown_2025 dc_columbia_heights_2021
  dc_columbia_heights_2025 dc_northwest_2021 dc_northwest_2025 dc_anacostia_2021 dc_anacostia_2025"
TEST="golden boulder genesee dc_brookland_2021 dc_brookland_2025"

FREE_GB=$(df -BG --output=avail data | tail -1 | tr -dc 0-9)
echo "free in data/: ${FREE_GB} GB"

python3 us_fetch.py $TRAIN $TEST 2>&1 | tr '\r' '\n' | grep -v '^  [0-9]*/[0-9]* chunks' | tee logs/us_fetch.log \
  || { echo "fetch failed, see logs/us_fetch.log"; exit 1; }

python3 center_patch.py --model unet --size 512 --target 384 --class-weight-power 0.25 --lighting --ema 0.9995 \
  --batch 32 --steps 50000 --eval-every 5000 --val-batches 50 --channels-last --compile --full-eval \
  --areas $TRAIN --test-areas $TEST --out data/us_road.pt 2>&1 | tee logs/us_road_train.log

cp -v data/us_road.pt logs/us_fetch.log logs/us_road_train.log "$DEST"/
