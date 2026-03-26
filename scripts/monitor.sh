#!/bin/bash
# superGPT RunPod Training Monitor v3
# Usage: ./scripts/monitor.sh
# Stop: Ctrl+C

IP="63.141.33.29"
PORT="22020"
KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@$IP -p $PORT -i $KEY"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

TARGET_ITERS=30000
TARGET_TOKENS=1000000000

while true; do
    clear
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║     ⚡ superGPT RunPod Training Monitor ⚡              ║${NC}"
    echo -e "${BOLD}║     Model: large (350M) | Data: 1B FineWeb-Edu         ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
    echo -e "  ${DIM}$(date '+%A, %B %d %Y — %I:%M:%S %p')${NC}"
    echo ""

    OUTPUT=$($SSH '
        echo "###MEM###"
        free -h | head -2
        echo "###LOG###"
        tail -15 /workspace/bigrun.log 2>/dev/null || echo "no log"
        echo "###GPU###"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null
        echo "###PROC###"
        ps aux | grep -E "train|data_pipeline" | grep python | grep -v grep | wc -l
        echo "###SHARDS###"
        ls -la /workspace/superGPT/data/*.bin 2>/dev/null | wc -l
        echo "###DISK###"
        du -sh /workspace/superGPT/data/ 2>/dev/null || echo "0"
    ' 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo -e "  ${RED}✗ Cannot reach RunPod (SSH timeout)${NC}"
        echo -e "  ${DIM}Check that your pod is still running at runpod.io${NC}"
        echo ""
        echo -e "  ${CYAN}Retrying in 30s... (Ctrl+C to stop)${NC}"
        sleep 30
        continue
    fi

    MEM_INFO=$(echo "$OUTPUT" | sed -n '/###MEM###/,/###LOG###/p' | grep -v "###" | tail -1)
    LOGS=$(echo "$OUTPUT" | sed -n '/###LOG###/,/###GPU###/p' | grep -v "###")
    GPU_LINE=$(echo "$OUTPUT" | sed -n '/###GPU###/,/###PROC###/p' | grep -v "###" | head -1)
    PROCS=$(echo "$OUTPUT" | sed -n '/###PROC###/,/###SHARDS###/p' | grep -v "###" | head -1)
    SHARDS=$(echo "$OUTPUT" | sed -n '/###SHARDS###/,/###DISK###/p' | grep -v "###" | head -1)
    DISK=$(echo "$OUTPUT" | sed -n '/###DISK###/,$p' | grep -v "###" | head -1)

    GPU_UTIL=$(echo "$GPU_LINE" | cut -d',' -f1 | tr -d ' %')
    GPU_MEM=$(echo "$GPU_LINE" | cut -d',' -f2 | tr -d ' MiB')
    GPU_TOTAL=$(echo "$GPU_LINE" | cut -d',' -f3 | tr -d ' MiB')
    GPU_TEMP=$(echo "$GPU_LINE" | cut -d',' -f4 | tr -d ' ')
    GPU_POWER=$(echo "$GPU_LINE" | cut -d',' -f5 | tr -d ' W')

    MEM_USED=$(echo "$MEM_INFO" | awk '{print $3}')
    MEM_TOTAL=$(echo "$MEM_INFO" | awk '{print $2}')

    # Memory bar
    echo -e "  ${DIM}RAM: ${MEM_USED}/${MEM_TOTAL} | Shards: ${SHARDS} files | Disk: ${DISK}${NC}"
    echo ""

    # Detect phase
    if echo "$LOGS" | grep -q "ALL DONE\|Training complete"; then
        PHASE="COMPLETE"
    elif echo "$LOGS" | grep -q "iter.*loss"; then
        PHASE="TRAINING"
    elif echo "$LOGS" | grep -q "Shard.*tokens\|Tokenizing\|Data Pipeline"; then
        PHASE="DATA_PREP"
    elif echo "$LOGS" | grep -q "Compiling"; then
        PHASE="COMPILING"
    elif echo "$LOGS" | grep -q "download\|Loading\|cache"; then
        PHASE="DOWNLOADING"
    else
        PHASE="STARTING"
    fi

    if [ "$PHASE" = "DOWNLOADING" ]; then
        echo -e "  ${YELLOW}${BOLD}📥 DOWNLOADING DATASET${NC}"
        echo -e "  ${DIM}Downloading FineWeb-Edu from HuggingFace...${NC}"

    elif [ "$PHASE" = "DATA_PREP" ]; then
        SHARD_LINE=$(echo "$LOGS" | grep "Shard" | tail -1)
        CURRENT_TOKENS=$(echo "$SHARD_LINE" | grep -oE '[0-9,]+ tokens' | tr -d ', tokens' | head -1)
        TOK_SPEED=$(echo "$SHARD_LINE" | grep -oE '[0-9,]+ tok/s' | tr -d ', tok/s' | head -1)

        if [ -n "$CURRENT_TOKENS" ] && [ "$CURRENT_TOKENS" -gt 0 ] 2>/dev/null; then
            PCT=$((CURRENT_TOKENS * 100 / TARGET_TOKENS))
            REMAINING=$((TARGET_TOKENS - CURRENT_TOKENS))
            if [ -n "$TOK_SPEED" ] && [ "$TOK_SPEED" -gt 0 ] 2>/dev/null; then
                ETA_SEC=$((REMAINING / TOK_SPEED))
                ETA_MIN=$((ETA_SEC / 60))
            fi
            BAR_WIDTH=40
            FILLED=$((PCT * BAR_WIDTH / 100))
            EMPTY=$((BAR_WIDTH - FILLED))
            BAR=$(printf "%${FILLED}s" | tr ' ' '█')
            BAR_E=$(printf "%${EMPTY}s" | tr ' ' '░')

            echo -e "  ${YELLOW}${BOLD}📦 PHASE 1: DATA PREPARATION (shard pattern)${NC}"
            echo ""
            echo -e "  ${BOLD}[${GREEN}${BAR}${DIM}${BAR_E}${NC}${BOLD}]${NC} ${PCT}%"
            echo ""
            echo -e "  Tokens:    ${BOLD}$(printf "%'d" $CURRENT_TOKENS)${NC} / $(printf "%'d" $TARGET_TOKENS)"
            echo -e "  Speed:     ${BOLD}$(printf "%'d" $TOK_SPEED) tok/s${NC}"
            if [ -n "$ETA_MIN" ]; then
                echo -e "  ETA:       ${BOLD}${ETA_MIN} min${NC}"
            fi
        else
            echo -e "  ${YELLOW}${BOLD}📦 PHASE 1: DATA PREPARATION${NC}"
            echo -e "  ${DIM}Tokenizing with multiprocessing...${NC}"
        fi

    elif [ "$PHASE" = "COMPILING" ]; then
        echo -e "  ${MAGENTA}${BOLD}⚙️  COMPILING MODEL (torch.compile)${NC}"
        echo ""
        echo -e "  ${DIM}This takes 1-2 min, then training runs 2x faster.${NC}"

    elif [ "$PHASE" = "TRAINING" ]; then
        ITER_LINE=$(echo "$LOGS" | grep "iter" | tail -1)
        CURRENT_ITER=$(echo "$ITER_LINE" | grep -oE 'iter +[0-9]+' | grep -oE '[0-9]+')
        CURRENT_LOSS=$(echo "$ITER_LINE" | grep -oE 'loss [0-9.]+' | grep -oE '[0-9.]+')
        TRAIN_SPEED=$(echo "$ITER_LINE" | grep -oE '[0-9]+ tok/s' | grep -oE '[0-9]+')
        VAL_LOSS=$(echo "$LOGS" | grep "val" | tail -1 | grep -oE 'val loss:? *[0-9.]+' | grep -oE '[0-9.]+')

        if [ -n "$CURRENT_ITER" ] && [ "$CURRENT_ITER" -gt 0 ] 2>/dev/null; then
            PCT=$((CURRENT_ITER * 100 / TARGET_ITERS))
            REMAINING=$((TARGET_ITERS - CURRENT_ITER))
            if [ -n "$TRAIN_SPEED" ] && [ "$TRAIN_SPEED" -gt 0 ] 2>/dev/null; then
                TOKS_PER_ITER=16384
                SECS_PER_ITER=$(echo "scale=4; $TOKS_PER_ITER / $TRAIN_SPEED" | bc 2>/dev/null || echo "0.13")
                ETA_SEC=$(echo "scale=0; $REMAINING * $SECS_PER_ITER" | bc 2>/dev/null || echo "0")
                ETA_HR=$(echo "scale=0; $ETA_SEC / 3600" | bc 2>/dev/null || echo "?")
                ETA_MIN=$(echo "scale=0; ($ETA_SEC % 3600) / 60" | bc 2>/dev/null || echo "?")
            fi
            BAR_WIDTH=40
            FILLED=$((PCT * BAR_WIDTH / 100))
            EMPTY=$((BAR_WIDTH - FILLED))
            BAR=$(printf "%${FILLED}s" | tr ' ' '█')
            BAR_E=$(printf "%${EMPTY}s" | tr ' ' '░')

            echo -e "  ${GREEN}${BOLD}🏋️  PHASE 2: TRAINING (350M params)${NC}"
            echo ""
            echo -e "  ${BOLD}[${GREEN}${BAR}${DIM}${BAR_E}${NC}${BOLD}]${NC} ${PCT}%"
            echo ""
            echo -e "  Iteration: ${BOLD}$(printf "%'d" $CURRENT_ITER)${NC} / $(printf "%'d" $TARGET_ITERS)"
            echo -e "  Loss:      ${BOLD}${CURRENT_LOSS}${NC}"
            if [ -n "$VAL_LOSS" ]; then
                echo -e "  Val Loss:  ${BOLD}${VAL_LOSS}${NC}"
            fi
            echo -e "  Speed:     ${BOLD}$(printf "%'d" $TRAIN_SPEED) tok/s${NC}"
            if [ -n "$ETA_HR" ] && [ "$ETA_HR" != "?" ]; then
                echo -e "  ETA:       ${BOLD}${ETA_HR}h ${ETA_MIN}m remaining${NC}"
            fi
            echo ""
            echo -e "  ${DIM}── GPU ──${NC}"
            echo -e "  Util:      ${BOLD}${GPU_UTIL}%${NC}  |  Memory: ${BOLD}${GPU_MEM}/${GPU_TOTAL} MiB${NC}"
            echo -e "  Temp:      ${GPU_TEMP}°C  |  Power: ${GPU_POWER}W"
        fi

    elif [ "$PHASE" = "COMPLETE" ]; then
        echo -e "  ${GREEN}${BOLD}✅ TRAINING COMPLETE!${NC}"
        echo -e "  ${BOLD}Checkpoint: checkpoints/best.pt${NC}"

    else
        echo -e "  ${YELLOW}${BOLD}⏳ STARTING UP...${NC}"
    fi

    echo ""
    echo -e "  ${DIM}── Last 5 Log Lines ──${NC}"
    echo "$LOGS" | tail -5 | while IFS= read -r line; do
        if echo "$line" | grep -q "loss"; then
            echo -e "  ${GREEN}$line${NC}"
        elif echo "$line" | grep -q "best\|Saved\|New\|Merged"; then
            echo -e "  ${CYAN}${BOLD}$line${NC}"
        elif echo "$line" | grep -q "Shard\|tokens\|samples"; then
            echo -e "  ${YELLOW}$line${NC}"
        elif echo "$line" | grep -q "Error\|error\|Traceback"; then
            echo -e "  ${RED}$line${NC}"
        elif [ -n "$line" ]; then
            echo -e "  ${DIM}$line${NC}"
        fi
    done

    echo ""
    echo -e "  ${DIM}Refreshes every 30s · Ctrl+C to stop${NC}"
    sleep 30
done
