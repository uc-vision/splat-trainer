MACHINES=()
for i in {19090..19119}; do
    MACHINES+=("cs${i}bs")
done

LOG_DIR="$HOME/splat-trainer/log"
SUCCESS_COUNT=0 
CONNECTED_MACHINES=()   


for MACHINE in "${MACHINES[@]}"; do
    
    printf "\nConnecting to %s..." "$MACHINE"
    ssh -o BatchMode=yes -o ConnectTimeout=2 $MACHINE 'exit' >/dev/null 2>&1

    if [ $? -ne 0 ]; then
        SSH_ERROR=$(ssh -o BatchMode=no -o ConnectTimeout=2 $MACHINE 'exit' 2>&1)
        if [ $? -eq 0 ]; then
            printf "Copying SSH key to %s..." "$MACHINE"
            ssh-copy-id $MACHINE > /dev/null 2>&1
            printf "\nConnected to $MACHINE "
        else
            printf "\n%s\n" "$SSH_ERROR"
            echo "Failed to connect to $MACHINE."
            continue
        fi
    fi

    echo "successfully."
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    CONNECTED_MACHINES+=("$MACHINE")

    ssh -T $MACHINE << EOF
        trap 'echo "Shutting down worker on $MACHINE"; pkill -f "rq:worker"; wait; exit' SIGINT SIGTERM
        pkill -f "rq:worker" > /dev/null 2>&1 && echo "Killed existing RQ workers on $MACHINE." || echo "No existing RQ workers to shut down on $MACHINE."
        source ~/.bashrc
        conda activate splat-trainer-py10
        cd ~/splat-trainer
        mkdir -p $LOG_DIR
        nohup rq worker --url redis://cs24004kw:6379 --burst > $LOG_DIR/rq_worker_${MACHINE}.log 2>&1 &
        exit
EOF

done

echo
echo "Total number of machines tried to connect: ${#MACHINES[@]}"
echo "Number of successful connections: $SUCCESS_COUNT"

