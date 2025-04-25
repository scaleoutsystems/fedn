set -e
set -x  # Enable command echoing

# Define a cleanup function to run on exit
cleanup() {
    echo "Running cleanup..."
    # Add any cleanup commands here
    # For example, killing background processes
    for i in $(seq 0 $(($FEDN_NR_CLIENTS - 1))); do
        eval "kill \$PID${i}" || true
    done
    fedn project delete -id $FEDN_PROJECT -H $STUDIO_HOST -y
    echo "Cleanup completed."
}

# Register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

# Load environment variables from .env file
if [ -f "$(dirname "$0")/.env" ]; then
    echo "Loading environment variables from $(dirname "$0")/.env"
    export $(cat "$(dirname "$0")/.env" | xargs)
    # Echo each variable
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^# && "$line" =~ = ]]; then
            varname=$(echo "$line" | cut -d '=' -f 1)
            echo "$varname=${!varname}"
        fi
    done < "$(dirname "$0")/.env"
fi

fedn studio login -u $STUDIO_USER -P $STUDIO_PASSWORD -H $STUDIO_HOST
fedn project create -n citest -H $STUDIO_HOST --no-interactive
sleep 5
FEDN_PROJECT=$(fedn project list -H $STUDIO_HOST | awk 'NR>=1 {print $1; exit}')
fedn project set-context -id $FEDN_PROJECT -H $STUDIO_HOST
pushd examples/$FEDN_EXAMPLE
fedn client get-config -n test -g $FEDN_NR_CLIENTS -H $STUDIO_HOST
fedn run build --path client --keep-venv
fedn model set-active -f seed.npz -H $STUDIO_HOST
for i in $(seq 0 $(($FEDN_NR_CLIENTS - 1))); do
    fedn client start --init test_${i}.yaml --local-package > test_${i}.log 2>&1 & eval "PID${i}=$!"
done
popd
sleep 5
# add server functions so we can import it in start_session
export PYTHONPATH="$PYTHONPATH:$(pwd)/examples/server-functions"
pytest .ci/tests/studio/tests.py -x
sleep 5
# run with server functions
export FEDN_SERVER_FUNCTIONS="1"
pytest .ci/tests/studio/tests.py -x