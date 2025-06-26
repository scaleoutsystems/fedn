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
TAG=sha-$(git rev-parse HEAD | cut -c1-7)
PROJECT_NAME=citest_$TAG
fedn project create -n $PROJECT_NAME -H $STUDIO_HOST --branch $STUDIO_BRANCH --repository ghcr.io/scaleoutsystems/fedn --image fedn:$TAG --no-interactive
for i in {1..10}; do echo "Attempt $i of 10"; if ! fedn project list -H $STUDIO_HOST | awk -F'|' -v project="$PROJECT_NAME" '$1 ~ project {gsub(/^[ \t]+|[ \t]+$/, "", $4); if ($4 == "active") {exit 1} else {exit 0}}'; then echo "Status is active, exiting"; break; else echo "Status is not active, sleeping..."; sleep 5; fi; done
FEDN_PROJECT=$(fedn project list -H $STUDIO_HOST | awk -F'|' -v project="$PROJECT_NAME" '$1 ~ project {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
fedn project set-context -id $FEDN_PROJECT -H $STUDIO_HOST
pushd examples/$FEDN_EXAMPLE
fedn client get-config -n test -g $FEDN_NR_CLIENTS -H $STUDIO_HOST
fedn run build --path client --keep-venv
for attempt in {1..10}; do
    fedn model set-active -f seed.npz -H $STUDIO_HOST && break
    echo "fedn model set-active failed (attempt $attempt), retrying in 5s..."
    sleep 10
    if [ "$attempt" -eq 10 ]; then
        echo "fedn model set-active failed after 10 attempts, exiting."
        exit 1
    fi
done
# Check that the intial model is in the model trail
for attempt in {1..10}; do
    if fedn model list -H $STUDIO_HOST | grep -q "model_id"; then
        echo "Initial model found in model trail."
        break
    else
        echo "Initial model not found in model trail (attempt $attempt), retrying in 5s..."
        sleep 10
        if [ "$attempt" -eq 10 ]; then
            echo "Initial model not found in model trail after 10 attempts, exiting."
            exit 1
        fi
    fi
done
# Check that the combiner has been registered
for attempt in {1..10}; do
    if fedn combiner list -H $STUDIO_HOST | grep -q "combiner_id"; then
        echo "Combiner is registered, proceeding."
        break
    else
        echo "Combiner not registered (attempt $attempt), retrying in 5s..."
        sleep 10
        if [ "$attempt" -eq 10 ]; then
            echo "Combiner registration failed after 10 attempts, exiting."
            exit 1
        fi
    fi
done
for i in $(seq 0 $(($FEDN_NR_CLIENTS - 1))); do
    fedn client start --log-level DEBUG --init test_${i}.yaml --local-package > test_${i}.log 2>&1 & eval "PID${i}=$!"
done
popd
sleep 5
# add server functions so we can import it in start_session
export PYTHONPATH="$PYTHONPATH:$(pwd)/examples/server-functions"
pytest .ci/tests/studio/tests.py -x
#sleep 5
# run with server functions
#export FEDN_SERVER_FUNCTIONS="1"
#export SESSION_NUMBER="2"
#pytest .ci/tests/studio/tests.py -x