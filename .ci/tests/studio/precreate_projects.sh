set -e
set -x  # Enable command echoing

fedn studio login -u $STUDIO_USER -P $STUDIO_PASSWORD -H $STUDIO_HOST -p http

# Count the number of lines in the output of the fedn project list command that start with "precreated-"
project_count=$(fedn project list -H $STUDIO_HOST -p http | grep '^precreated-' | wc -l)
echo "Number of precreated projects: $project_count"

# If project_count is less than 5, create projects until the total number of projects reaches 5
if [ "$project_count" -lt 5 ]; then
    projects_to_create=$((5 - project_count))
    echo "Creating $projects_to_create projects..."
    for i in $(seq 1 $projects_to_create); do
        fedn project create -n precreated -H $STUDIO_HOST -p http --no-interactive
    done
fi
