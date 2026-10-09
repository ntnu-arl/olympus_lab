#!/bin/bash



current_dir=$(pwd)

# Assert that the current directory is olympus_lab
if [ "$(basename "$current_dir")" != "olympus_lab" ]; then
    echo "Error: The script must be run from the 'olympus_lab' directory!"
    exit 1
fi

# Path to the template file
TEMPLATE_FILE="docker/docker-compose-template.yaml"
OUTPUT_FILE="docker/docker-compose.yaml"
USER_ENV_FILE="docker/.env.user"


# Check if the template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Template file '$TEMPLATE_FILE' not found. Exiting."
    exit 1
fi

# Get the number of GPUs available using nvidia-smi
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Check if num_gpus is 0
if [ "$num_gpus" -eq 0 ]; then
    echo "No GPUs found. Exiting."
    exit 1
fi

touch "$USER_ENV_FILE"
> "$USER_ENV_FILE"  # Clear the file if it exists
echo "#Configure custom environemt here" >> "$USER_ENV_FILE"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20" >> "$USER_ENV_FILE"
echo "HEADLESS=0">> "$USER_ENV_FILE"
echo "DEFAULT_TRAIN_TASK=Olympus-Walk #Olympus-Attitude-Control #Olympus-Jump">> "$USER_ENV_FILE"
echo "DEFAULT_PLAY_TASK=Olympus-Walk #Olympus-Attitude-Control #Olympus-Jump #Olympus-Jump-Play etc.">> "$USER_ENV_FILE"

# Copy the template file to docker-compose.yaml
cp "$TEMPLATE_FILE" "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"  # Add an empty line before services

# Start appending services to the docker-compose.yaml file
cat << EOF >> "$OUTPUT_FILE"
services:
EOF

# Loop to create services based on the number of GPUs
for i in $(seq 0 $(($num_gpus - 1))); do
    cat << EOF >> "$OUTPUT_FILE"
    olympus-lab-${USER}-gpu-$i:
        profiles: [ "${USER}-gpu$i" ]
        env_file:
          - .env.base
          - .env.user
        user: ${UID}:${UID}
        build:
          context: ../
          dockerfile: docker/Dockerfile.base
          args:
            - ISAACSIM_BASE_IMAGE_ARG=\${ISAACSIM_BASE_IMAGE}
            - ISAACSIM_VERSION_ARG=\${ISAACSIM_VERSION}
            - ISAACSIM_ROOT_PATH_ARG=\${DOCKER_ISAACSIM_ROOT_PATH}
            - ISAACLAB_PATH_ARG=\${DOCKER_ISAACLAB_PATH}
            - DOCKER_USER_HOME_ARG=\${DOCKER_USER_HOME}
            - DOCKER_USER_ID_ARG=${UID}
            - DOCKER_USER_GID_ARG=${UID}
        image: olympus-lab-${USER}-gpu$i
        container_name: "olympus-lab-${USER}-gpu$i"
        environment: *default-olympus-lab-environment
        volumes:
            - type: volume
              source: isaac-cache-kit-${USER}-gpu$i
              target: \${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache
            - type: volume
              source: isaac-cache-ov-${USER}-gpu$i
              target: \${DOCKER_USER_HOME}/.cache/ov
            - type: volume
              source: isaac-cache-pip-${USER}-gpu$i
              target: \${DOCKER_USER_HOME}/.cache/pip
            - type: volume
              source: isaac-cache-gl-${USER}-gpu$i
              target: \${DOCKER_USER_HOME}/.cache/nvidia/GLCache
            - type: volume
              source: isaac-cache-compute-${USER}-gpu$i
              target: \${DOCKER_USER_HOME}/.nv/ComputeCache
            - type: volume
              source: isaac-logs-${USER}-gpu$i
              target: \${DOCKER_USER_HOME}/.nvidia-omniverse/logs
            - type: volume
              source: isaac-carb-logs-${USER}-gpu$i
              target: \${DOCKER_ISAACSIM_ROOT_PATH}/kit/logs/Kit/Isaac-Sim
            - type: volume
              source: isaac-data-${USER}-gpu$i
              target: \${DOCKER_USER_HOME}/.local/share/ov/data
            - type: volume
              source: isaac-docs-${USER}-gpu$i
              target: \${DOCKER_USER_HOME}/Documents
            - type: bind
              source: ../
              target: \${DOCKER_OLYMPUSLAB_PATH}
            - type: bind
              source: ~/.ssh
              target: \${DOCKER_USER_HOME}/.ssh
              read_only: true

        network_mode: \${DOCKER_NETWORK}
        deploy: 
            resources:
                reservations:
                  devices:
                    - driver: nvidia
                      device_ids: ["$i"] 
                      capabilities: [ gpu ]

        # This is the entrypoint for the container
        entrypoint: ["\${DOCKER_USER_HOME}/entrypoint.sh","$(git config user.name)","$(git config user.email)"]  

        stdin_open: true
        tty: true
EOF
done

# Append the volumes section to docker-compose.yaml using a loop
cat << EOF >> "$OUTPUT_FILE"
volumes:
  # isaac-sim
EOF
# List of base volume names
base_volumes=(
  "isaac-cache-kit"
  "isaac-cache-ov"
  "isaac-cache-pip"
  "isaac-cache-gl"
  "isaac-cache-compute"
  "isaac-logs"
  "isaac-carb-logs"
  "isaac-data"
  "isaac-docs"
  # isaac-lab
  "isaac-lab-docs"
  "isaac-lab-logs"
  "isaac-lab-data"
)
# Loop to create volumes
for i in $(seq 0 2); do  # Loop for 3 sets of volumes
    for volume in "${base_volumes[@]}"; do
        cat << EOF >> "$OUTPUT_FILE"
  ${volume}-${USER}-gpu$i:
EOF
    done
done



# Path to the template file
X11_OUTPUT_FILE="docker/x11.yaml"



# Copy the template file to docker-compose.yaml
touch "$X11_OUTPUT_FILE"
echo -n > "$X11_OUTPUT_FILE"

# Start appending services to the docker-compose.yaml file
cat << EOF >> "$X11_OUTPUT_FILE"
services:
EOF

# Loop to create services based on the number of GPUs
for i in $(seq 0 $(($num_gpus - 1))); do
    cat << EOF >> "$X11_OUTPUT_FILE"
    olympus-lab-${USER}-gpu-$i:
      environment:
      - DISPLAY
      - TERM
      - QT_X11_NO_MITSHM=1
      - XAUTHORITY=\${__ISAACLAB_TMP_XAUTH}
      user: ${UID}:${UID}
      volumes:
      - type: bind
        source: \${__ISAACLAB_TMP_DIR}
        target: \${__ISAACLAB_TMP_DIR}
      - type: bind
        source: /tmp/.X11-unix
        target: /tmp/.X11-unix
      - type: bind
        source: /etc/localtime
        target: /etc/localtime
        read_only: true
        
        
EOF
done



# Output the generated compose file
echo "docker-compose.yaml created with $num_gpus GPU containers."



