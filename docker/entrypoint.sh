#!/bin/bash


NAME=$1
EMAIL=$2
# Create shared group if it doesn't exist
#groupadd -g $GUID $GUN 2>/dev/null || true

# Add root to shared group
#usermod -aG sharedgroup root



# Configure git safe directory
git config --global --add safe.directory ${DOCKER_OLYMPUSLAB_PATH}
git config --global --add safe.directory ${DOCKER_OLYMPUSLAB_PATH}/submodules/olympus_usd
git config --global user.name "$NAME"
git config --global user.email "$EMAIL"


${ISAACLAB_PATH}/_isaac_sim/python.sh ${DOCKER_OLYMPUSLAB_PATH}/.vscode/tools/setup_vscode.py




#echo "entrypoint"


bash
