#!/bin/bash


NAME=$1
EMAIL=$2
# Create shared group if it doesn't exist
#groupadd -g $GUID $GUN 2>/dev/null || true

# Add root to shared group
#usermod -aG sharedgroup root



# Configure git safe directory
git config --global --add safe.directory /workspace/Olympus-Lab
git config --global --add safe.directory /workspace/Olympus-Lab/submodules/olympus-usd
git config --global user.name "$NAME"
git config --global user.email "$EMAIL"


${ISAACLAB_PATH}/_isaac_sim/python.sh ${OLYMPUSLAB_PATH}/.vscode/tools/setup_vscode.py




#echo "entrypoint"


bash
