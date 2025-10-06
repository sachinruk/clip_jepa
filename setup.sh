#!/usr/bin/env bash
set -euo pipefail

# Display help message
show_help() {
    cat << EOF
Usage: ./setup.sh [OPTIONS]

Setup script for creating and managing the Python virtual environment.

OPTIONS:
    -h, --help      Display this help message and exit
    -u, --update    Update the virtual environment by reinstalling packages
                    from requirements.txt. If no virtual environment exists,
                    it will be created.

EXAMPLES:
    ./setup.sh              # Create virtual environment if it doesn't exist
    ./setup.sh --update     # Update/reinstall packages in virtual environment
    ./setup.sh --help       # Show this help message

EOF
}

# Parse command line arguments
UPDATE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--update)
            UPDATE_MODE=true
            shift
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Setting up virtual environment..."

if [ -d ".venv" ]; then
    if [ "$UPDATE_MODE" = true ]; then
        echo "Updating virtual environment..."
        source .venv/bin/activate
        echo "Reinstalling packages from requirements.txt..."
        pip install -r requirements.txt --upgrade
        echo "Virtual environment updated successfully!"
    else
        echo "Virtual environment already exists"
        echo "Use --update flag to reinstall/update packages"
    fi
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
    echo "Virtual environment created successfully!"
    if [[ "$(uname -s)" == "Linux" ]]; then
        pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
    fi
fi
