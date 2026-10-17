#!/usr/bin/env bash
# ==============================================================================
# Agri_Sorter - Raspberry Pi 5 Environment & Hardware Setup Script
# ==============================================================================
set -e

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}${BOLD}"
    echo "================================================================================"
    echo "            Agri_Sorter - Raspberry Pi 5 Automated Setup Script                 "
    echo "================================================================================"
    echo -e "${NC}"
}

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

print_header
log_info "Working directory: ${REPO_ROOT}"

# Identify actual non-root user if executed via sudo
TARGET_USER="${SUDO_USER:-$USER}"
log_info "Configuring environment for user: ${BOLD}${TARGET_USER}${NC}"

# ------------------------------------------------------------------------------
# 1. Architecture & Hardware Detection
# ------------------------------------------------------------------------------
log_info "Checking system architecture and hardware platform..."
ARCH="$(uname -m)"
DEVICE_MODEL="Unknown"

if [ -f /proc/device-tree/model ]; then
    DEVICE_MODEL="$(tr -d '\0' < /proc/device-tree/model)"
fi

echo -e "  Architecture: ${BOLD}${ARCH}${NC}"
echo -e "  Device Model: ${BOLD}${DEVICE_MODEL}${NC}"

IS_RPI5=false
if [[ "$ARCH" == "aarch64" ]] && [[ "$DEVICE_MODEL" =~ "Raspberry Pi 5" ]]; then
    IS_RPI5=true
    log_success "Raspberry Pi 5 (aarch64) verified."
else
    log_warn "Target platform mismatch!"
    log_warn "Detected Architecture: ${ARCH}, Model: ${DEVICE_MODEL}"
    log_warn "This script is tailored for Raspberry Pi 5 (aarch64 / Bookworm)."
    log_warn "Setup will proceed, but some hardware drivers (picamera2, gpiod RP1) may not work natively on this machine."
fi

# ------------------------------------------------------------------------------
# 2. System Packages Installation via APT
# ------------------------------------------------------------------------------
log_info "Updating package lists and installing required system packages via APT..."
APT_PACKAGES=(
    python3-venv
    python3-pip
    python3-picamera2
    python3-libcamera
    python3-gpiod
    libgpiod-dev
    python3-lgpio
    i2c-tools
    python3-smbus
    build-essential
    python3-dev
    libgl1-mesa-glx
    libglib2.0-0
)

log_info "Running: sudo apt update && sudo apt install -y ${APT_PACKAGES[*]}"
sudo apt update && sudo apt install -y "${APT_PACKAGES[@]}"
log_success "System packages installed successfully."

# ------------------------------------------------------------------------------
# 3. I2C Configuration Check (/boot/firmware/config.txt)
# ------------------------------------------------------------------------------
log_info "Checking Raspberry Pi I2C configuration..."

CONFIG_TXT="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_TXT" ] && [ -f "/boot/config.txt" ]; then
    CONFIG_TXT="/boot/config.txt"
fi

I2C_ON_OK=false
I2C_BAUD_OK=false

if [ -f "$CONFIG_TXT" ]; then
    if grep -q -E '^\s*dtparam=i2c_arm=on' "$CONFIG_TXT"; then
        I2C_ON_OK=true
    fi
    if grep -q -E '^\s*dtparam=i2c_arm_baudrate=400000' "$CONFIG_TXT"; then
        I2C_BAUD_OK=true
    fi

    if $I2C_ON_OK && $I2C_BAUD_OK; then
        log_success "Found optimal I2C settings in ${CONFIG_TXT} (i2c_arm=on, 400kHz baudrate)."
    else
        log_warn "I2C configuration in ${CONFIG_TXT} needs attention:"
        if ! $I2C_ON_OK; then
            log_warn "  [-] Missing: dtparam=i2c_arm=on"
        fi
        if ! $I2C_BAUD_OK; then
            log_warn "  [-] Missing: dtparam=i2c_arm_baudrate=400000"
        fi
        echo -e "${YELLOW}Please add the following lines to ${CONFIG_TXT}:${NC}"
        echo -e "--------------------------------------------------------"
        echo -e "dtparam=i2c_arm=on"
        echo -e "dtparam=i2c_arm_baudrate=400000"
        echo -e "--------------------------------------------------------"
        echo -e "You can apply this automatically by running:"
        echo -e "  sudo tee -a ${CONFIG_TXT} << 'EOF'"
        echo -e "dtparam=i2c_arm=on"
        echo -e "dtparam=i2c_arm_baudrate=400000"
        echo -e "EOF"
        echo -e "${YELLOW}(A system reboot will be required after modifying ${CONFIG_TXT})${NC}"
    fi
else
    log_warn "${CONFIG_TXT} not found. Skipping Raspberry Pi boot config check (normal on non-Pi / container environments)."
fi

# ------------------------------------------------------------------------------
# 4. Hardware Group Permissions
# ------------------------------------------------------------------------------
log_info "Configuring user permissions for hardware access (gpio, i2c, video)..."
for grp in gpio i2c video; do
    if getent group "$grp" > /dev/null 2>&1; then
        sudo usermod -aG "$grp" "$TARGET_USER"
        log_success "Added user '${TARGET_USER}' to group '${grp}'."
    else
        log_warn "Group '${grp}' does not exist on this system; skipping."
    fi
done

# ------------------------------------------------------------------------------
# 5. Virtual Environment Setup with System Site Packages
# ------------------------------------------------------------------------------
VENV_PATH="${REPO_ROOT}/.venv"
log_info "Setting up Python virtual environment at ${VENV_PATH}..."

if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
    # Run venv creation as target user to keep correct file ownership
    sudo -u "$TARGET_USER" python3 -m venv --system-site-packages "$VENV_PATH"
else
    python3 -m venv --system-site-packages "$VENV_PATH"
fi
log_success "Virtual environment initialized with --system-site-packages."

# ------------------------------------------------------------------------------
# 6. Install Python Dependencies
# ------------------------------------------------------------------------------
REQUIREMENTS_FILE="${REPO_ROOT}/requirements-pi.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    log_info "Installing Python dependencies from ${REQUIREMENTS_FILE} into .venv..."
    if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
        sudo -u "$TARGET_USER" "$VENV_PATH/bin/pip" install --upgrade pip setuptools wheel
        sudo -u "$TARGET_USER" "$VENV_PATH/bin/pip" install -r "$REQUIREMENTS_FILE"
    else
        "$VENV_PATH/bin/pip" install --upgrade pip setuptools wheel
        "$VENV_PATH/bin/pip" install -r "$REQUIREMENTS_FILE"
    fi
    log_success "Python dependencies installed successfully."
else
    log_error "Requirements file not found at ${REQUIREMENTS_FILE}!"
    exit 1
fi

# ------------------------------------------------------------------------------
# 7. Model Assets Verification
# ------------------------------------------------------------------------------
log_info "Checking pre-trained model files..."
DETECTOR_MODEL="${REPO_ROOT}/models/detector/yolo11n.pt"
CLASSIFIER_MODEL="${REPO_ROOT}/models/external/best.pt"

MODELS_OK=true
if [ -f "$DETECTOR_MODEL" ]; then
    log_success "Found detector model: ${DETECTOR_MODEL} ($(du -h "$DETECTOR_MODEL" | cut -f1))"
else
    log_error "Missing detector model: ${DETECTOR_MODEL}"
    MODELS_OK=false
fi

if [ -f "$CLASSIFIER_MODEL" ]; then
    log_success "Found classifier model: ${CLASSIFIER_MODEL} ($(du -h "$CLASSIFIER_MODEL" | cut -f1))"
else
    log_error "Missing classifier model: ${CLASSIFIER_MODEL}"
    MODELS_OK=false
fi

if ! $MODELS_OK; then
    log_warn "One or more model weights are missing. Please verify models/ directory before production."
fi

# ------------------------------------------------------------------------------
# 8. Success Banner and Next Steps
# ------------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}================================================================================${NC}"
echo -e "${GREEN}${BOLD}                 Agri_Sorter RPi5 Setup Completed Successfully!                 ${NC}"
echo -e "${GREEN}${BOLD}================================================================================${NC}"
echo -e "Next steps to verify and start Agri_Sorter:"
echo ""
echo -e "  1. Activate virtual environment:"
echo -e "     ${BOLD}${CYAN}source .venv/bin/activate${NC}"
echo ""
echo -e "  2. Run hardware diagnostics (on Raspberry Pi 5):"
echo -e "     ${BOLD}${CYAN}python scripts/check_rpi5_hardware.py${NC}"
echo ""
echo -e "  3. If testing on a development PC / Laptop:"
echo -e "     ${BOLD}${CYAN}python scripts/check_rpi5_hardware.py --mock${NC}"
echo ""
echo -e "  4. (Optional) Run the conveyor pipeline in test mode:"
echo -e "     ${BOLD}${CYAN}python app/conveyor_pipeline.py --mock${NC}"
echo -e "${GREEN}${BOLD}================================================================================${NC}"
