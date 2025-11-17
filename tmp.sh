#!/usr/bin/env bash
echo "hostname: $(hostname)"
echo "user: $(whoami)"
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "os: macOS $(sw_vers -productVersion)"
  echo "model: $(sysctl -n hw.model)"
  echo "serial: $(system_profiler SPHardwareDataType | awk -F': ' '/Serial Number/{print $2}')"
  echo "encryption: $(fdesetup status 2>/dev/null)"
else
  echo "os: $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2)"
  echo "model: $(sudo dmidecode -s system-product-name 2>/dev/null || cat /sys/class/dmi/id/product_name)"
  echo "serial: $(sudo dmidecode -s system-serial-number 2>/dev/null || cat /sys/class/dmi/id/product_serial)"
  echo "encryption: $(lsblk -o NAME,MOUNTPOINT,FSTYPE | grep crypt 2>/dev/null && echo 'encrypted' || echo 'not encrypted')"
fi
