#!/usr/bin/env bash
set -euo pipefail

adb_bin=/opt/android/platform-tools/adb
emulator_bin=/opt/android/emulator/emulator

"$adb_bin" start-server >/dev/null
if ! "$adb_bin" devices | rg -q '^emulator-5554[[:space:]]+device'; then
  ANDROID_SDK_ROOT=/opt/android ANDROID_HOME=/opt/android \
    nohup "$emulator_bin" @Pixel_6_API_33 -no-window -no-snapshot -noaudio \
      -no-boot-anim -memory 4096 -accel off -gpu swiftshader_indirect \
      -grpc 8554 -no-metrics >/tmp/android-emulator.log 2>&1 &
  "$adb_bin" -e wait-for-device
fi

if ! "$adb_bin" -e shell ps -A 2>/dev/null | rg -q 'commands.monkey'; then
  nohup "$adb_bin" -e shell monkey --port 1080 --ignore-crashes \
    --ignore-timeouts 1 >/tmp/android-monkey.log 2>&1 &
fi

for attempt in $(seq 1 180); do
  boot_value=$("$adb_bin" -e shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')
  [[ "$boot_value" == 1 ]] && break
  [[ "$attempt" == 180 ]] && { echo 'Android boot timed out' >&2; exit 1; }
  sleep 5
done

"$adb_bin" -e shell settings put global device_provisioned 1
"$adb_bin" -e shell settings put secure user_setup_complete 1
"$adb_bin" -e shell pm disable-user --user 0 com.google.android.setupwizard >/dev/null || true
"$adb_bin" -e shell settings put global window_animation_scale 0.0
"$adb_bin" -e shell settings put global transition_animation_scale 0.0
"$adb_bin" -e shell settings put global animator_duration_scale 0.0

cd /root/android_world
export ANDROID_HOME=/opt/android
export ANDROID_SDK_ROOT=/opt/android
export ANDROID_WORLD_A11Y_METHOD=none
export ANDROID_WORLD_ADB_COMMAND_TIMEOUT=180
export ANDROID_WORLD_SETUP_APPS=0
export ANDROID_WORLD_FREEZE_DATETIME=0
exec .venv/bin/uvicorn server.android_server:app --host 127.0.0.1 --port 5000
