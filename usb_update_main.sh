#!/bin/sh
set -eu

USB_MAIN="${USB_MAIN:-/run/media/sda1/main.py}"
DEFAULT_DEST_MAIN="/home/Tronlong/rknn_model_zoo/examples/yolov5/python/main.py"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
BACKUP_OLD="${BACKUP_OLD:-1}"
ONCE=0

case "${1:-}" in
    -h|--help)
        cat <<EOF
Usage:
  sh usb_update_main.sh [DEST_MAIN]
  sh usb_update_main.sh --once [DEST_MAIN]

Environment:
  USB_MAIN       source file, default: /run/media/sda1/main.py
  DEST_MAIN      destination file, default: /home/Tronlong/rknn_model_zoo/examples/yolov5/python/main.py
  POLL_INTERVAL  watch interval seconds, default: 2
  BACKUP_OLD     create timestamp backup before overwrite, default: 1
EOF
        exit 0
        ;;
esac

if [ "${1:-}" = "--once" ]; then
    ONCE=1
    shift
fi

DEST_MAIN="${DEST_MAIN:-${1:-$DEFAULT_DEST_MAIN}}"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

checksum() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        cksum "$1" | awk '{print $1 ":" $2}'
    fi
}

get_python_assignment() {
    key="$1"
    file="$2"
    awk -v key="$key" '
        BEGIN { pattern = "^[[:space:]]*" key "[[:space:]]*=" }
        $0 ~ pattern {
            sub(/^[^=]*=[[:space:]]*/, "", $0)
            sub(/[[:space:]]*#.*/, "", $0)
            print
            exit
        }
    ' "$file"
}

set_python_assignment() {
    key="$1"
    value="$2"
    file="$3"
    tmp_assign="$file.assign.$$"
    awk -v key="$key" -v value="$value" '
        BEGIN { pattern = "^[[:space:]]*" key "[[:space:]]*=" }
        $0 ~ pattern {
            match($0, /^[[:space:]]*/)
            indent = substr($0, 1, RLENGTH)
            print indent key " = " value
            next
        }
        { print }
    ' "$file" > "$tmp_assign"
    mv "$tmp_assign" "$file"
}

preserve_board_config() {
    tmp_file="$1"
    old_file="$2"
    if [ ! -f "$old_file" ]; then
        log "no existing main.py found; BOARD_ID and BOARD_ROW_IDX will use USB values"
        return 0
    fi

    old_board_id=$(get_python_assignment "BOARD_ID" "$old_file" || true)
    old_row_idx=$(get_python_assignment "BOARD_ROW_IDX" "$old_file" || true)

    if [ -n "$old_board_id" ]; then
        set_python_assignment "BOARD_ID" "$old_board_id" "$tmp_file"
        log "preserved BOARD_ID = $old_board_id"
    fi

    if [ -n "$old_row_idx" ]; then
        set_python_assignment "BOARD_ROW_IDX" "$old_row_idx" "$tmp_file"
        log "preserved BOARD_ROW_IDX = $old_row_idx"
    fi
}

copy_main() {
    if [ ! -f "$USB_MAIN" ]; then
        return 1
    fi

    dest_dir=$(dirname -- "$DEST_MAIN")
    if [ ! -d "$dest_dir" ]; then
        log "destination directory does not exist: $dest_dir"
        return 1
    fi

    tmp="$DEST_MAIN.tmp.$$"
    cp "$USB_MAIN" "$tmp"
    preserve_board_config "$tmp" "$DEST_MAIN"
    src_sum=$(checksum "$tmp")
    if [ -f "$DEST_MAIN" ]; then
        dest_sum=$(checksum "$DEST_MAIN")
        if [ "$src_sum" = "$dest_sum" ]; then
            log "main.py is already up to date: $DEST_MAIN"
            rm -f "$tmp"
            return 0
        fi
        if [ "$BACKUP_OLD" = "1" ]; then
            backup="$DEST_MAIN.bak.$(date '+%Y%m%d_%H%M%S')"
            cp "$DEST_MAIN" "$backup"
            log "backup created: $backup"
        fi
    fi

    if [ -f "$DEST_MAIN" ]; then
        mode=$(stat -c '%a' "$DEST_MAIN" 2>/dev/null || printf '644')
        chmod "$mode" "$tmp" 2>/dev/null || chmod 644 "$tmp"
    else
        chmod 644 "$tmp"
    fi
    mv "$tmp" "$DEST_MAIN"
    sync
    log "updated $DEST_MAIN from $USB_MAIN"
    return 0
}

if [ "$ONCE" = "1" ]; then
    copy_main
    exit $?
fi

log "watching $USB_MAIN -> $DEST_MAIN"
last_sum=""
while :; do
    if [ -f "$USB_MAIN" ]; then
        current_sum=$(checksum "$USB_MAIN")
        if [ "$current_sum" != "$last_sum" ]; then
            if copy_main; then
                last_sum="$current_sum"
            fi
        fi
    fi
    sleep "$POLL_INTERVAL"
done
