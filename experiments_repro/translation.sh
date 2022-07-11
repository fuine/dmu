#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH/.." || exit

neurs () {
    for ((i = 0; i < $1; i++)); do
        echo -n "$2 "
    done
}

for LANG in "spa" "deu" "por"; do
    guild run dmu:translation decoder-cells="200" encoder-cells="200" learning-rate=0.005 learning-rate-rnn=0.0025 scheduler-gamma=0.9 weight-decay=1e-4 lang-pair="${LANG}-eng" --force-flags -y;
    guild run dmu:translation decoder-cells="340 200" encoder-cells="340 200" learning-rate=0.01 learning-rate-rnn=0.0025 scheduler-gamma=0.9 weight-decay=1e-4 lang-pair="${LANG}-eng" --force-flags -y;
    guild run dmu:translation decoder-cells="$(neurs 4 300) 200" encoder-cells="$(neurs 4 300) 200" learning-rate=0.003 learning-rate-rnn=0.0003 weight-decay=1e-4 lang-pair="${LANG}-eng" --force-flags -y;
    guild run dmu:translation decoder-cells="$(neurs 9 300) 200" encoder-cells="$(neurs 9 300) 200" learning-rate=0.003 learning-rate-rnn=0.00015 weight-decay=1e-4 lang-pair="${LANG}-eng" --force-flags -y;

    for DEPTH in 1 2 5 10; do
        guild run rhn:translation decoder-cells="200 $DEPTH" encoder-cells="$(neurs DEPTH 200)" learning-rate=0.01 weight-decay=1e-4 scheduler-gamma=0.9 lang-pair="${LANG}-eng" --force-flags -y;
    done
done
