#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH/.." || exit

neurs () {
    for ((i = 0; i < $1; i++)); do
        echo -n "$2 "
    done
}

guild run dmu:polyphonic recurrent-cells='100' learning-rate=0.005 learning-rate-rnn=0.0025 weight-decay=0.0001 -y --force-flags
guild run dmu:polyphonic recurrent-cells="$(neurs 2 122)" learning-rate=0.005 learning-rate-rnn=0.00125  weight-decay=0.0001 -y --force-flags
guild run dmu:polyphonic recurrent-cells="$(neurs 5 131)" learning-rate=0.005 learning-rate-rnn=0.0005 weight-decay=0.0001 -y --force-flags
guild run dmu:polyphonic recurrent-cells="$(neurs 10 136)" learning-rate=0.005 learning-rate-rnn=0.00025 weight-decay=0.0001 -y --force-flags

for DEPTH in 1 2 5 10; do
    guild run rhn:polyphonic recurrent-cells="100 $DEPTH" learning-rate=0.005 weight-decay=1e-3 scheduler-gamma=1.0 checkpoint-monitor="loss/val" --force-flags -y;
done
