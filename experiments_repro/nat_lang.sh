#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH/.." || exit

neurs () {
    for ((i = 0; i < $1; i++)); do
        echo -n "$2 "
    done
}

guild run dmu:penn_treebank recurrent-cells='100' learning-rate=0.02 learning-rate-rnn=0.01 scheduler-gamma=0.9  weight-decay=0.0001 -y --force-flags
guild run dmu:penn_treebank recurrent-cells='122 122' learning-rate=0.02 learning-rate-rnn=0.005 scheduler-gamma=0.9 weight-decay=0.0001 -y --force-flags
guild run dmu:penn_treebank recurrent-cells="$(neurs 5 131)" learning-rate=0.01 learning-rate-rnn=0.001 scheduler-gamma=0.98 weight-decay=0.0001 -y --force-flags
guild run dmu:penn_treebank recurrent-cells="$(neurs 10 136)" learning-rate=0.02 learning-rate-rnn=0.001 scheduler-gamma=0.98 weight-decay=0.0001 -y --force-flags

for DEPTH in 1 2 5 10; do
    guild run rhn:penn_treebank recurrent-cells="100 $DEPTH" learning-rate=0.02 weight-decay=1e-4 scheduler-gamma=0.9 checkpoint-monitor="loss/val" --force-flags -y;
done
