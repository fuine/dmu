#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH/.." || exit

guild run rnn:addition learning-rate=0.01 recurrent-cells="5 5" -y
guild run lstm:addition learning-rate=0.001 recurrent-cells="2 2" -y
guild run gru:addition learning-rate=0.05 recurrent-cells="3 2" -y
guild run rhn:addition learning-rate=0.02 recurrent-cells="4 3" -y --force-flags
guild run dmu:addition learning-rate=0.02 recurrent-cells="5 5" increased-bias=3 -y --force-flags
