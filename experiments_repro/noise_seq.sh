#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH/.." || exit

guild run rnn:noisy learning-rate=0.01 recurrent-cells="5 5" -y
guild run lstm:noisy learning-rate=0.002 recurrent-cells="2 2" -y
guild run gru:noisy learning-rate=0.1 recurrent-cells="2 3" -y
guild run rhn:noisy learning-rate=0.05 recurrent-cells="3 3" -y --force-flags
guild run dmu:noisy learning-rate=0.02 recurrent-cells="5 4" increased-bias=3 -y --force-flags
