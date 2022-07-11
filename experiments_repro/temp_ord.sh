#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH/.." || exit

guild run rnn:order learning-rate=0.01 recurrent-cells="6 6" -y
guild run lstm:order learning-rate=0.005 recurrent-cells="2 3" -y
guild run gru:order learning-rate=0.02 recurrent-cells="2 4" -y
guild run rhn:order learning-rate=0.02 recurrent-cells="4 3" -y --force-flags
guild run dmu:order learning-rate=0.05 recurrent-cells="5 6" increased-bias=3 -y --force-flags
