#!/bin/bash

git ls-files test/ src/ include/ | grep -E "\.(cpp|hpp|h)$" | xargs clang-format-9 -i
