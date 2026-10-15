#!/bin/sh
# Vercel build: copy every demo page into dist/ (vercel.json's buildCommand is capped at 256 characters).
set -e
rm -rf dist
mkdir -p dist
cp index.html dist/index.html
for d in latent-space tic-tac-toe rock-paper-scissors connect-four playground autoencoder \
         handwired handwired/tic-tac-toe handwired/connect-four handwired/interpreter handwired/tower handwired/unfrozen; do
  mkdir -p "dist/$d"
  cp "$d/index.html" "dist/$d/index.html"
done
