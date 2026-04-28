#!/bin/bash

echo "🚀 Sync code + data..."

# ajouter code + data
git add .

# éviter commit vide
git diff --cached --quiet && echo "⚠️ Nothing to commit" && exit 0

git commit -m "sync update"

git push

echo "✅ GitHub updated"