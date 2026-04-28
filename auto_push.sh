#!/bin/bash

echo "🚀 Auto push..."

git add .

git diff --cached --quiet && echo "⚠️ Nothing to commit" && exit 0

git commit -m "auto sync update"

git push

echo "✅ Done"