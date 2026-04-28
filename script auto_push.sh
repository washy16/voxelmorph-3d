#!/bin/bash

echo "🚀 STAGING FILES..."
git add .

echo "📝 COMMIT..."
git commit -m "auto update $(date)"

echo "📤 PUSH..."
git push origin main

echo "✅ DONE"