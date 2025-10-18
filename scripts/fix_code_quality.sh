#!/bin/bash
# Quick script to fix code quality issues

set -e

echo "🔧 Fixing Code Quality Issues..."
echo "================================"

# Install tools if needed
echo "📦 Installing/upgrading tools..."
pip install --upgrade black isort flake8 pylint bandit safety --quiet

echo ""
echo "1️⃣ Running Black (auto-fix formatting)..."
black src/ tests/ app.py config.py logging_config.py main.py
echo "✅ Black formatting complete"

echo ""
echo "2️⃣ Running isort (auto-fix imports)..."
isort src/ tests/ app.py config.py logging_config.py main.py
echo "✅ Import sorting complete"

echo ""
echo "3️⃣ Running flake8 (check only, no auto-fix)..."
flake8 src/ tests/ app.py config.py logging_config.py main.py || echo "⚠️  Flake8 found some issues (manual fix needed)"

echo ""
echo "4️⃣ Running pylint (check only)..."
pylint src/ --exit-zero --score=yes || echo "⚠️  Pylint found some issues (manual fix needed)"

echo ""
echo "5️⃣ Running tests..."
PYTHONPATH=. pytest -v

echo ""
echo "✅ Code quality fixes complete!"
echo "Now you can:"
echo "  - git add ."
echo "  - git commit -m 'Fix code quality issues'"
echo "  - git push"