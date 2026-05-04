#!/bin/bash

# Solver installation script
# For macOS ARM64 (Apple Silicon)

echo "🔧 Starting solver installation..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda command not found, please install Anaconda or Miniconda first"
    exit 1
fi

# Check if myenv environment exists
if ! conda env list | grep -q "myenv"; then
    echo "❌ Error: myenv environment not found, please create it first"
    echo "Create command: conda create -n myenv python=3.10 -y"
    exit 1
fi

echo "✅ Found myenv environment, starting solver installation..."

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate myenv

# Install basic solvers
echo "📦 Install GLPK..."
conda install -c conda-forge glpk -y

echo "📦 Install CBC..."
conda install -c conda-forge coin-or-cbc -y

echo "📦 Install IPOPT..."
conda install -c conda-forge ipopt -y

echo "📦 Install SCIP..."
conda install -c conda-forge scip -y

# Install Python interface package
echo "📦 Install Python interface package..."
pip install cylp

# Verify installation
echo "🔍 Verify solver installation..."
python -c "
import pyomo.environ as pyo
print('\\n📊 Solver status check:')
solvers = ['glpk', 'cbc', 'ipopt']
for solver_name in solvers:
    try:
        solver = pyo.SolverFactory(solver_name)
        if solver.available():
            print(f'✅ {solver_name}: available')
        else:
            print(f'❌ {solver_name}: not available')
    except Exception as e:
        print(f'❌ {solver_name}: error - {e}')
"

echo ""
echo "🎯 Installation completed!"
echo "📋 Installed solvers:"
conda list | grep -E "(glpk|cbc|ipopt|scip|cylp)"

echo ""
echo "💡 Usage tips:"
echo "1. Activate environment: conda activate myenv"
echo "2. Check solvers: python -c \"import pyomo.environ as pyo; print(pyo.SolverFactory('glpk').available())\""
echo "3. View environment configuration: cat solver_environment.md"
echo ""
echo "🚀 Now you can run your optimization problems!"
