# Installation Guide

## Fixing the `pip` Command Issue

If you see `zsh: command not found: pip`, use one of these solutions:

### Solution 1: Use `python3 -m pip` (Recommended)

```bash
python3 -m pip install -r requirements.txt
```

### Solution 2: Use `pip3`

```bash
pip3 install -r requirements.txt
```

### Solution 3: Create an alias (Add to ~/.zshrc)

```bash
echo 'alias pip="python3 -m pip"' >> ~/.zshrc
source ~/.zshrc
```

## Full Installation Steps

1. **Check Python version** (need 3.8+):
```bash
python3 --version
```

2. **Create virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
python3 -m pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python3 -c "import pandas, xgboost, mlflow; print('All packages installed!')"
```

## Troubleshooting

### Issue: `pip` command not found
**Solution**: Use `python3 -m pip` instead

### Issue: Permission denied
**Solution**: Use `--user` flag or activate virtual environment
```bash
python3 -m pip install --user -r requirements.txt
```

### Issue: Cloudscraper installation fails
**Solution**: Install system dependencies first (Linux/Mac)
```bash
# macOS
brew install python3

# Then retry
python3 -m pip install cloudscraper
```

### Issue: SSL certificate errors
**Solution**: Update certificates (macOS)
```bash
/Applications/Python\ 3.12/Install\ Certificates.command
```

## Quick Test

After installation, test the scraper:

```bash
python3 -m src.data_collection.fbref_scraper
```

This should run without errors (though it may take time to scrape data).

