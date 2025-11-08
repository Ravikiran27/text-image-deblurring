@echo off
REM Setup script for Git and Streamlit Cloud deployment (Windows)

echo 🚀 Setting up Text Image Deblurring for deployment...
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed. Please install Git first.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo ✅ Git found
echo.

REM Initialize Git repository
echo 📦 Initializing Git repository...
git init

REM Install Git LFS
echo.
echo 📦 Setting up Git LFS for large model files...
git lfs install
git lfs track "*.h5"
git lfs track "*.keras"

REM Add all files
echo.
echo 📦 Adding files to Git...
git add .

REM First commit
echo.
echo 📝 Creating initial commit...
git commit -m "Initial commit: Text Image Deblurring with U-Net"

echo.
echo ✅ Git repository initialized successfully!
echo.
echo 📋 Next steps:
echo 1. Create a new repository on GitHub
echo 2. Run: git remote add origin https://github.com/YOUR_USERNAME/text-image-deblurring.git
echo 3. Run: git branch -M main
echo 4. Run: git push -u origin main
echo 5. Deploy on Streamlit Cloud: https://share.streamlit.io/
echo.
echo 📖 For detailed instructions, see DEPLOYMENT.md
echo.
pause
