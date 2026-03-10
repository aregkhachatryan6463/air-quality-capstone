@echo off
cd /d "%~dp0"
echo Pushing to https://github.com/aregkhachatryan6463/air-quality-capstone
echo.
echo If prompted, sign in to GitHub (use a Personal Access Token as password if 2FA is on).
echo.
git push -u origin main
pause
