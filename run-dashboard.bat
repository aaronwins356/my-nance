@echo off
REM AI Dashboard Launcher

echo Starting AI Dashboard services...

REM Launch backend server using npm script (nodemon)
start "AI Backend" cmd /k "npm run dev"

REM Launch frontend server using Vite dev server with auto-open
start "AI Frontend" cmd /k "npx vite --host --open"

echo All services have been started.
