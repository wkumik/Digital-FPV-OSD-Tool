# Release Notes

---

## v1.1 — 2026-02-20

### New
- BetaFPV P1 embedded OSD support
- Caddx Ascent OSD support
- Light theme with in-app custom colour editor
- UI scaling (100% / 125% / 150% / 175%)
- Version number displayed in the app header

### Changed
- App rebranded to **VueOSD**
- Cleaned up unused files (`OnyxFPV OSD Tool.bat`, `nul`, `splash.png`)

---

## v1.0.0 — Initial Release

### GitHub Release — Step by Step

#### 1. Create the repository

1. Go to https://github.com/new
2. **Repository name:** `Digital-FPV-OSD-Tool`
3. **Description:** `MSP-OSD overlay tool for FPV DVR video — Betaflight / INAV / Ardupilot`
4. Set to **Public**
5. ✅ Add a README ← uncheck this (we have our own)
6. **License:** MIT (already included)
7. Click **Create repository**

---

#### 2. Prepare your local folder

Open a terminal (cmd or PowerShell) in the `Digital-FPV-OSD-Tool` folder:

```bat
cd "C:\path\to\Digital-FPV-OSD-Tool"
git init
git add .
git commit -m "Initial release v1.0.0"
```

---

#### 3. Push to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bat
git remote add origin https://github.com/YOUR_USERNAME/Digital-FPV-OSD-Tool.git
git branch -M main
git push -u origin main
```

GitHub will ask for your username + a **Personal Access Token** (not your password).
Generate one at: https://github.com/settings/tokens/new
→ Select scope: **repo** → Generate → copy it → paste as the password.

---

#### 4. Create the v1.0.0 release

```bat
git tag v1.0.0
git push origin v1.0.0
```

Then on GitHub:
1. Click **Releases** → **Draft a new release**
2. **Tag:** `v1.0.0`
3. **Title:** `VueOSD v1.0.0 — Digital FPV OSD Tool`
4. **Description:** paste the section from README.md
5. Attach the zip file as a release asset (optional — users can also clone)
6. Click **Publish release**

---

#### 5. Recommended .gitignore

Create a file called `.gitignore` with:

```
__pycache__/
*.pyc
*.pyo
.venv/
build/
dist/
*.spec
*.zip
*.egg-info/
```

Add and commit it before pushing:
```bat
git add .gitignore
git commit -m "Add .gitignore"
git push
```

---

#### 6. Update CREDITS.md

Before publishing, update `CREDITS.md`:
- Set your actual GitHub URL in place of `https://github.com/wkumik`
- Add yourself as the author

---

#### 7. After publishing — update the README badge URLs

In `README.md`, update any `github.com/wkumik` links as needed.
