# 🚀 GitHub Setup Guide

Quick guide to upload this project to GitHub and enable Colab notebooks.

## 📋 Prerequisites

- GitHub account ([Sign up here](https://github.com/join))
- Git installed on your computer ([Download here](https://git-scm.com/downloads))

---

## 🔧 Step-by-Step Setup

### 1️⃣ Initialize Local Git Repository

Open terminal in the project folder and run:

```bash
cd oxford_pets_classification

# Initialize git
git init

# Check .gitignore is present
cat .gitignore

# Add all files
git add .

# First commit
git commit -m "Initial commit: Oxford-IIIT Pet Classification project"
```

### 2️⃣ Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click the **"+"** icon → **"New repository"**
3. Repository name: `oxford-pets-classification`
4. Description: `Deep learning project for cat/dog classification with custom CNNs and transfer learning`
5. **Public** or **Private** (choose based on preference)
6. **Do NOT** initialize with README, .gitignore, or license (we already have them)
7. Click **"Create repository"**

### 3️⃣ Connect Local to GitHub

Copy the commands shown on GitHub (they look like this):

```bash
git remote add origin https://github.com/danort92/oxford-pets-classification.git
git branch -M main
git push -u origin main
```

**Important:** Replace `danort92` with your actual GitHub username!

### 4️⃣ Update Notebook Links

After pushing to GitHub, update the Colab badges in the notebooks:

**Files to update:**
- `notebooks/00_quick_demo.ipynb`
- `notebooks/01_multiclass_classification.ipynb`  
- `notebooks/02_transfer_learning.ipynb`
- `README.md`

**Find and replace** `danort92` with your GitHub username.

**Example:**
```markdown
# Before
[![Open In Colab](badge.svg)](https://colab.research.google.com/.../danort92/...)

# After (if your username is "johndoe")
[![Open In Colab](badge.svg)](https://colab.research.google.com/.../johndoe/...)
```

Then commit and push the changes:

```bash
git add .
git commit -m "Update Colab links with correct username"
git push
```

### 5️⃣ Test Colab Notebooks

1. Go to your GitHub repository
2. Navigate to `notebooks/00_quick_demo.ipynb`
3. GitHub will show a preview
4. Click the **"Open in Colab"** badge
5. Colab should open with your notebook!

---

## 🎨 Optional: Add Project Banner

Create a nice banner image for your README:

1. Use [Canva](https://www.canva.com) or similar tool
2. Create image with project title and sample results
3. Save as `assets/banner.png` in your repo
4. Add to README:

```markdown
![Project Banner](assets/banner.png)
```

---

## 📊 Optional: Add Results Screenshots

After training, add result images:

```bash
# Create assets folder
mkdir -p assets/results

# Copy your best results
cp outputs/binary_classification_v3_curves.png assets/results/
cp outputs/transfer_learning_resnet50_gradcam.png assets/results/

# Add to git
git add assets/
git commit -m "Add training results screenshots"
git push
```

Update README with:

```markdown
## 📈 Results

### Training Curves
![Training Curves](assets/results/binary_classification_v3_curves.png)

### Grad-CAM Visualization
![Grad-CAM](assets/results/transfer_learning_resnet50_gradcam.png)
```

---

## 🏷️ Optional: Add GitHub Topics

Make your repo more discoverable:

1. Go to your GitHub repository
2. Click the ⚙️ icon next to "About"
3. Add topics:
   - `deep-learning`
   - `pytorch`
   - `computer-vision`
   - `image-classification`
   - `transfer-learning`
   - `cnn`
   - `machine-learning`
   - `jupyter-notebook`
   - `google-colab`

---

## 📝 Git Workflow for Future Updates

```bash
# Make changes to your code

# Check what changed
git status
git diff

# Add changes
git add .

# Commit with descriptive message
git commit -m "Add feature X" 

# Push to GitHub
git push

# The Colab notebooks will automatically use the latest version!
```

---

## 🔄 Keeping Colab Notebooks Synced

Every time you push to GitHub, the Colab notebooks automatically use the latest code because they clone from GitHub!

**Workflow:**
1. Make changes locally
2. Test locally
3. Commit and push to GitHub
4. Colab users get the latest version automatically

---

## ✅ Verification Checklist

- [ ] Repository created on GitHub
- [ ] Local repo connected to GitHub (`git remote -v` shows origin)
- [ ] All files pushed (`git push` successful)
- [ ] `.gitignore` working (no `data/oxford-iiit-pet/` or `checkpoints/` in repo)
- [ ] Colab badges updated with correct username
- [ ] Colab notebooks open correctly
- [ ] README looks good on GitHub
- [ ] (Optional) Topics added
- [ ] (Optional) Results screenshots added

---

## 🆘 Troubleshooting

### "Permission denied (publickey)"

Set up SSH keys or use HTTPS with personal access token:

```bash
# Use HTTPS instead
git remote set-url origin https://github.com/danort92/oxford-pets-classification.git
```

### "Repository too large"

Check that `.gitignore` is working:

```bash
git status

# Should NOT show:
# - data/oxford-iiit-pet/
# - checkpoints/
# - Large .pth files
```

If you accidentally committed large files:

```bash
# Remove from git but keep locally
git rm --cached -r data/oxford-iiit-pet/
git rm --cached -r checkpoints/

git commit -m "Remove large files from git"
git push
```

### Colab notebook doesn't load

1. Check repository is **Public** (or use authentication for private)
2. Verify GitHub username in URLs is correct
3. Check file path: `notebooks/00_quick_demo.ipynb`
4. Try direct Colab URL:
   ```
   https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/FILE.ipynb
   ```

---

## 🎉 You're Done!

Your project is now:
- ✅ Version controlled with Git
- ✅ Backed up on GitHub
- ✅ Accessible via Colab
- ✅ Ready for your portfolio

**Share your repo!**
- Add to LinkedIn profile
- Include in CV/Resume
- Share with professors/recruiters

---

## 📚 Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

---

**Need help?** Feel free to reach out or check GitHub's documentation!
