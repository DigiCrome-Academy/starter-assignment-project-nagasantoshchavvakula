[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/13zBya6Q)
﻿# GitHub Classroom Assignment Guide

Welcome to your first GitHub Classroom assignment! This guide will walk you through everything you need to know about using GitHub for your coursework.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Accepting the Assignment](#step-1-accepting-the-assignment)
- [Step 2: Cloning the Repository](#step-2-cloning-the-repository)
- [Step 3: Understanding the Repository Structure](#step-3-understanding-the-repository-structure)
- [Step 4: Creating a Branch](#step-4-creating-a-branch)
- [Step 5: Working on Your Assignment](#step-5-working-on-your-assignment)
- [Step 6: Running Tests Locally](#step-6-running-tests-locally)
- [Step 7: Committing Your Changes](#step-7-committing-your-changes)
- [Step 8: Pushing Your Code](#step-8-pushing-your-code)
- [Step 9: Understanding CI/CD](#step-9-understanding-cicd)
- [Step 10: Creating a Pull Request](#step-10-creating-a-pull-request)
- [Step 11: Merging Your Branch](#step-11-merging-your-branch)
- [Step 12: Submitting Your Assignment](#step-12-submitting-your-assignment)
- [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
- [Best Practices](#best-practices)
- [Additional Resources](#additional-resources)

---

## Prerequisites

Before you begin, make sure you have:

- [ ] A GitHub account (create one at [github.com](https://github.com))
- [ ] Git installed on your computer
  - **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
  - **Mac**: Install via Terminal: `brew install git` or download from [git-scm.com](https://git-scm.com/download/mac)
  - **Linux**: `sudo apt-get install git` (Ubuntu/Debian) or `sudo yum install git` (Fedora)
- [ ] Python installed (version 3.8 or higher recommended)
- [ ] A code editor (VS Code, PyCharm, or similar)
- [ ] Basic command line knowledge

### Verify Your Installation

Open your terminal/command prompt and run:

```bash
git --version
python --version  # or python3 --version
```

You should see version numbers displayed for both commands.

---

## Step 1: Accepting the Assignment

### 1.1 Click the Assignment Link

Your instructor will provide you with a unique GitHub Classroom assignment link. It will look something like:

```
https://classroom.github.com/a/xxxxxxxxx
```

### 1.2 Authorize GitHub Classroom

- Click the assignment link
- If this is your first time, you'll be asked to authorize GitHub Classroom
- Click **"Authorize GitHub Classroom"**

### 1.3 Accept the Assignment

- You'll see a page showing the assignment details
- Click the **"Accept this assignment"** button
- Wait a few seconds while GitHub creates your personal repository

### 1.4 Access Your Repository

- Once created, you'll see a link to your assignment repository
- The URL will look like: `https://github.com/your-org/assignment-name-your-username`
- Click this link to view your repository
- **Bookmark this page** - you'll need to return to it often!

---

## Step 2: Cloning the Repository

Cloning creates a local copy of your repository on your computer.

### 2.1 Copy the Repository URL

On your GitHub repository page:
1. Click the green **"Code"** button
2. Make sure **"HTTPS"** is selected
3. Click the clipboard icon to copy the URL

### 2.2 Open Your Terminal

- **Windows**: Open Git Bash or Command Prompt
- **Mac/Linux**: Open Terminal

### 2.3 Navigate to Your Desired Directory

```bash
# Navigate to where you want to store your assignments
cd ~/Documents/Coursework  # Example path, adjust as needed

# Create a directory for this course if you haven't already
mkdir CS101
cd CS101
```

### 2.4 Clone the Repository

```bash
git clone https://github.com/your-org/assignment-name-your-username.git
```

### 2.5 Enter the Repository Directory

```bash
cd assignment-name-your-username
```

### 2.6 Verify the Clone

```bash
# List files in the directory
ls  # Mac/Linux
dir # Windows

# Check git status
git status
```

You should see a message like: `On branch main` or `On branch master`

---

## Step 3: Understanding the Repository Structure

Your assignment repository typically contains:

```
assignment-name-your-username/
├── .github/
│   └── workflows/
│       └── classroom.yml      # CI/CD configuration
├── tests/
│   ├── __init__.py
│   └── test_assignment.py     # Test files
├── src/
│   ├── __init__.py
│   └── your_code.py           # Your assignment code goes here
├── .gitignore                 # Files Git should ignore
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── assignment.md              # Assignment instructions
```

**Important Files:**
- **assignment.md**: Read this first! Contains what you need to do
- **src/**: Write your code here
- **tests/**: Contains tests that verify your code works
- **requirements.txt**: Lists Python packages needed for the assignment

---

## Step 4: Creating a Branch

Branches allow you to work on your assignment without affecting the main code until you're ready.

### 4.1 Why Use Branches?

- Keep your main branch clean and stable
- Work on features independently
- Practice professional development workflows
- Easy to discard changes if needed

### 4.2 Create a New Branch

```bash
# Create and switch to a new branch named "development"
git checkout -b development

# Alternative: use switch (newer Git versions)
git switch -c development
```

### 4.3 Verify Your Branch

```bash
git branch
```

You should see:
```
  main
* development
```

The asterisk (*) shows which branch you're currently on.

### 4.4 Branch Naming Conventions (Optional but Recommended)

For different types of work, you might create:

```bash
git checkout -b feature/add-calculator     # For new features
git checkout -b fix/resolve-bug           # For bug fixes
git checkout -b docs/update-readme        # For documentation
```

---

## Step 5: Working on Your Assignment

### 5.1 Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt

# On some systems, you might need:
pip3 install -r requirements.txt

# Or use a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 5.2 Read the Assignment Instructions

```bash
# Open assignment.md in your text editor or:
cat assignment.md  # Mac/Linux
type assignment.md # Windows
```

### 5.3 Write Your Code

Open the appropriate files in your code editor (usually in the `src/` directory) and complete the assignment tasks.

### 5.4 Save Your Work Frequently

Use `Ctrl+S` (Windows/Linux) or `Cmd+S` (Mac) to save your files as you work.

---

## Step 6: Running Tests Locally

**Always test your code locally before pushing to GitHub!**

### 6.1 Understanding Test Files

Tests verify that your code works correctly. They're located in the `tests/` directory.

### 6.2 Running Tests with pytest

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_assignment.py

# Run tests and show print statements
pytest -s
```

### 6.3 Running Tests with unittest

If your assignment uses unittest instead:

```bash
# Run all tests
python -m unittest discover

# Run a specific test file
python -m unittest tests.test_assignment

# Run with verbose output
python -m unittest discover -v
```

### 6.4 Understanding Test Output

**Passing tests:**
```
tests/test_assignment.py::test_function_one PASSED
tests/test_assignment.py::test_function_two PASSED
========================= 2 passed in 0.05s =========================
```

**Failing tests:**
```
tests/test_assignment.py::test_function_one FAILED
========================= FAILURES =================================
_________________________ test_function_one ________________________
    def test_function_one():
>       assert add(2, 2) == 4
E       assert 5 == 4
```

The `E` line shows what went wrong. Fix your code and run tests again!

### 6.5 Test-Driven Development Workflow

1. Read what the test expects
2. Write code to make the test pass
3. Run the test
4. If it fails, fix your code
5. Repeat until all tests pass

---

## Step 7: Committing Your Changes

Commits are snapshots of your work at specific points in time.

### 7.1 Check What Changed

```bash
git status
```

You'll see files in red (modified but not staged) or green (staged for commit).

### 7.2 Stage Your Changes

```bash
# Stage specific files
git add src/your_code.py

# Stage all changed files
git add .

# Stage multiple specific files
git add src/file1.py src/file2.py
```

### 7.3 Commit Your Changes

```bash
git commit -m "Add function to calculate sum"
```

**Writing Good Commit Messages:**
- ✅ "Add user input validation"
- ✅ "Fix division by zero error"
- ✅ "Implement bubble sort algorithm"
- ❌ "changes"
- ❌ "fixed stuff"
- ❌ "asdf"

### 7.4 Make Multiple Commits

As you work, commit frequently:

```bash
# Work on feature 1
git add src/feature1.py
git commit -m "Implement feature 1"

# Work on feature 2
git add src/feature2.py
git commit -m "Implement feature 2"

# Fix a bug
git add src/feature1.py
git commit -m "Fix edge case in feature 1"
```

### 7.5 View Commit History

```bash
# See all commits
git log

# See compact version
git log --oneline

# See last 5 commits
git log -5
```

---

## Step 8: Pushing Your Code

Pushing uploads your local commits to GitHub.

### 8.1 First Time Push (Set Upstream)

```bash
# Push your branch to GitHub
git push -u origin development
```

The `-u` flag sets up tracking so future pushes are easier.

### 8.2 Subsequent Pushes

```bash
# After the first push, just use:
git push
```

### 8.3 Push Workflow

```bash
# 1. Check status
git status

# 2. Stage changes
git add .

# 3. Commit changes
git commit -m "Descriptive message"

# 4. Push to GitHub
git push
```

### 8.4 Verify Your Push

1. Go to your repository on GitHub
2. Click on your branch name dropdown (top left)
3. Select your branch
4. You should see your latest commits!

---

## Step 9: Understanding CI/CD

CI/CD (Continuous Integration/Continuous Deployment) automatically tests your code when you push to GitHub.

### 9.1 What is CI/CD?

- **Continuous Integration (CI)**: Automatically runs tests when you push code
- **Continuous Deployment (CD)**: Automatically deploys code if tests pass (not used in this course)

### 9.2 How It Works in Your Assignment

1. You push code to GitHub
2. GitHub Actions automatically runs
3. It sets up a fresh environment
4. Installs dependencies from `requirements.txt`
5. Runs all tests
6. Reports success or failure

### 9.3 Viewing CI/CD Results

**Method 1: Actions Tab**
1. Go to your repository on GitHub
2. Click the **"Actions"** tab
3. You'll see a list of all workflow runs
4. Click on a run to see details

**Method 2: Commit Status**
1. Look for a ✅ (passed) or ❌ (failed) next to your commits
2. Click the icon to see details

### 9.4 Understanding Workflow Status

- **🟡 Yellow dot**: Tests are running
- **✅ Green checkmark**: All tests passed!
- **❌ Red X**: Tests failed
- **⚪ Gray circle**: Tests haven't run yet

### 9.5 Reading CI/CD Logs

If tests fail:
1. Click on the failed workflow run
2. Click on the job name (usually "Autograding")
3. Expand the "Run tests" section
4. Read the error messages
5. Fix your code and push again

### 9.6 Common CI/CD Failures

| Issue | Cause | Solution |
|-------|-------|----------|
| Import errors | Missing dependencies | Check `requirements.txt` |
| Test failures | Code doesn't meet requirements | Fix your code logic |
| Syntax errors | Python syntax mistakes | Check your code for typos |
| Timeout | Infinite loops | Review your loops |

---

## Step 10: Creating a Pull Request

Pull Requests (PRs) are proposals to merge your changes into the main branch.

### 10.1 When to Create a Pull Request

- When your code is complete and tested
- All tests pass locally
- CI/CD checks pass on GitHub

### 10.2 Create a Pull Request

**On GitHub:**
1. Go to your repository
2. Click **"Pull requests"** tab
3. Click **"New pull request"**
4. Set base branch to `main`
5. Set compare branch to `development` (or your branch name)
6. Click **"Create pull request"**

### 10.3 Fill in PR Details

**Title**: Brief description of what you did
```
Complete assignment 1: Calculator functions
```

**Description**: Explain your changes
```
## Changes Made
- Implemented addition function
- Implemented subtraction function
- Implemented multiplication function
- Added input validation
- All tests passing

## Testing
- ✅ All unit tests pass
- ✅ CI/CD checks pass
- ✅ Tested with edge cases
```

### 10.4 Request Review (Optional)

If your instructor requires it:
- Click **"Reviewers"** on the right
- Select your instructor or TA

### 10.5 Verify Checks Pass

Wait for the CI/CD checks to complete on your PR:
- ✅ All checks should be green
- If any fail, fix the issues and push again
- The PR will automatically update

---

## Step 11: Merging Your Branch

Once your PR is approved (or if auto-grading passes):

### 11.1 Check PR Status

Make sure:
- [ ] All CI/CD checks pass (green checkmarks)
- [ ] Any required reviews are approved
- [ ] No merge conflicts exist

### 11.2 Merge the Pull Request

1. Scroll to the bottom of your PR
2. Click **"Merge pull request"**
3. Click **"Confirm merge"**
4. Optionally, delete the branch after merging

### 11.3 Update Your Local Repository

```bash
# Switch back to main branch
git checkout main

# Pull the merged changes
git pull origin main
```

### 11.4 Delete Local Branch (Optional)

```bash
# Delete the development branch locally
git branch -d development

# If you need to force delete
git branch -D development
```

---

## Step 12: Submitting Your Assignment

### 12.1 Submission Checklist

Before considering your assignment submitted:

- [ ] All code is written and working
- [ ] All tests pass locally (`pytest` or `python -m unittest`)
- [ ] Code is pushed to GitHub
- [ ] CI/CD checks pass on GitHub (green checkmarks)
- [ ] Pull request is created (if required)
- [ ] Pull request is merged to main (if required)
- [ ] Assignment deadline hasn't passed

### 12.2 Final Verification

1. Go to your repository on GitHub
2. Check the **Actions** tab - all workflows should be green ✅
3. Check the main branch has your latest code
4. Verify the timestamp of your last commit

### 12.3 What Gets Graded?

Typically, instructors will grade:
- **Code on the main branch** at the deadline
- **Test results** from CI/CD
- **Commit history** (shows your work progression)
- **Code quality** and style

### 12.4 Submitting After the Deadline

If you need to submit late:
1. Contact your instructor FIRST
2. Continue working on your branch
3. Push your changes
4. Mention in your submission that it's late

---

## Common Issues and Troubleshooting

### Issue 1: "Permission denied (publickey)"

**Problem**: Can't push to GitHub

**Solution**: Set up SSH keys or use HTTPS
```bash
# Check your remote URL
git remote -v

# If it shows SSH (git@github.com), switch to HTTPS:
git remote set-url origin https://github.com/your-org/your-repo.git
```

### Issue 2: "Your branch is behind 'origin/main'"

**Problem**: Your local code is outdated

**Solution**: Pull the latest changes
```bash
git pull origin main
```

### Issue 3: Merge Conflicts

**Problem**: Git can't automatically merge changes

**Solution**:
```bash
# 1. Pull latest changes
git pull origin main

# 2. Open conflicted files
# Look for conflict markers:
# <<<<<<< HEAD
# Your changes
# =======
# Other changes
# >>>>>>> branch-name

# 3. Edit files to resolve conflicts
# Remove conflict markers
# Keep the code you want

# 4. Stage resolved files
git add .

# 5. Commit the merge
git commit -m "Resolve merge conflict"

# 6. Push
git push
```

### Issue 4: "fatal: not a git repository"

**Problem**: You're not in the repository directory

**Solution**:
```bash
# Navigate to your repository
cd path/to/your/repository

# Verify
git status
```

### Issue 5: Tests Fail on GitHub but Pass Locally

**Possible causes**:
- Different Python versions
- Missing files (check `.gitignore`)
- Different dependencies

**Solution**:
```bash
# Check what files are tracked
git ls-files

# Make sure all necessary files are committed
git status

# Push any missing files
git add missing-file.py
git commit -m "Add missing file"
git push
```

### Issue 6: Can't Find pytest or unittest

**Problem**: Module not found error

**Solution**:
```bash
# Install pytest
pip install pytest

# Or install all requirements
pip install -r requirements.txt

# Make sure you're using the right Python
which python  # Mac/Linux
where python  # Windows
```

### Issue 7: "Updates were rejected"

**Problem**: Remote has changes you don't have locally

**Solution**:
```bash
# Pull with rebase
git pull --rebase origin main

# Or merge
git pull origin main

# Then push
git push
```

---

## Best Practices

### Code Quality

1. **Write clean, readable code**
   - Use meaningful variable names
   - Add comments for complex logic
   - Follow Python style guide (PEP 8)

2. **Test frequently**
   ```bash
   # Run tests after every major change
   pytest -v
   ```

3. **Commit often**
   - Commit after completing each task
   - Write descriptive commit messages

### Git Workflow

1. **Never work directly on main**
   - Always create a branch
   - Keep main branch stable

2. **Pull before you push**
   ```bash
   git pull origin main
   git push
   ```

3. **Review your changes before committing**
   ```bash
   git diff          # See changes
   git status        # Check what's staged
   ```

### Organization

1. **Keep your workspace organized**
   ```
   ~/Documents/
   └── Coursework/
       └── CS101/
           ├── assignment-1/
           ├── assignment-2/
           └── assignment-3/
   ```

2. **Use virtual environments**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Document your code**
   ```python
   def calculate_average(numbers):
       """
       Calculate the average of a list of numbers.
       
       Args:
           numbers (list): List of numeric values
           
       Returns:
           float: The average value
       """
       return sum(numbers) / len(numbers)
   ```

---

## Additional Resources

### Git and GitHub

- [GitHub Docs](https://docs.github.com/en)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Pro Git Book (Free)](https://git-scm.com/book/en/v2)
- [GitHub Learning Lab](https://lab.github.com/)

### Testing

- [pytest Documentation](https://docs.pytest.org/)
- [unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Testing Best Practices](https://realpython.com/pytest-python-testing/)

### Python

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Real Python Tutorials](https://realpython.com/)

### Command Line

- [Command Line Crash Course](https://developer.mozilla.org/en-US/docs/Learn/Tools_and_testing/Understanding_client-side_tools/Command_line)
- [Linux Command Line Basics](https://ubuntu.com/tutorials/command-line-for-beginners)

### Getting Help

1. **Check the error message** - Read it carefully!
2. **Search online** - Google the error message
3. **Ask classmates** - Collaboration is encouraged (for concepts, not copying code)
4. **Office hours** - Visit your instructor or TA
5. **GitHub Discussions** - Use the course discussion board
6. **Stack Overflow** - Search for similar questions

---

## Quick Reference Commands

### Essential Git Commands

```bash
# Check status
git status

# Create and switch to new branch
git checkout -b branch-name

# Stage changes
git add filename.py        # Specific file
git add .                  # All files

# Commit changes
git commit -m "Message"

# Push changes
git push origin branch-name

# Pull changes
git pull origin main

# View commit history
git log --oneline

# Switch branches
git checkout branch-name

# List branches
git branch
```

### Testing Commands

```bash
# pytest
pytest                     # Run all tests
pytest -v                  # Verbose output
pytest tests/test_file.py  # Specific file

# unittest
python -m unittest discover           # Run all tests
python -m unittest tests.test_file    # Specific file
```

### Python Commands

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Freeze dependencies
pip freeze > requirements.txt
```

---

## Assignment Workflow Summary

Here's the complete workflow from start to finish:

```bash
# 1. Accept assignment (on GitHub Classroom)
# 2. Clone repository
git clone https://github.com/your-org/assignment-name-your-username.git
cd assignment-name-your-username

# 3. Create branch
git checkout -b development

# 4. Install dependencies
pip install -r requirements.txt

# 5. Work on assignment
# (Edit files in your code editor)

# 6. Test your code
pytest -v

# 7. Commit changes (repeat as you work)
git add .
git commit -m "Implement feature X"

# 8. Push to GitHub
git push -u origin development

# 9. Create Pull Request (on GitHub)
# 10. Verify CI/CD passes
# 11. Merge Pull Request
# 12. Done! 🎉
```

---

## Final Tips for Success

1. **Start early** - Don't wait until the last minute
2. **Read instructions carefully** - Make sure you understand what's required
3. **Test frequently** - Catch errors early
4. **Commit often** - Small, frequent commits are better than large ones
5. **Ask for help** - When stuck, reach out!
6. **Learn from mistakes** - Review failed tests to understand what went wrong
7. **Stay organized** - Keep your code and files well-structured
8. **Follow conventions** - Use consistent naming and style
9. **Document your work** - Add comments and docstrings
10. **Have fun!** - Learning Git and GitHub is a valuable skill

---

**Good luck with your assignment! 🚀**

If you have any questions, please reach out during office hours or post in the course discussion forum.

---

*Last updated: February 2026*