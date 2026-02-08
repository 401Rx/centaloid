# CentiloidPlugin for OsiriX MD - Build Instructions

This document explains how to build and install the Centiloid plugin for OsiriX MD.

## Prerequisites

1. **macOS** with Xcode installed
2. **OsiriX MD** installed
3. **Python 3** with centaloid package installed:
   ```bash
   pip3 install /path/to/centaloid
   ```

## Option A: Use the Pre-built Shell Script Wrapper (Recommended)

The simplest approach is to use the included shell script wrapper. This doesn't require compiling the Objective-C plugin.

### Step 1: Install the Python Package

```bash
cd ~/Desktop/centaloid
pip3 install -e .
```

### Step 2: Create an OsiriX Workflow

1. Open OsiriX MD
2. Go to **Plugins** → **Database** → **Create Database Plugin...**
3. Set the plugin name to "Compute Centiloid"
4. For the script path, browse to:
   ```
   ~/Desktop/centaloid/OsiriXPlugin/CentiloidPlugin.osirixplugin/Contents/Resources/run_centaloid.sh
   ```
5. Configure the input/output parameters

Alternatively, you can run Centiloid from Terminal on exported DICOM:

```bash
# Export DICOM from OsiriX to a folder, then:
python3 -m centaloid.main run /path/to/dicom --output /path/to/output --dicom-sc --dicom-sr

# Then import the .dcm files back into OsiriX
```

---

## Option B: Build the Native Objective-C Plugin

For a more integrated experience, you can build the native OsiriX plugin.

### Step 1: Get OsiriX Plugin SDK

Download the OsiriX Plugin SDK from:
https://github.com/pixmeo/osirix

You need the OsiriXAPI.framework headers.

### Step 2: Create Xcode Project

1. Open Xcode
2. Create a new project: **macOS** → **Bundle**
3. Name it `CentiloidPlugin`
4. Set the bundle extension to `osirixplugin`

### Step 3: Configure Project Settings

In the project settings:

1. **Build Settings**:
   - Set "Wrapper Extension" to `osirixplugin`
   - Add OsiriXAPI.framework to "Framework Search Paths"
   - Set "Installation Directory" to `$(HOME)/Library/Application Support/OsiriX/Plugins`

2. **Info.plist**:
   - Copy contents from `CentiloidPlugin.osirixplugin/Contents/Info.plist`

### Step 4: Add Source Files

Add these files to the project:
- `CentiloidPluginFilter.h`
- `CentiloidPluginFilter.m`

### Step 5: Link Frameworks

Link these frameworks:
- Foundation.framework
- Cocoa.framework
- OsiriXAPI.framework

### Step 6: Build and Install

1. Build the project (Cmd+B)
2. The plugin will be installed to:
   ```
   ~/Library/Application Support/OsiriX/Plugins/CentiloidPlugin.osirixplugin
   ```
3. Restart OsiriX MD

---

## Using the Plugin

1. In OsiriX, select a PET series in the database
2. Go to **Plugins** → **Database** → **Compute Centiloid**
3. Wait for the computation to complete
4. The results will appear as new series in the same study:
   - **Centiloid Analysis - [Tracer]**: Secondary Capture image with visual report
   - **Centiloid Analysis SR - [Tracer]**: Structured Report with measurements

---

## Troubleshooting

### "Python not found"
Make sure Python 3 is installed and accessible:
```bash
which python3
python3 --version
```

### "centaloid module not found"
Install the centaloid package:
```bash
cd ~/Desktop/centaloid
pip3 install -e .
```

### "No DICOM files generated"
Check the output directory for error logs. Common issues:
- Invalid or non-PET DICOM data
- Missing dependencies (numpy, scipy, nibabel, pydicom, Pillow)

### Plugin doesn't appear in OsiriX
- Make sure the plugin is in `~/Library/Application Support/OsiriX/Plugins/`
- Restart OsiriX MD
- Check Console.app for error messages

---

## Manual Workflow (Alternative)

If the plugin doesn't work, you can manually run Centiloid:

1. **Export DICOM from OsiriX**:
   - Select series → Right-click → Export to DICOM files

2. **Run Centiloid**:
   ```bash
   python3 -m centaloid.main run /path/to/exported/dicom \
       --output /path/to/results \
       --dicom-sc \
       --dicom-sr
   ```

3. **Import Results**:
   - Drag the generated `.dcm` files into OsiriX
   - Or: File → Import → Select the DICOM files
