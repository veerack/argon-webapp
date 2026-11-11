# Gemini 4 Studio (Argon & Carbon)

A local web application for running Google DeepMind's **Gemini 4 Argon** (Deep Reasoning & Architecture) and **Gemini 4 Carbon** (Code & Security Auditor) models via Google Cloud Vertex AI.

Includes real-time streaming of thinking tokens, local file editing tools, integrated workspace terminal execution, clipboard screenshot pasting (<kbd>Ctrl</kbd> + <kbd>V</kbd>), and dynamic workspace folder configuration.

---

## Prerequisites

1. **Python 3.10 or higher**: [python.org/downloads](https://www.python.org/downloads/)
2. **Google Cloud Account**:
   - New accounts receive **$300 in free credits** via [Google Cloud Free Tier](https://cloud.google.com/free).
3. **Google Cloud CLI (`gcloud`)**:
   - Install the CLI from [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install).

---

## Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/veerack/argon-webapp.git
cd argon-webapp
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Google Cloud Setup & Authentication

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one) and copy your **Project ID** (e.g., `my-project-123456`).
3. Enable the **Vertex AI API** for your project:
   👉 **[Enable Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com)**
4. In your terminal, authenticate your local machine using Application Default Credentials (ADC):
   ```bash
   gcloud auth application-default login
   ```
   *This opens a browser window to sign in with your Google account. No manual JSON service account keys are required.*

### Step 4: Start the Web App

- **Windows**: Double-click `start.cmd` or run:
  ```cmd
  python server.py
  ```
- **macOS / Linux**: Run:
  ```bash
  python3 server.py
  ```

Open your browser and navigate to:
👉 **`http://localhost:4321`**

---

## Configuring Settings in the UI

When launching for the first time, click the **Settings** button (gear icon) in the bottom-left corner:

1. **AI Provider & Project ID**:
   - Provider: **Vertex AI (Google Cloud)**
   - Project ID: Enter your **Google Cloud Project ID** (from Step 3).
   - Location: `global` (default) or `us-central1`.
2. **Workspace Directory**:
   - Enter the absolute path to your project folder (e.g., `C:\Users\username\my-app` on Windows or `/home/username/my-app` on Linux/macOS).
   - The UI validates the folder in real time to verify the path exists.
   - Argon and Carbon will have direct autonomous access to inspect, read, search, and edit files inside this directory.
3. **Save**:
   - Click **Save Settings**. The AI client will initialize immediately.

---

## Features

- **Gemini 4 Argon**: Deep reasoning and architecture model for complex refactoring, multi-file planning, and logic implementation.
- **Gemini 4 Carbon**: Fast code inspection and security auditing model for vulnerability analysis and bug remediation.
- **Live Thinking Stream**: Inspect reasoning steps and duration in real time before responses are synthesized.
- **Autonomous Workspace Tools**:
  - `read_file` - Inspect any file in your project directory.
  - `write_file` - Create new files.
  - `edit_file` - Targeted search-and-replace edits.
  - `list_directory` - Browse folder trees.
  - `search_code` - Fast regex/text grep across project files.
- **Integrated Terminal Drawer**: Run local commands directly inside your configured workspace directory.
- **Clipboard Image Support**: Paste screenshots or images directly with <kbd>Ctrl</kbd> + <kbd>V</kbd> for multimodal analysis.
- **Cloud Billing Link**: The **Credits** button in the sidebar footer opens your Google Cloud Billing dashboard directly to monitor remaining credits and usage.

---

## Local Storage & Privacy

- All conversation history, settings, and workspace preferences are saved locally in the `data/` folder on your machine.
- Requests route directly from your local machine to Google Cloud Vertex AI via the official Google GenAI SDK.
- The `.gitignore` is pre-configured to keep your `data/` folder, credentials, and conversation logs out of version control.
