# DialysisGuard: AI-Driven Real-Time Monitoring System

## Overview

**DialysisGuard** is an enterprise-grade AI-driven real-time monitoring and adverse event prediction system for hemodialysis. The system uses a GRU (Gated Recurrent Unit) deep learning model trained on temporal physiological data to predict patient instability during dialysis sessions.

The system features:
- A GRU model **trained on `synthetic_hemodialysis_timeseries.csv`** (5000 patients × 30 time steps)
- **Explainable AI (XAI)** — 7 techniques making every prediction transparent and actionable
- A **FastAPI** backend with REST APIs and WebSocket support
- **Enterprise role-based access control** — Super Admin → Organization Admin → Doctors/Nurses
- A MongoDB database for persistent storage
- A **physiologically realistic simulation engine** that generates random but clinically plausible vital trajectories
- A Next.js frontend with dual-theme clinical UI (light/dark), role-based dashboards, real-time monitoring, XAI, and alerts
- **8 innovative features** pushing beyond standard monitoring (predictive forecasting, smart escalation, anomaly detection, etc.)

---

## Role Hierarchy & Access Control

DialysisGuard implements a 4-tier role-based access control system:

### 1. **SUPER_ADMIN** (Platform Level)
- **Permissions**: 
  - Create and manage Organizations
  - Create and manage Organization Admins
  - Suspend/activate organizations
  - Disable/reset passwords for any user
  - View all organizations and their staff/patient counts
  - Access platform administration dashboard
- **Cannot access**: Clinical workspace (patient monitoring, sessions)
- **Database field**: `role = "super_admin"`

### 2. **ORG_ADMIN** (Organization Level)
- **Permissions**:
  - Manage staff (Doctors and Nurses) in their organization
  - Create, update, disable, and activate staff accounts
  - View organization summary and statistics
  - Reset staff passwords
  - Access organization admin dashboard
- **Cannot access**: Clinical workspace unless they are also assigned a clinical role
- **Database field**: `role = "org_admin"`, `org_id = <organization_id>`

### 3. **DOCTOR** (Clinical Level)
- **Permissions**:
  - View all patients in their organization
  - Create and manage patient records
  - Start and monitor dialysis sessions
  - Access full explainable AI features
  - Receive and acknowledge alerts
  - View session reports
- **Cannot access**: User management, organization settings
- **Database field**: `role = "doctor"`, `org_id = <organization_id>`

### 4. **NURSE** (Clinical Level)
- **Permissions**:
  - Similar to Doctor but with organization-scoped access
  - Monitor dialysis sessions (view-only or edit based on settings)
  - Receive alerts
  - Assist with patient monitoring
- **Cannot access**: Advanced analytics, organization administration
- **Database field**: `role = "nurse"`, `org_id = <organization_id>`

---

## Fresh Clone Bootstrap

Use these versions for a reliable first run:

- Python `3.11.x`
- Node.js `LTS` (`18+`, `20+`, or newer LTS)
- MongoDB running locally on `mongodb://localhost:27017`

### Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env to set super admin credentials
python scripts/preflight.py
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## Step-by-Step: Creating Users & Organizations

### Prerequisites

Before you can create any users, you need to **boot the backend and seed the super admin account**.

#### Step 1: Configure Super Admin in .env

Edit `backend/.env` and add:

```bash
DEBUG=true
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=dialysisguard
JWT_SECRET=change-this-before-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
SIMULATION_INTERVAL_MIN_SECONDS=5
SIMULATION_INTERVAL_MAX_SECONDS=7
SIMULATION_TIME_STEPS=30

# Super Admin account (created on first startup)
SUPER_ADMIN_EMAIL=admin@dialysisguard.com
SUPER_ADMIN_PASSWORD=SuperAdminPassword123!
SUPER_ADMIN_NAME=System Administrator
```

#### Step 2: Start the Backend

```bash
cd backend
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uvicorn main:app --reload --port 8000
```

**What happens automatically:**
- The `seed_super_admin()` function runs at startup (triggered in `main.py` lifespan)
- It checks if `SUPER_ADMIN_EMAIL` and `SUPER_ADMIN_PASSWORD` are set in environment variables
- If they are, it creates a new Super Admin user or updates the existing one
- The user is created with `role="super_admin"`, `org_id=null`, `status="active"`, and `must_change_password=False`
- Returns status: `"created"`, `"verified"`, or `"skipped"` (check server logs)

#### Step 3: Login as Super Admin

In the frontend or via API, login with:

**Email**: `admin@dialysisguard.com`  
**Password**: `SuperAdminPassword123!`

**API Call:**
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@dialysisguard.com",
    "password": "SuperAdminPassword123!"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "507f1f77bcf86cd799439011",
    "name": "System Administrator",
    "email": "admin@dialysisguard.com",
    "role": "super_admin",
    "org_id": null,
    "org_name": null,
    "status": "active",
    "must_change_password": false,
    "created_at": "2026-04-29T12:00:00.000000"
  }
}
```

Use the `access_token` in the `Authorization: Bearer <token>` header for all subsequent API calls.

---

### Creating an Organization

**Role Required**: SUPER_ADMIN  
**Endpoint**: `POST /api/admin/organizations`

**Request:**
```bash
curl -X POST http://localhost:8000/api/admin/organizations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <super_admin_token>" \
  -d '{
    "name": "City Medical Center",
    "code": "CMC",
    "address": "123 Healthcare Ave, Medical City, MC 12345",
    "phone": "+1-555-123-4567",
    "email": "admin@citymedicl.com"
  }'
```

**Response:**
```json
{
  "id": "507f1f77bcf86cd799439012",
  "name": "City Medical Center",
  "code": "CMC",
  "status": "active",
  "address": "123 Healthcare Ave, Medical City, MC 12345",
  "phone": "+1-555-123-4567",
  "email": "admin@citymedicl.com",
  "staff_count": 0,
  "patient_count": 0,
  "created_by": "507f1f77bcf86cd799439011",
  "created_at": "2026-04-29T12:00:00.000000",
  "updated_at": "2026-04-29T12:00:00.000000"
}
```

Save the organization `id` — you'll need it to create org admins.

---

### Creating an Organization Admin

**Role Required**: SUPER_ADMIN  
**Endpoint**: `POST /api/admin/organizations/{org_id}/org-admins`

**Request:**
```bash
curl -X POST http://localhost:8000/api/admin/organizations/507f1f77bcf86cd799439012/org-admins \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <super_admin_token>" \
  -d '{
    "name": "Dr. Hospital Manager",
    "email": "hospital.manager@citymedicl.com",
    "role": "org_admin"
  }'
```

**Response:**
```json
{
  "user": {
    "id": "507f1f77bcf86cd799439013",
    "name": "Dr. Hospital Manager",
    "email": "hospital.manager@citymedicl.com",
    "role": "org_admin",
    "org_id": "507f1f77bcf86cd799439012",
    "org_name": "City Medical Center",
    "status": "active",
    "must_change_password": true,
    "created_at": "2026-04-29T12:00:00.000000"
  },
  "temporary_password": "aBc123dEfG45hI"
}
```

**Important**: The `temporary_password` is shown only once. Provide it to the new org admin securely. On first login, they will be forced to change it.

---

### Org Admin: Creating Staff (Doctors/Nurses)

**Role Required**: ORG_ADMIN (for their organization)  
**Endpoint**: `POST /api/org/staff`

The org admin logs in with their email and new password (after changing the temporary one).

**Request:**
```bash
curl -X POST http://localhost:8000/api/org/staff \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <org_admin_token>" \
  -d '{
    "name": "Dr. Sarah Johnson",
    "email": "sarah.johnson@citymedicl.com",
    "role": "doctor"
  }'
```

**Response:**
```json
{
  "user": {
    "id": "507f1f77bcf86cd799439014",
    "name": "Dr. Sarah Johnson",
    "email": "sarah.johnson@citymedicl.com",
    "role": "doctor",
    "org_id": "507f1f77bcf86cd799439012",
    "org_name": "City Medical Center",
    "status": "active",
    "must_change_password": true,
    "created_at": "2026-04-29T12:00:00.000000"
  },
  "temporary_password": "xYz789qWe23rT"
}
```

**To create a Nurse:**
```bash
curl -X POST http://localhost:8000/api/org/staff \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <org_admin_token>" \
  -d '{
    "name": "Nurse John Smith",
    "email": "john.smith@citymedicl.com",
    "role": "nurse"
  }'
```

---

### Login as Different Roles

#### Login as Organization Admin

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "hospital.manager@citymedicl.com",
    "password": "<new_password_after_change>"
  }'
```

**First Time**: User will be forced to change password.
```bash
curl -X POST http://localhost:8000/api/auth/change-password \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <temp_token>" \
  -d '{
    "current_password": "aBc123dEfG45hI",
    "new_password": "MyNewPassword456!"
  }'
```

Then login again with the new password.

#### Login as Doctor

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "sarah.johnson@citymedicl.com",
    "password": "<new_password_after_change>"
  }'
```

#### Login as Nurse

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john.smith@citymedicl.com",
    "password": "<new_password_after_change>"
  }'
```

---

---

## Frontend Routes by Role

After logging in, users are automatically redirected to their role-specific dashboard. Here are all the available routes:

### Login & Authentication Routes

| Route | Purpose | Access |
|---|---|---|
| `http://localhost:3000/login` | **Login page** - Enter email and password | Public |
| `http://localhost:3000/register` | **Register page** | Disabled (admin-only provisioning) |
| `http://localhost:3000/change-password` | **Change password** - Forced on first login | Authenticated users |

### Super Admin Routes

| Route | Purpose | Requires |
|---|---|---|
| `http://localhost:3000/admin/organizations` | **List all organizations** - Create, view, suspend/activate orgs | SUPER_ADMIN |
| `http://localhost:3000/admin/organizations/[id]` | **View organization details** - Edit org info, manage org admins | SUPER_ADMIN |
| `http://localhost:3000/admin/staff` | **Manage all staff across organizations** | SUPER_ADMIN |

### Organization Admin Routes

| Route | Purpose | Requires |
|---|---|---|
| `http://localhost:3000/admin/staff` | **List and manage doctors/nurses** - Create, update, disable/activate staff | ORG_ADMIN |

### Clinical Staff Routes (Doctors & Nurses)

| Route | Purpose | Requires |
|---|---|---|
| `http://localhost:3000/dashboard/doctor` | **Doctor dashboard** - Overview, statistics, quick access | DOCTOR |
| `http://localhost:3000/dashboard/nurse` | **Nurse dashboard** - Simplified view of patients and alerts | NURSE |
| `http://localhost:3000/patients` | **Patient management** - List, create, view, edit patients | DOCTOR or NURSE |
| `http://localhost:3000/patients/[id]` | **Patient details** - Full patient profile and history | DOCTOR or NURSE |
| `http://localhost:3000/patients/[id]/history` | **Patient session history** - Past dialysis sessions | DOCTOR or NURSE |
| `http://localhost:3000/monitor` | **Real-time monitoring** - Active session monitoring with XAI | DOCTOR or NURSE |
| `http://localhost:3000/alerts` | **Alerts dashboard** - View, filter, acknowledge alerts | DOCTOR or NURSE |
| `http://localhost:3000/model-info` | **Model transparency** - Model card, fairness, limitations | DOCTOR or NURSE |

---

## Frontend UI for Admin Roles

Yes! DialysisGuard includes **fully functional web UIs for Super Admin and Organization Admin**. No need to use API calls — everything is available through an intuitive dashboard.

### Super Admin UI

**Access**: `http://localhost:3000/admin/organizations`

**What you can do:**
- View all organizations in the system
- Create new organizations (hospital/clinic)
- Edit organization details (name, code, address, phone, email)
- Suspend or activate organizations
- View organization statistics (staff count, patient count)
- Click organization to view its details
- View all staff across organizations
- Create organization admins
- Disable/activate/reset password for any user in the system

**Key Features:**
- Search organizations by name or code
- Real-time organization list
- Inline organization creation form
- Organization detail page with editing
- Staff management across organizations
- Temporary password generation and display

### Organization Admin UI

**Access**: `http://localhost:3000/admin/staff`

**What you can do:**
- View your organization's summary (staff count, patient count, etc.)
- List all doctors and nurses in your organization
- Create new doctors
- Create new nurses
- Edit staff member details (name, email, role)
- Disable/activate staff members
- Reset staff passwords
- View organization info

**Key Features:**
- Organization stats dashboard
- Search staff by name or email
- Inline staff creation form
- Staff detail editing
- Temporary password generation
- Copy-to-clipboard password functionality
- Real-time staff list updates

---

### Step-by-Step: How to Login and Use the Admin UI

#### Super Admin Login (Simplest Path)

1. **Start the backend** (if not already running):
   ```bash
   cd backend
   .venv\Scripts\activate  # Windows
   # or: source .venv/bin/activate  # Mac/Linux
   uvicorn main:app --reload --port 8000
   ```

2. **Start the frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend runs on `http://localhost:3000`

3. **Open browser**: Go to `http://localhost:3000/login`

4. **Login with Super Admin credentials**:
   - Email: `admin@dialysisguard.com` (or whatever you set in `.env`)
   - Password: `SuperAdminPassword123!` (or whatever you set in `.env`)

5. **Click "Login"**

6. **You're automatically redirected to**: `http://localhost:3000/admin/organizations`

7. **You can now:**
   - See all organizations in a list
   - Click "+ Create Organization" button
   - Fill in: name, code, address, phone, email
   - Click "Create"
   - Organization created! You'll be redirected to its detail page
   - Click "View Staff" to see org admins and staff
   - Click "Create Org Admin" to create a new organization admin (will show temporary password)

---

#### Organization Admin Login (After Super Admin Creates You)

1. **Super Admin created your account** and provided you with:
   - Email: `hospital.manager@citymedicl.com`
   - Temporary Password: (shown once after creation, e.g., `aBc123dEfG45hI`)

2. **Go to**: `http://localhost:3000/login`

3. **Login with temporary password**:
   - Email: `hospital.manager@citymedicl.com`
   - Password: `aBc123dEfG45hI` (the temporary one)

4. **You see "Password Change Required" page** at `/change-password`
   - Enter current password: `aBc123dEfG45hI`
   - Enter new password: `MyNewPassword456!`
   - Enter new password again to confirm
   - Click "Change Password"

5. **Login again** with your new password:
   - Email: `hospital.manager@citymedicl.com`
   - Password: `MyNewPassword456!`

6. **You're automatically redirected to**: `http://localhost:3000/admin/staff`

7. **You can now:**
   - See your organization info at top
   - See staff statistics (doctors, nurses count)
   - See list of all doctors and nurses
   - Click "+ Create Staff" button
   - Fill in: name, email, role (Doctor or Nurse)
   - Click "Create Staff"
   - Staff created! Temporary password shown on screen
   - Provide the temporary password to the new doctor/nurse
   - They login and must change password on first login (same as org admin flow)

---

### Admin UI Screenshots (What You'll See)

#### Organizations List (Super Admin)
```
┌─────────────────────────────────────┐
│ DialysisGuard - Platform Admin      │
├─────────────────────────────────────┤
│                                     │
│ [Search] [+ Create Organization]    │
│                                     │
│ Organizations:                      │
│ ┌─────────────────────────────────┐ │
│ │ City Medical Center              │ │
│ │ Code: CMC                        │ │
│ │ Staff: 5 | Patients: 24          │ │
│ │ Status: Active                   │ │
│ │ [View] [Suspend]                 │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ County Hospital                  │ │
│ │ Code: CH                         │ │
│ │ Staff: 8 | Patients: 42          │ │
│ │ Status: Active                   │ │
│ │ [View] [Suspend]                 │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

#### Staff List (Org Admin)
```
┌──────────────────────────────────────┐
│ DialysisGuard - Organization Admin   │
├──────────────────────────────────────┤
│ Organization: City Medical Center    │
│ Stats: 5 Staff | 24 Patients         │
├──────────────────────────────────────┤
│                                      │
│ [Search] [+ Create Staff]            │
│                                      │
│ Staff List:                          │
│ ┌────────────────────────────────┐   │
│ │ Dr. Sarah Johnson              │   │
│ │ Email: sarah@citymedicl.com    │   │
│ │ Role: Doctor | Status: Active  │   │
│ │ [Edit] [Disable] [Reset Pwd]   │   │
│ └────────────────────────────────┘   │
│                                      │
│ ┌────────────────────────────────┐   │
│ │ Nurse John Smith               │   │
│ │ Email: john@citymedicl.com     │   │
│ │ Role: Nurse | Status: Active   │   │
│ │ [Edit] [Disable] [Reset Pwd]   │   │
│ └────────────────────────────────┘   │
└──────────────────────────────────────┘
```

---

---

### Step-by-Step Frontend Workflows

#### Super Admin Flow

1. Go to `http://localhost:3000/login`
2. Enter Super Admin credentials:
   - Email: `admin@dialysisguard.com`
   - Password: `SuperAdminPassword123!`
3. Click **Login**
4. Auto-redirected to `http://localhost:3000/admin/organizations`
5. From here you can:
   - **Create Organization** - Click "Create Organization" button
   - **View Organization** - Click on any organization in the list
   - **Manage Org Admins** - Click organization → "Staff" tab
   - **Navigate to Staff** - Click "Staff" in sidebar
   - **Manage Users** - List all users, disable/activate, reset passwords

#### Organization Admin Flow

1. Go to `http://localhost:3000/login`
2. Enter Org Admin credentials (provided by Super Admin):
   - Email: `hospital.manager@citymedicl.com`
   - Password: `<temporary_password_from_super_admin>`
3. Click **Login**
4. You'll be redirected to `http://localhost:3000/change-password`
5. **REQUIRED**: Change your password (temporary password enforced)
   - Enter old password (the temporary one)
   - Enter new password
   - Click **Change Password**
6. Login again with new password
7. Auto-redirected to `http://localhost:3000/admin/staff`
8. From here you can:
   - **Create Doctor** - Click "Create Staff" → Select "Doctor" role
   - **Create Nurse** - Click "Create Staff" → Select "Nurse" role
   - **View Staff** - List all doctors and nurses in your organization
   - **Edit Staff** - Click staff member to edit details
   - **Disable Staff** - Click staff member → Disable button
   - **Activate Staff** - Click disabled staff → Activate button
   - **Reset Password** - Click staff member → Reset Password button

#### Doctor Flow

1. Go to `http://localhost:3000/login`
2. Enter Doctor credentials:
   - Email: `sarah.johnson@citymedicl.com`
   - Password: `<new_password_after_change>`
3. Click **Login**
4. Auto-redirected to `http://localhost:3000/dashboard/doctor`
5. Doctor Dashboard shows:
   - **Organization Info** - Your hospital/clinic name
   - **Statistics** - Total patients, active sessions
   - **Recent Alerts** - Latest critical/high alerts
   - **Recent Patients** - Quick access to recently viewed patients
6. From here you can navigate to:
   - **Patients** (`/patients`) - Manage patient registry
   - **Monitoring** (`/monitor`) - View active sessions with real-time XAI
   - **Alerts** (`/alerts`) - View all alerts across your patients
   - **Model Info** (`/model-info`) - Understand the AI model

#### Nurse Flow

1. Go to `http://localhost:3000/login`
2. Enter Nurse credentials:
   - Email: `john.smith@citymedicl.com`
   - Password: `<new_password_after_change>`
3. Click **Login**
4. Auto-redirected to `http://localhost:3000/dashboard/nurse`
5. Nurse Dashboard shows:
   - **Organization Info** - Your hospital/clinic name
   - **Vital Signs** - Large, easy-to-read displays
   - **Alerts** - Prominent alert notifications
   - **Quick Actions** - Fast access to monitoring
6. Can access:
   - **Patients** (`/patients`) - View patients (read-only or limited edit)
   - **Monitoring** (`/monitor`) - View active sessions
   - **Alerts** (`/alerts`) - View alerts relevant to assigned patients

---

### Navigation Sidebar (Role-Aware)

The sidebar automatically shows menu items based on your role:

**Super Admin sees:**
- Organizations
- Staff

**Org Admin sees:**
- Staff
- Organization Summary

**Doctor/Nurse sees:**
- Patients
- Monitoring
- Alerts
- Model Info
- Change Password
- Logout

---

### How Role-Based Routing Works

When you login, the system:

1. Validates email/password
2. Checks user's role from database
3. Normalizes role (e.g., "caregiver" → "nurse")
4. Checks if password change required
5. **Auto-redirects** to appropriate dashboard:
   - `super_admin` → `/admin/organizations`
   - `org_admin` → `/admin/staff`
   - `doctor` → `/dashboard/doctor`
   - `nurse` → `/dashboard/nurse`

If password change is required:
- User is redirected to `/change-password` instead
- Must change password to proceed
- After change, must login again
- Then auto-redirected to role dashboard

---

## User Management API Reference

### Super Admin Endpoints

| Endpoint | Method | Description | Requires |
|---|---|---|---|
| `/api/admin/organizations` | GET | List all organizations | SUPER_ADMIN |
| `/api/admin/organizations` | POST | Create organization | SUPER_ADMIN |
| `/api/admin/organizations/{org_id}` | GET | Get organization details | SUPER_ADMIN |
| `/api/admin/organizations/{org_id}` | PUT | Update organization | SUPER_ADMIN |
| `/api/admin/organizations/{org_id}/suspend` | POST | Suspend organization | SUPER_ADMIN |
| `/api/admin/organizations/{org_id}/activate` | POST | Activate organization | SUPER_ADMIN |
| `/api/admin/organizations/{org_id}/users` | GET | List all users in org | SUPER_ADMIN |
| `/api/admin/organizations/{org_id}/org-admins` | POST | Create org admin | SUPER_ADMIN |
| `/api/admin/users/{user_id}/disable` | POST | Disable any user | SUPER_ADMIN |
| `/api/admin/users/{user_id}/activate` | POST | Activate any user | SUPER_ADMIN |
| `/api/admin/users/{user_id}/reset-password` | POST | Reset user password | SUPER_ADMIN |

### Organization Admin Endpoints

| Endpoint | Method | Description | Requires |
|---|---|---|---|
| `/api/org/summary` | GET | Get org stats | ORG_ADMIN |
| `/api/org/staff` | GET | List doctors/nurses | ORG_ADMIN |
| `/api/org/staff` | POST | Create doctor/nurse | ORG_ADMIN |
| `/api/org/staff/{staff_id}` | PUT | Update staff details | ORG_ADMIN |
| `/api/org/staff/{staff_id}/disable` | POST | Disable staff | ORG_ADMIN |
| `/api/org/staff/{staff_id}/activate` | POST | Activate staff | ORG_ADMIN |

### Auth Endpoints (All Roles)

| Endpoint | Method | Description |
|---|---|---|
| `/api/auth/login` | POST | Login and get JWT token |
| `/api/auth/change-password` | POST | Change your password |
| `/api/auth/me` | GET | Get current user profile |

---

## Preflight Checks

The system includes automatic startup validation:

```bash
python backend/scripts/preflight.py
```

Checks performed:
- ✅ Python version (3.11+)
- ✅ TensorFlow/Keras runtime compatibility
- ✅ ML model files present and valid
- ✅ Scaler and encoder files present
- ✅ MongoDB connectivity
- ✅ Required collections exist

---

## Environment Variables Reference

### Backend Configuration (.env)

```bash
# App Runtime
DEBUG=true                                      # Enable debug logging
MONGODB_URI=mongodb://localhost:27017          # MongoDB connection string
MONGODB_DB=dialysisguard                       # Database name
JWT_SECRET=change-this-before-production       # JWT signing secret (must be 32+ chars in production)
JWT_ALGORITHM=HS256                            # JWT algorithm
JWT_EXPIRATION_HOURS=24                        # Token expiration (hours)

# Super Admin Account (auto-created on startup if set)
SUPER_ADMIN_EMAIL=admin@dialysisguard.com      # Email for root admin
SUPER_ADMIN_PASSWORD=SuperAdminPassword123!    # Password for root admin
SUPER_ADMIN_NAME=System Administrator          # Name for root admin

# Realtime Simulation
SIMULATION_INTERVAL_MIN_SECONDS=5              # Minimum seconds between vital updates
SIMULATION_INTERVAL_MAX_SECONDS=7              # Maximum seconds between vital updates
SIMULATION_TIME_STEPS=30                       # Total steps per session (240 minutes)

# ML Model Paths (auto-configured, no need to change)
MODEL_DIR=backend/ml
MODEL_PATH=backend/ml/gru_model.h5
SCALER_PATH=backend/ml/scaler.pkl
ENCODERS_PATH=backend/ml/label_encoders.pkl

# XAI Configuration
SHAP_BACKGROUND_SAMPLES=100                    # Samples for SHAP background
MC_DROPOUT_PASSES=20                           # Passes for uncertainty (full analysis)
REALTIME_MC_DROPOUT_PASSES=6                   # Passes for uncertainty (realtime)
```

---

## Project Architecture & Components

### Component 1: Project Structure

```text
.
├── final.ipynb                          # Reference notebook
├── Hemodialysis_Data 2.csv              # Original dataset (reference)
├── synthetic_hemodialysis_timeseries.csv # Training dataset
├── Readme.md                            # This file
├── backend/
│   ├── main.py                          # FastAPI app entry point
│   ├── config.py                        # Configuration & settings
│   ├── database.py                      # MongoDB connection
│   ├── requirements.txt                 # Python dependencies
│   ├── .env.example                     # Environment template
│   ├── models/
│   │   ├── schemas.py                   # Pydantic request/response models
│   │   └── __init__.py
│   ├── routes/
│   │   ├── auth.py                      # Authentication (login, register, token)
│   │   ├── admin.py                     # Super admin routes (orgs, org-admins, user management)
│   │   ├── org_admin.py                 # Organization admin routes (staff management)
│   │   ├── patients.py                  # Patient CRUD
│   │   ├── sessions.py                  # Session management
│   │   ├── predictions.py               # Model predictions
│   │   ├── alerts.py                    # Alert endpoints
│   │   ├── explanations.py              # XAI endpoints
│   │   └── __init__.py
│   ├── services/
│   │   ├── ml_service.py                # GRU model inference + MC Dropout
│   │   ├── xai_service.py               # Explainable AI service
│   │   ├── simulation_service.py        # Physiological simulation engine
│   │   └── alert_service.py             # Alert + escalation logic
│   ├── ml/
│   │   ├── train_model.py               # Model training script
│   │   ├── attention_gru.py             # Attention-augmented GRU architecture
│   │   ├── gru_model.h5                 # Trained model (legacy format)
│   │   ├── gru_model.keras              # Trained model (Keras v3 format)
│   │   ├── best_weights.weights.h5      # Training checkpoint weights
│   │   ├── scaler.pkl                   # Fitted scaler
│   │   ├── label_encoders.pkl           # Label encoders
│   │   ├── feature_config.json          # Feature names/ranges
│   │   └── model_card.json              # Model transparency card
│   ├── websocket/
│   │   └── realtime.py                  # WebSocket for real-time streaming
│   └── scripts/
│       └── preflight.py                 # Startup validation
└── frontend/
    ├── package.json
    ├── next.config.mjs
    └── src/
        ├── app/
        │   ├── layout.js / globals.css  # Dark theme
        │   ├── login/ & register/       # Auth pages (role-aware)
        │   ├── dashboard/
        │   │   ├── doctor/page.js       # Doctor dashboard
        │   │   ├── org-admin/page.js    # Org admin dashboard
        │   ├── patients/page.js         # Patient management
        │   ├── monitor/page.js          # Real-time monitoring & XAI
        │   ├── alerts/page.js           # Alerts dashboard
        │   └── model-info/page.js       # Model transparency
        └── components/
            ├── ErrorBoundary.js         # Error handling wrapper
            ├── PageShell.js             # Shared page container
            ├── Sidebar.js               # Navigation menu (role-aware)
            └── ui/
                └── HypnoRing.js         # Custom loading animation
```

---

### Component 2: Database Schema

#### Users Collection

```javascript
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "Dr. Sarah Johnson",
  "email": "sarah@hospital.com",
  "password_hash": "$2b$12$...",                    // bcrypt hash
  "role": "doctor",                                 // super_admin | org_admin | doctor | nurse
  "org_id": "507f1f77bcf86cd799439012",            // null for super_admin
  "status": "active",                               // active | disabled
  "must_change_password": false,                    // Force change on first login
  "created_by": "507f1f77bcf86cd799439011",
  "created_at": "2026-04-29T12:00:00.000000",
  "updated_at": "2026-04-29T12:00:00.000000",
  "last_login_at": "2026-04-29T13:30:00.000000"
}
```

#### Organizations Collection

```javascript
{
  "_id": ObjectId("507f1f77bcf86cd799439012"),
  "name": "City Medical Center",
  "code": "CMC",                                      // Unique, auto-normalized to uppercase
  "status": "active",                                // active | suspended
  "address": "123 Healthcare Ave, Medical City",
  "phone": "+1-555-123-4567",
  "email": "admin@citymedicl.com",
  "created_by": "507f1f77bcf86cd799439011",
  "created_at": "2026-04-29T12:00:00.000000",
  "updated_at": "2026-04-29T12:00:00.000000"
}
```

#### Patients Collection

```javascript
{
  "_id": ObjectId("507f1f77bcf86cd799439100"),
  "org_id": "507f1f77bcf86cd799439012",              // Organization this patient belongs to
  "name": "John Doe",
  "age": 65,
  "gender": "M",
  "weight": 75.5,
  // ... clinical fields
  "created_by": "507f1f77bcf86cd799439014",
  "created_at": "2026-04-29T12:00:00.000000"
}
```

#### Sessions Collection

```javascript
{
  "_id": ObjectId("507f1f77bcf86cd799439200"),
  "patient_id": "507f1f77bcf86cd799439100",
  "org_id": "507f1f77bcf86cd799439012",              // Organization scoping
  "started_by": "507f1f77bcf86cd799439014",
  "start_time": "2026-04-29T14:00:00.000000",
  "end_time": "2026-04-29T14:30:00.000000",
  "status": "completed",                             // active | paused | completed | stopped
  "time_series_data": [ /* vital readings */ ],
  "predictions": [ /* risk predictions */ ],
  "explanations": [ /* XAI data */ ]
}
```

---

### Component 3: ML Model Training

**Training Data**: `synthetic_hemodialysis_timeseries.csv` — 5000 patients, 30 time steps each at 8-min intervals

**Feature Processing**:
- **Static features**: Age, Gender, Weight, Diabetes, Hypertension, Kidney Failure Cause, Creatinine, Urea, Potassium, Hemoglobin, Hematocrit, Albumin, Dialysis Duration, Dialysis Frequency, Dialysate Composition, Vascular Access Type, Dialyzer Type, Urine Output, Dry Weight, Fluid Removal Rate, Disease Severity
- **Temporal features**: Current_BP, Current_HR, Time_Minutes
- **Engineered features**: BP rate-of-change, HR rate-of-change, BP deviation from baseline, cumulative BP volatility
- **Target**: `Is_Unstable` (binary, per time step)

**Model**: Attention-Augmented GRU
```
Input (30 × N_features) →
GRU(128, return_sequences=True) → BatchNorm →
GRU(64, return_sequences=True) → BatchNorm →
★ Bahdanau Attention Layer ★ →
GRU(32) → BatchNorm →
Dense(64, relu) → Dropout(0.3) →
Dense(32, relu) → Dropout(0.2) →
Dense(1, sigmoid)
```

---

### Component 4: Physiological Simulation Engine

**Implementation**: `backend/services/simulation_service.py`

Generates unique, clinically realistic vital sign trajectories:

```python
class PhysiologicalSimulator:
    """Generates realistic dialysis session vital trajectories"""
    
    def generate_session(self, patient_data, risk_profile="moderate"):
        """
        Generates 30 time steps (0-232 min) of vitals:
        
        Blood Pressure Model:
        - Starts at patient's pre-dialysis BP
        - Natural drift downward during dialysis (fluid removal effect)
        - Random noise (±5-10 mmHg per step)
        - If risk_profile="high": steeper drop + potential crash events
        - Bounded by physiological limits (60-200 mmHg)
        
        Heart Rate Model:
        - Starts at patient's baseline HR
        - Compensatory increase as BP drops (baroreceptor reflex)
        - Random noise (±3-5 bpm)
        - Tachycardia events in high-risk profiles
        - Bounded (40-150 bpm)
        """
```

Risk profiles (random selection with weights):
- `low` (40%) — stable vitals
- `moderate` (35%) — minor fluctuations
- `high` (20%) — significant instability
- `critical` (5%) — severe deterioration

---

### Component 5: Explainable AI (XAI) — 7 Pillars

```
Prediction: 78% Risk
     ↓
┌────┬────┬────┬────┬────┬────┐
│ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │
└────┴────┴────┴────┴────┴────┘

1️⃣ SHAP — Feature Attribution
   Feature contributions showing which inputs drove the prediction

2️⃣ Attention Weights — Temporal Focus
   Which time steps mattered most in the GRU sequence

3️⃣ Counterfactual — What-If Analysis
   Minimal changes needed to reach target risk level

4️⃣ MC Dropout — Uncertainty Quantification
   Confidence intervals (mean ± σ) from 20 forward passes

5️⃣ Natural Language — Human-Readable
   Template-based explanations adapting to risk level

6️⃣ Temporal Heatmap — Evolution Over Time
   2D matrix [time × features] showing how contributions evolved

7️⃣ Model Card — Transparency & Trust
   Architecture, performance, fairness, limitations, ethics
```

---

### Component 6: WebSocket Real-Time Streaming

**Endpoint**: `WebSocket /ws/monitor/{session_id}`

Every 5-7 seconds, emits:

```json
{
  "time_minutes": 48,
  "vitals": {"bp": 118.5, "hr": 82.3, "weight": 96.8},
  "prediction": {
    "risk_probability": 0.72,
    "risk_level": "HIGH",
    "confidence": {"mean": 0.72, "lower": 0.64, "upper": 0.80}
  },
  "xai": {
    "top_features": [
      {"name": "BP Drop", "contribution": 0.18, "direction": "risk_increasing"}
    ],
    "attention_weights": [0.02, 0.03, 0.04, 0.05, 0.08, 0.12],
    "nl_explanation": "Risk is HIGH primarily due to...",
    "risk_trend": "increasing",
    "risk_forecast_5step": [0.72, 0.75, 0.78, 0.81, 0.84]
  },
  "anomalies": [
    {"feature": "Current_BP", "type": "rapid_decline", "severity": "warning"}
  ],
  "alert": null
}
```

---

### Component 7: Frontend Dashboard

**Role-Based Navigation:**
- **SUPER_ADMIN**: Admin panel (orgs, org-admins)
- **ORG_ADMIN**: Staff management, organization stats
- **DOCTOR/NURSE**: Patient management, monitoring, alerts

**Key Pages:**
- **Login** — Role-aware authentication
- **Doctor Dashboard** — Active sessions, patient list, alerts
- **Org Admin Dashboard** — Staff stats, organization info
- **Monitoring** — Real-time vitals, XAI explanations, alerts
- **Patients** — Patient registry and management
- **Alerts** — Alert history and escalation tracking
- **Model Info** — Model transparency and fairness metrics

---

## Testing & Validation

### Test the Role Hierarchy

```bash
# 1. Login as Super Admin
curl -X POST http://localhost:8000/api/auth/login \
  -d '{"email":"admin@dialysisguard.com","password":"SuperAdminPassword123!"}'
# ✅ Expect: access_token, role="super_admin"

# 2. Try to access admin endpoints as org_admin
curl -X GET http://localhost:8000/api/admin/organizations \
  -H "Authorization: Bearer <org_admin_token>"
# ❌ Expect: 403 Forbidden (Insufficient permissions)

# 3. Access org endpoints as org_admin
curl -X GET http://localhost:8000/api/org/summary \
  -H "Authorization: Bearer <org_admin_token>"
# ✅ Expect: 200 OK with organization summary

# 4. Create organization (super admin only)
curl -X POST http://localhost:8000/api/admin/organizations \
  -H "Authorization: Bearer <super_admin_token>" \
  -d '{"name":"Test Hospital","code":"TH"}'
# ✅ Expect: 200 OK with organization created

# 5. Create org admin and retrieve temporary password
curl -X POST http://localhost:8000/api/admin/organizations/{org_id}/org-admins \
  -H "Authorization: Bearer <super_admin_token>" \
  -d '{"name":"Manager","email":"mgr@test.com","role":"org_admin"}'
# ✅ Expect: 200 OK with user and temporary_password

# 6. Login with org admin (must change password first)
curl -X POST http://localhost:8000/api/auth/login \
  -d '{"email":"mgr@test.com","password":"temporary_password"}'
# ❌ Expect: 403 Password change required

# 7. Change password
curl -X POST http://localhost:8000/api/auth/change-password \
  -H "Authorization: Bearer <temp_token>" \
  -d '{"current_password":"temporary_password","new_password":"NewPass123"}'
# ✅ Expect: 200 OK

# 8. Login with new password
curl -X POST http://localhost:8000/api/auth/login \
  -d '{"email":"mgr@test.com","password":"NewPass123"}'
# ✅ Expect: access_token, role="org_admin"

# 9. Org admin creates doctor
curl -X POST http://localhost:8000/api/org/staff \
  -H "Authorization: Bearer <org_admin_token>" \
  -d '{"name":"Dr. Smith","email":"smith@test.com","role":"doctor"}'
# ✅ Expect: 200 OK with user and temporary_password

# 10. Org admin tries to create super admin (should fail)
curl -X POST http://localhost:8000/api/org/staff \
  -H "Authorization: Bearer <org_admin_token>" \
  -d '{"name":"Bad Actor","email":"bad@test.com","role":"super_admin"}'
# ❌ Expect: 400 Bad Request (Staff role must be doctor or nurse)
```

### Manual Testing Runbook

1. **Super Admin Flow**: Login → Create org → Create org admin → Verify org admin login
2. **Org Admin Flow**: Login → Create doctors/nurses → View staff list → Disable staff
3. **Doctor Flow**: Login → Access patient list → Start session → Monitor in real-time
4. **Role Isolation**: Try accessing endpoints of other roles → Verify 403 responses
5. **Organization Scoping**: Doctor from Org A cannot see patients from Org B
6. **Password Change**: Verify force change on first login with temporary password

---

## Deployment

### Production Checklist

- [ ] Set `DEBUG=false` in `.env`
- [ ] Generate strong `JWT_SECRET` (min 32 characters)
- [ ] Set unique `SUPER_ADMIN_EMAIL` and `SUPER_ADMIN_PASSWORD`
- [ ] Configure `MONGODB_URI` to point to production database
- [ ] Update `CORS_ORIGINS` to match frontend domain
- [ ] Enable HTTPS on frontend and backend
- [ ] Set up MongoDB backups
- [ ] Monitor logs for errors and unauthorized access attempts
- [ ] Run preflight checks before deployment

### Docker Deployment (Optional)

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/next.config.mjs ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./
EXPOSE 3000
CMD ["npm", "start"]
```

---

## Troubleshooting

### Issue: "Super admin seed skipped"
**Solution**: Make sure `SUPER_ADMIN_EMAIL` and `SUPER_ADMIN_PASSWORD` are set in `.env`. Restart the backend.

### Issue: "Invalid credentials" on super admin login
**Solution**: Double-check the email and password in `.env` match exactly. Check server logs for the seed status.

### Issue: "User is not assigned to an organization"
**Solution**: Non-super-admin users must have an `org_id`. Make sure the user was created with an organization.

### Issue: "Insufficient permissions"
**Solution**: User role doesn't have access to that endpoint. Check role requirements in the API docs.

### Issue: "Organization suspended"
**Solution**: Super admin suspended the organization. Use the activate endpoint to reactivate.

### Issue: "Password change required" error
**Solution**: User was created with a temporary password. They must change it before accessing other endpoints.

---

## API Documentation

Full OpenAPI documentation available at:
```
http://localhost:8000/docs
```

This interactive interface allows you to:
- View all endpoints and their parameters
- Test API calls directly
- See request/response examples
- View authentication requirements

---

## Performance & Scalability

- **MongoDB Indexing**: Ensure indices on `email`, `org_id`, `role` for fast lookups
- **JWT Token Caching**: Frontend caches token in localStorage
- **WebSocket Connection Pooling**: Multiple simultaneous sessions supported
- **Session Timeout**: 24 hours by default (configurable via `JWT_EXPIRATION_HOURS`)

---

## Security Notes

1. **Password Hashing**: bcrypt with auto-salting
2. **JWT Tokens**: HS256 algorithm, expiring after 24 hours
3. **CORS Protection**: Restricted to configured origins
4. **Organization Scoping**: All data queries scoped by `org_id` for multi-tenancy
5. **Role-Based Access Control**: Enforced at endpoint level with `require_roles()`
6. **Super Admin Protection**: Can only be disabled/modified by other super admins
7. **Temporary Passwords**: Shown once, force change on first login
8. **Session Isolation**: Users can only access their organization's data

---

## Contributing & Development

1. Fork/clone the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make changes and test locally
4. Run backend tests: `pytest backend/tests/`
5. Run frontend tests: `npm test`
6. Commit with meaningful messages
7. Push and create a pull request

---

## License

Proprietary - DialysisGuard Healthcare System

---

## Support

For issues, questions, or contributions:
- Check the API docs at `http://localhost:8000/docs`
- Review logs in `backend/` for detailed error messages
- Contact the development team at `support@dialysisguard.com`

---

**Last Updated**: April 29, 2026  
**Version**: 1.0.0 with Enterprise Role-Based Access Control
