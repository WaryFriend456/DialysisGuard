"""
Physiological Simulation Engine — Generates realistic dialysis vital trajectories.

Instead of replaying CSV data, this engine generates fresh random but clinically
plausible vital signs for each simulation session. It models the physiological
dynamics of hemodialysis including BP drops from fluid removal, compensatory HR
increases, and potential deterioration events.
"""
import random
import math
import numpy as np
from typing import Dict, List, Optional


class PhysiologicalSimulator:
    """
    Generates realistic hemodialysis vital sign trajectories.
    
    Models:
    - Blood pressure: natural decline from ultrafiltration, with noise and events
    - Heart rate: compensatory increase via baroreceptor reflex
    - Other features: derived from patient baseline data
    
    Risk profiles control how likely deterioration events are.
    """
    
    RISK_PROFILES = {
        "low": {"bp_drop_rate": 0.15, "event_prob": 0.02, "noise_scale": 1.0},
        "moderate": {"bp_drop_rate": 0.30, "event_prob": 0.06, "noise_scale": 1.2},
        "high": {"bp_drop_rate": 0.50, "event_prob": 0.12, "noise_scale": 1.5},
        "critical": {"bp_drop_rate": 0.70, "event_prob": 0.20, "noise_scale": 1.8}
    }
    
    RISK_WEIGHTS = {"low": 0.40, "moderate": 0.35, "high": 0.20, "critical": 0.05}
    
    TIME_STEPS = 30
    TIME_INTERVAL = 8  # minutes between readings
    
    def __init__(self):
        self.rng = random.Random()
    
    def pick_risk_profile(self) -> str:
        """Randomly select a risk profile with weighted probabilities."""
        profiles = list(self.RISK_WEIGHTS.keys())
        weights = list(self.RISK_WEIGHTS.values())
        return self.rng.choices(profiles, weights=weights, k=1)[0]
    
    def generate_session(self, patient_data: dict, risk_profile: str = None) -> List[Dict]:
        """
        Generate a full 30-step vital trajectory for a dialysis session.
        
        Args:
            patient_data: Patient clinical data (from database)
            risk_profile: "low", "moderate", "high", "critical" (or auto-pick)
        
        Returns:
            List of 30 dictionaries, each containing all features for one time step
        """
        if risk_profile is None:
            risk_profile = self.pick_risk_profile()
        
        profile = self.RISK_PROFILES.get(risk_profile, self.RISK_PROFILES["moderate"])
        
        # Extract patient baselines
        pre_bp = self._get_baseline_bp(patient_data)
        pre_hr = self._get_baseline_hr(patient_data)
        weight = patient_data.get("weight", 75.0)
        fluid_rate = patient_data.get("fluid_removal_rate", 350.0)
        
        # Generate BP trajectory
        bp_trajectory = self._generate_bp_trajectory(pre_bp, profile)
        
        # Generate HR trajectory (compensatory to BP)
        hr_trajectory = self._generate_hr_trajectory(pre_hr, bp_trajectory, profile)
        
        # Build full time series
        session_data = []
        for step in range(self.TIME_STEPS):
            time_min = step * self.TIME_INTERVAL
            
            # Compute derived features
            bp = bp_trajectory[step]
            hr = hr_trajectory[step]
            
            bp_change = bp - bp_trajectory[step - 1] if step > 0 else 0.0
            hr_change = hr - hr_trajectory[step - 1] if step > 0 else 0.0
            bp_deviation = bp - bp_trajectory[0]
            hr_deviation = hr - hr_trajectory[0]
            
            # Rolling volatility (std of last 5 readings)
            start = max(0, step - 4)
            bp_window = bp_trajectory[start:step + 1]
            hr_window = hr_trajectory[start:step + 1]
            bp_volatility = float(np.std(bp_window)) if len(bp_window) > 1 else 0.0
            hr_volatility = float(np.std(hr_window)) if len(hr_window) > 1 else 0.0
            
            step_data = {
                # Static features (constant across steps)
                "Age": patient_data.get("age", 55),
                "Gender": patient_data.get("gender", "Male"),
                "Weight": weight,
                "Diabetes": int(patient_data.get("diabetes", False)),
                "Hypertension": int(patient_data.get("hypertension", False)),
                "Kidney Failure Cause": patient_data.get("kidney_failure_cause", "Other"),
                "Creatinine": patient_data.get("creatinine", 5.0),
                "Urea": patient_data.get("urea", 50.0),
                "Potassium": patient_data.get("potassium", 4.5),
                "Hemoglobin": patient_data.get("hemoglobin", 11.0),
                "Hematocrit": patient_data.get("hematocrit", 33.0),
                "Albumin": patient_data.get("albumin", 3.8),
                "Dialysis Duration (hours)": patient_data.get("dialysis_duration", 4.0),
                "Dialysis Frequency (per week)": patient_data.get("dialysis_frequency", 3),
                "Dialysate Composition": patient_data.get("dialysate_composition", "Standard"),
                "Vascular Access Type": patient_data.get("vascular_access_type", "Fistula"),
                "Dialyzer Type": patient_data.get("dialyzer_type", "High-flux"),
                "Urine Output (ml/day)": patient_data.get("urine_output", 500),
                "Dry Weight (kg)": patient_data.get("dry_weight", 70.0),
                "Fluid Removal Rate (ml/hour)": fluid_rate,
                "Disease Severity": patient_data.get("disease_severity", "Moderate"),
                
                # Temporal features
                "Current_BP": round(bp, 1),
                "Current_HR": round(hr, 1),
                "Time_Minutes": time_min,
                
                # Engineered features
                "BP_Change": round(bp_change, 2),
                "HR_Change": round(hr_change, 2),
                "BP_Deviation": round(bp_deviation, 2),
                "HR_Deviation": round(hr_deviation, 2),
                "BP_Volatility": round(bp_volatility, 2),
                "HR_Volatility": round(hr_volatility, 2),
                "Fluid_Rate_Per_Kg": round(fluid_rate / weight, 4),
                "Session_Progress": round(time_min / 232.0, 4),
            }
            
            session_data.append(step_data)
        
        return session_data
    
    def generate_step(self, patient_data: dict, previous_steps: list,
                      risk_profile: str = "moderate") -> Dict:
        """
        Generate a single next time step (for real-time streaming).
        Uses the existing trajectory + generates the next point.
        """
        profile = self.RISK_PROFILES.get(risk_profile, self.RISK_PROFILES["moderate"])
        step = len(previous_steps)
        time_min = step * self.TIME_INTERVAL
        
        if step == 0:
            # First step — use baseline
            bp = self._get_baseline_bp(patient_data)
            hr = self._get_baseline_hr(patient_data)
        else:
            last = previous_steps[-1]
            prev_bp = last["Current_BP"]
            prev_hr = last["Current_HR"]
            
            # BP: gradual decline + noise + possible event
            base_drift = -profile["bp_drop_rate"]
            noise = self.rng.gauss(0, 3.0 * profile["noise_scale"])
            
            # Inject deterioration event?
            event_drop = 0
            if self.rng.random() < profile["event_prob"]:
                event_drop = -self.rng.uniform(8, 20)  # Sudden BP crash
            
            # Late-session steeper decline
            progress = step / self.TIME_STEPS
            late_factor = 1.0 + progress * 0.5
            
            bp = prev_bp + base_drift * late_factor + noise + event_drop
            bp = max(55, min(195, bp))  # Physiological bounds
            
            # HR: compensatory (baroreceptor reflex)
            bp_drop_from_baseline = previous_steps[0]["Current_BP"] - bp
            hr_compensation = bp_drop_from_baseline * 0.3
            hr_noise = self.rng.gauss(0, 2.0 * profile["noise_scale"])
            hr = previous_steps[0]["Current_HR"] + hr_compensation + hr_noise
            hr = max(45, min(145, hr))  # Physiological bounds
        
        weight = patient_data.get("weight", 75.0)
        fluid_rate = patient_data.get("fluid_removal_rate", 350.0)
        
        # Compute derived features
        all_bps = [s["Current_BP"] for s in previous_steps] + [bp]
        all_hrs = [s["Current_HR"] for s in previous_steps] + [hr]
        
        bp_change = bp - all_bps[-2] if len(all_bps) > 1 else 0.0
        hr_change = hr - all_hrs[-2] if len(all_hrs) > 1 else 0.0
        bp_deviation = bp - all_bps[0] if all_bps else 0.0
        hr_deviation = hr - all_hrs[0] if all_hrs else 0.0
        
        start = max(0, len(all_bps) - 5)
        bp_volatility = float(np.std(all_bps[start:])) if len(all_bps[start:]) > 1 else 0.0
        hr_volatility = float(np.std(all_hrs[start:])) if len(all_hrs[start:]) > 1 else 0.0
        
        return {
            "Age": patient_data.get("age", 55),
            "Gender": patient_data.get("gender", "Male"),
            "Weight": weight,
            "Diabetes": int(patient_data.get("diabetes", False)),
            "Hypertension": int(patient_data.get("hypertension", False)),
            "Kidney Failure Cause": patient_data.get("kidney_failure_cause", "Other"),
            "Creatinine": patient_data.get("creatinine", 5.0),
            "Urea": patient_data.get("urea", 50.0),
            "Potassium": patient_data.get("potassium", 4.5),
            "Hemoglobin": patient_data.get("hemoglobin", 11.0),
            "Hematocrit": patient_data.get("hematocrit", 33.0),
            "Albumin": patient_data.get("albumin", 3.8),
            "Dialysis Duration (hours)": patient_data.get("dialysis_duration", 4.0),
            "Dialysis Frequency (per week)": patient_data.get("dialysis_frequency", 3),
            "Dialysate Composition": patient_data.get("dialysate_composition", "Standard"),
            "Vascular Access Type": patient_data.get("vascular_access_type", "Fistula"),
            "Dialyzer Type": patient_data.get("dialyzer_type", "High-flux"),
            "Urine Output (ml/day)": patient_data.get("urine_output", 500),
            "Dry Weight (kg)": patient_data.get("dry_weight", 70.0),
            "Fluid Removal Rate (ml/hour)": fluid_rate,
            "Disease Severity": patient_data.get("disease_severity", "Moderate"),
            "Current_BP": round(bp, 1),
            "Current_HR": round(hr, 1),
            "Time_Minutes": time_min,
            "BP_Change": round(bp_change, 2),
            "HR_Change": round(hr_change, 2),
            "BP_Deviation": round(bp_deviation, 2),
            "HR_Deviation": round(hr_deviation, 2),
            "BP_Volatility": round(bp_volatility, 2),
            "HR_Volatility": round(hr_volatility, 2),
            "Fluid_Rate_Per_Kg": round(fluid_rate / weight, 4),
            "Session_Progress": round(time_min / 232.0, 4),
        }
    
    def _get_baseline_bp(self, patient_data: dict) -> float:
        """Get realistic baseline BP based on patient data."""
        base = 130.0
        if patient_data.get("hypertension"):
            base += self.rng.uniform(10, 25)
        if patient_data.get("age", 55) > 65:
            base += self.rng.uniform(5, 15)
        if patient_data.get("diabetes"):
            base += self.rng.uniform(0, 10)
        return base + self.rng.gauss(0, 8)
    
    def _get_baseline_hr(self, patient_data: dict) -> float:
        """Get realistic baseline HR."""
        base = 75.0
        if patient_data.get("age", 55) > 65:
            base -= self.rng.uniform(0, 5)
        return base + self.rng.gauss(0, 6)
    
    def _generate_bp_trajectory(self, start_bp: float, profile: dict) -> list:
        """Generate full 30-step BP trajectory."""
        trajectory = [start_bp]
        
        for step in range(1, self.TIME_STEPS):
            prev = trajectory[-1]
            progress = step / self.TIME_STEPS
            
            # Natural decline (ultrafiltration effect)
            drift = -profile["bp_drop_rate"] * (1.0 + progress * 0.5)
            
            # Noise
            noise = self.rng.gauss(0, 3.0 * profile["noise_scale"])
            
            # Deterioration event
            event = 0
            if self.rng.random() < profile["event_prob"]:
                event = -self.rng.uniform(8, 20)
            
            bp = prev + drift + noise + event
            bp = max(55, min(195, bp))
            trajectory.append(bp)
        
        return trajectory
    
    def _generate_hr_trajectory(self, start_hr: float, bp_trajectory: list,
                                 profile: dict) -> list:
        """Generate HR trajectory compensating for BP changes."""
        trajectory = [start_hr]
        
        for step in range(1, self.TIME_STEPS):
            bp_drop = bp_trajectory[0] - bp_trajectory[step]
            compensation = bp_drop * 0.3  # Baroreceptor reflex
            noise = self.rng.gauss(0, 2.0 * profile["noise_scale"])
            
            hr = start_hr + compensation + noise
            hr = max(45, min(145, hr))
            trajectory.append(hr)
        
        return trajectory
    
    def detect_anomalies(self, current_step: dict, previous_steps: list) -> list:
        """
        Detect statistical anomalies in vital signs.
        Uses Z-score and rate-of-change detection.
        """
        anomalies = []
        
        if len(previous_steps) < 3:
            return anomalies
        
        # BP anomaly detection
        bps = [s["Current_BP"] for s in previous_steps]
        bp_mean = np.mean(bps)
        bp_std = np.std(bps) if np.std(bps) > 0 else 1.0
        bp_z = abs(current_step["Current_BP"] - bp_mean) / bp_std
        
        if bp_z > 2.0:
            anomalies.append({
                "feature": "Current_BP",
                "type": "statistical_outlier",
                "severity": "warning" if bp_z < 3.0 else "critical",
                "z_score": round(float(bp_z), 2),
                "value": current_step["Current_BP"],
                "mean": round(float(bp_mean), 1)
            })
        
        # Rate of change detection (sudden BP drop)
        if abs(current_step.get("BP_Change", 0)) > 12:
            anomalies.append({
                "feature": "Current_BP",
                "type": "rapid_decline" if current_step["BP_Change"] < 0 else "rapid_increase",
                "severity": "warning",
                "change": current_step["BP_Change"]
            })
        
        # HR anomaly detection
        hrs = [s["Current_HR"] for s in previous_steps]
        hr_mean = np.mean(hrs)
        hr_std = np.std(hrs) if np.std(hrs) > 0 else 1.0
        hr_z = abs(current_step["Current_HR"] - hr_mean) / hr_std
        
        if hr_z > 2.0:
            anomalies.append({
                "feature": "Current_HR",
                "type": "statistical_outlier",
                "severity": "warning" if hr_z < 3.0 else "critical",
                "z_score": round(float(hr_z), 2),
                "value": current_step["Current_HR"],
                "mean": round(float(hr_mean), 1)
            })
        
        # Tachycardia detection
        if current_step["Current_HR"] > 110:
            anomalies.append({
                "feature": "Current_HR",
                "type": "tachycardia",
                "severity": "warning" if current_step["Current_HR"] < 120 else "critical",
                "value": current_step["Current_HR"]
            })
        
        # Hypotension detection
        if current_step["Current_BP"] < 80:
            anomalies.append({
                "feature": "Current_BP",
                "type": "hypotension",
                "severity": "critical",
                "value": current_step["Current_BP"]
            })
        
        return anomalies


# Singleton
simulator = PhysiologicalSimulator()
