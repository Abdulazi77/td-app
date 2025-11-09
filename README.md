# Torque & Drag — Soft-String Model (PE GN 517 Project)

This app computes pickup, slack-off, and rotating torque/drag along a wellbore
using the **Johancsik soft-string** formulation (SPE-11380 PA) and **API RP 7G**
combined tension-torque limits.  
It integrates the open-source [torque_drag](https://github.com/pro-well-plan/torque_drag)
package with a Streamlit front-end.

## Inputs
- Directional survey (MD ft, Inc °, Az °)
- Drillstring OD/ID (in), length (ft)
- Casing OD/ID (in)
- Mud weight (ppg)
- Friction factor μ (cased / open hole)
- WOB (klbf), bit torque (kft-lbf)
- Scenario (PU / SL / ROB)

## Outputs
- Hookload vs depth, Torque vs depth
- 3-D well path
- Per-foot iteration trace and derivation
- Technical report (TXT) and Excel workbook (XLSX)

## Run locally
```bash
pip install -r requirements.txt
streamlit run td_app.py
