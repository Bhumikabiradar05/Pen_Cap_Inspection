import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
from utils.inference import load_model, score_image
from utils.review_queue import init_review_queue, add_to_queue

# Page Configuration
st.set_page_config(
    page_title="Industrial Pen Cap Quality Control - AI Dashboard",
    page_icon="🏭",
    layout="wide"
)

init_review_queue()

# Session State for Shift Metrics
if "shift_stats" not in st.session_state:
    st.session_state.shift_stats = {"total": 0, "pass": 0, "review": 0, "reject": 0}

# Load Model
with st.spinner("Initializing PatchCore inference engine..."):
    model = load_model()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Factory Line Settings")
line_id = st.sidebar.selectbox("Production Line", ["Line-A (High Speed)", "Line-B (Precision)", "Manual Test Bench"])
shift = st.sidebar.selectbox("Active Shift", ["Shift 1 (06:00 - 14:00)", "Shift 2 (14:00 - 22:00)", "Shift 3 (22:00 - 06:00)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Threshold Calibration")
good_thresh = st.sidebar.number_input("GOOD Threshold (≤)", value=0.2375, format="%.4f")
defect_thresh = st.sidebar.number_input("DEFECTIVE Threshold (>)", value=0.3454, format="%.4f")

st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Navigation Menu", ["Live Multi-View Inspection", "Supervisor Review Queue 📋", "Shift Analytics & Export 📊"])

# --- MAIN DASHBOARD ---
if app_mode == "Live Multi-View Inspection":
    st.title("🏭 Automated Multi-Angle Pen Cap Quality Inspection")
    st.markdown(f"**Active Line:** `{line_id}` | **Engine:** PatchCore (Wide-ResNet50) | **Status:** `{'ONLINE ✅' if model else 'OFFLINE ⚠️'}`")

    if model is None:
        st.error("🚨 Critical Error: Model checkpoint `models/patchcore_pen_cap.ckpt` not found. Please place it in the `models/` directory.")
    else:
        # Top Metrics Bar
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Inspected (Shift)", st.session_state.shift_stats["total"])
        col2.metric("Passed (GOOD)", st.session_state.shift_stats["pass"])
        col3.metric("Review Queue", len(st.session_state.review_queue))
        col4.metric("Rejected (DEFECTIVE)", st.session_state.shift_stats["reject"])
        
        st.markdown("---")

        # Multi-View Batch Upload
        st.subheader("📸 Batch Image Capture (Multi-Angle Inspection)")
        st.markdown("Upload up to 6 distinct angle captures for the current pen cap unit (Clip, Top Face, Bottom Rim, Body Front, Body Back, Logo).")
        
        uploaded_files = st.file_uploader(
            "Upload Inspection Angles", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            st.markdown(f"### Evaluating {len(uploaded_files)} Angle(s)...")
            
            grid_cols = st.columns(min(len(uploaded_files), 3))
            batch_scores = []
            worst_status = "GOOD"

            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                score, status = score_image(model, image, good_thresh, defect_thresh)
                batch_scores.append((image, score, status, uploaded_file.name))
                
                if status == "DEFECTIVE":
                    worst_status = "DEFECTIVE"
                elif status == "NEEDS REVIEW" and worst_status != "DEFECTIVE":
                    worst_status = "NEEDS REVIEW"

                col_idx = idx % len(grid_cols)
                with grid_cols[col_idx]:
                    st.image(image, caption=f"Angle {idx+1}: {uploaded_file.name}", use_column_width=True)
                    st.metric(label=f"Anomaly Score", value=f"{score:.4f}")
                    
                    if status == "GOOD":
                        st.success(f"Result: {status}")
                    elif status == "NEEDS REVIEW":
                        st.warning(f"Result: {status}")
                    else:
                        st.error(f"Result: {status}")
                    st.markdown("---")

            # Update Shift Stats
            st.session_state.shift_stats["total"] += 1
            if worst_status == "GOOD":
                st.session_state.shift_stats["pass"] += 1
            elif worst_status == "NEEDS REVIEW":
                st.session_state.shift_stats["review"] += 1
                add_to_queue(batch_scores[0][0], max([s[1] for s in batch_scores]), worst_status)
            else:
                st.session_state.shift_stats["reject"] += 1

            # Final Verdict Banner
            st.markdown("### 🏷️ Final Unit Inspection Verdict")
            if worst_status == "GOOD":
                st.success("✅ **PASS / ACCEPT**: All 6 angles conform to normal manufacturing tolerances. Unit cleared for packaging.")
            elif worst_status == "NEEDS REVIEW":
                st.warning("⚠️ **HOLD / BORDERLINE**: One or more angles triggered ambiguous variance. Automatically routed to Supervisor Review Queue.")
            else:
                st.error("❌ **REJECT / SCRAP**: Structural defect identified on unit. Diverted to reject bin.")

elif app_mode == "Supervisor Review Queue 📋":
    st.title("📋 Asynchronous Human Supervisor Review Queue")
    st.markdown("Review borderline items flagged by the AI model during live inspection.")
    
    queue = st.session_state.review_queue
    if not queue:
        st.info("✨ Queue is empty. No items currently require manual verification.")
    else:
        for i, item in enumerate(queue):
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                st.image(item["image"], width=140, caption=f"Score: {item['score']:.4f}")
            with col2:
                st.write(f"**Timestamp:** {item['timestamp']}")
                st.write(f"**Max Anomaly Score:** `{item['score']:.4f}`")
                st.write(f"**Initial AI Flag:** `{item['status']}`")
            with col3:
                decision = st.selectbox(
                    f"Supervisor Action {i}", 
                    ["Pending Review", "Override & Approve (GOOD)", "Confirm Defect (SCRAP)"], 
                    key=f"sup_dec_{i}"
                )
                item["decision"] = decision
                if decision != "Pending Review":
                    st.success(f"Recorded: {decision}")
            st.markdown("---")

elif app_mode == "Shift Analytics & Export 📊":
    st.title("📊 Shift Performance & Audit Analytics")
    st.markdown("Real-time telemetry and inspection yield summaries for plant management.")
    
    stats = st.session_state.shift_stats
    total = stats["total"] if stats["total"] > 0 else 1
    
    col1, col2, col3 = st.columns(3)
    col1.metric("First-Pass Yield (FPY)", f"{(stats['pass'] / total) * 100:.1f}%")
    col2.metric("Total Scrap Rate", f"{(stats['reject'] / total) * 100:.1f}%")
    col3.metric("Review Queue Rate", f"{(stats['review'] / total) * 100:.1f}%")
    
    st.markdown("---")
    st.subheader("📥 Export Inspection Logs")
    if st.button("Download Shift CSV Report"):
        df = pd.DataFrame([stats])
        st.download_button("Click to Download Report", df.to_csv(index=False), file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")