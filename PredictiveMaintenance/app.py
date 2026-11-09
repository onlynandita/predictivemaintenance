from flask import Flask, render_template, request,redirect
import pandas as pd
import joblib
import os
import pickle
import sqlite3
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file

app = Flask(__name__)


model = joblib.load("model/model.pkl")

@app.route('/')
def login_page():
    return render_template("login.html")
@app.route('/home')
def home():
    return render_template("index.html")
@app.route('/login', methods=['GET','POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin123456":

            return redirect("/home")

        else:
            return "Invalid Login"

    return render_template("login.html")
@app.route('/history')
def history():

    conn = sqlite3.connect("predictions.db")

    df = pd.read_sql_query("SELECT * FROM history", conn)

    conn.close()

    return render_template(
        "history.html",
        table=df.to_html(classes="table table-striped table-bordered", index=False)
    )

@app.route('/dashboard')
def dashboard():

    # =====================================================
    # 1. READ PREDICTION HISTORY
    # =====================================================

    conn = sqlite3.connect("predictions.db")

    try:
        df = pd.read_sql_query(
            "SELECT * FROM history",
            conn
        )
    finally:
        conn.close()

    total = len(df)

    if "prediction" in df.columns:
        healthy = int(
            df["prediction"].str.contains(
                "Healthy",
                case=False,
                na=False
            ).sum()
        )

        failure = int(
            df["prediction"].str.contains(
                "Failure",
                case=False,
                na=False
            ).sum()
        )

        failure_data = df[
            df["prediction"].str.contains(
                "Failure",
                case=False,
                na=False
            )
        ]

    else:
        healthy = 0
        failure = 0
        failure_data = df.iloc[0:0]

    # =====================================================
    # 2. FAILURE BY PRODUCT QUALITY TYPE
    # =====================================================

    if (
        "product_quality_type" in failure_data.columns
        and not failure_data.empty
    ):
        machine_failure = (
            failure_data
            .groupby("product_quality_type")
            .size()
        )

        machine_labels = machine_failure.index.tolist()
        machine_values = machine_failure.values.tolist()

    else:
        machine_labels = []
        machine_values = []

    # =====================================================
    # 3. FAILURE CAUSE ANALYSIS
    # =====================================================

    if (
        "cause" in failure_data.columns
        and not failure_data.empty
    ):
        cause_count = failure_data.groupby("cause").size()

        cause_labels = cause_count.index.tolist()
        cause_values = cause_count.values.tolist()

    else:
        cause_labels = []
        cause_values = []

    # =====================================================
    # 4. RECENT PREDICTIONS
    # =====================================================

    recent = df.tail(10).iloc[::-1]

    recent_table = recent.to_html(
        classes="table table-striped table-bordered",
        index=False
    )

    # =====================================================
    # 5. DYNAMIC FEATURE IMPORTANCE
    # =====================================================

    feature_importance = []

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model",
        "model.pkl"
    )

    try:
        print("\n========== FEATURE IMPORTANCE DEBUG ==========")
        print("Model path:", model_path)
        print("Model file exists:", os.path.exists(model_path))

        # Load model using joblib.
        model = joblib.load(model_path)

        if not hasattr(model, "feature_importances_"):
            raise TypeError(
                "The loaded model does not support feature importance."
            )

        feature_names = [
            "Type",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]

        importances = model.feature_importances_

        if len(importances) != len(feature_names):
            raise ValueError(
                "Model feature count does not match feature names."
            )

        feature_importance = [
            {
                "name": name,
                "value": round(float(importance) * 100, 2)
            }
            for name, importance in zip(
                feature_names,
                importances
            )
        ]

        # Sort from highest to lowest importance.
        feature_importance.sort(
            key=lambda item: item["value"],
            reverse=True
        )

        print("Feature importance loaded successfully.")

        for item in feature_importance:
            print(
                f"{item['name']}: {item['value']}%"
            )

        print("===============================================\n")

    except Exception as e:
        print(
            "Feature importance loading error:",
            repr(e)
        )

        feature_importance = []

    # =====================================================
    # 6. RENDER DASHBOARD
    # =====================================================

    return render_template(
        "dashboard.html",
        total=total,
        healthy=healthy,
        failure=failure,
        machine_labels=machine_labels,
        machine_values=machine_values,
        cause_labels=cause_labels,
        cause_values=cause_values,
        recent_table=recent_table,
        feature_importance=feature_importance
    )



@app.route('/model-evaluation')
def model_evaluation():
    return render_template('evaluation.html')

@app.route('/predict', methods=['POST'])
def predict():

    # =====================================================
    # 1. GET INPUT VALUES
    # =====================================================

    type_value = int(request.form['Type'])

    product_quality_type_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    product_quality_type = product_quality_type_map.get(
        type_value, "Unknown"
    )

    process_temp = float(request.form['ProcessTemp'])
    rpm = float(request.form['RPM'])
    torque = float(request.form['Torque'])
    tool_wear = float(request.form['ToolWear'])

    # =====================================================
    # 2. PREPARE MODEL INPUT
    # =====================================================

    data = pd.DataFrame(
        [[type_value, process_temp, rpm, torque, tool_wear]],
        columns=[
            "Type",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    )

    # =====================================================
    # 3. PREDICT MACHINE CONDITION
    # =====================================================

    # =====================================================
    # 3. PREDICT MACHINE CONDITION USING THRESHOLD
    # =====================================================

    # Experimental threshold selected for further validation.
    # Do not assume this is the final production threshold.
    FAILURE_THRESHOLD = 0.30

    # Obtain predicted probabilities
    probabilities = model.predict_proba(data)[0]

    classes = list(model.classes_)

    if 1 not in classes or 0 not in classes:
        return (
            "Error: The trained model must contain both "
            "Healthy class 0 and Failure class 1.",
            500
        )

    failure_index = classes.index(1)
    healthy_index = classes.index(0)

    # Estimated probability of each class
    failure_probability_raw = float(
        probabilities[failure_index]
    )

    healthy_probability_raw = float(
        probabilities[healthy_index]
    )

    failure_probability = round(
        failure_probability_raw * 100,
        2
    )

    healthy_probability = round(
        healthy_probability_raw * 100,
        2
    )

    # Apply the selected classification threshold
    predicted_class = int(
        failure_probability_raw >= FAILURE_THRESHOLD
    )

    # Confidence in the selected class according to its probability
    confidence = round(
        (
            failure_probability_raw
            if predicted_class == 1
            else healthy_probability_raw
        ) * 100,
        2
    )

    
    # =====================================================
    # 4. PREDICTION RESULT
    # =====================================================

    if predicted_class == 1:
        result = "⚠️ Machine Failure Predicted"
    else:
        result = "✅ Machine is Healthy"

    # =====================================================
    # 5. FAILURE RISK LEVEL
    # =====================================================
    # These thresholds are provisional.
    # Validate them using held-out evaluation data.

    if failure_probability >= 70:
        risk = "High Risk"
    elif failure_probability >= 40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    # =====================================================
    # 6. MODEL-DERIVED HEALTH INDICATOR
    # =====================================================
    # This is NOT a validated physical machine-health score.

    health_score = round(
        100 - failure_probability,
        2
    )

    # =====================================================
    # 7. MACHINE CONDITION
    # =====================================================

    if predicted_class == 1:

        if risk == "High Risk":
            health_status = "🔴 High Priority Inspection"
        else:
            health_status = "🟠 Inspection Recommended"

    else:

        if risk == "High Risk":
            health_status = "🔴 High Risk - Urgent Review"
        elif risk == "Medium Risk":
            health_status = "🟡 Needs Attention"
        else:
            health_status = "🟢 Normal Monitoring"

    # =====================================================
    # 8. MAINTENANCE RECOMMENDATION
    # =====================================================

    if predicted_class == 1:

        # The model predicts a possible failure.

        if tool_wear > 200:
            recommendation = (
                "The model predicts a possible machine failure. "
                "Inspect the cutting tool for excessive wear and "
                "assess whether replacement is necessary."
            )

        elif torque > 60:
            recommendation = (
                "The model predicts a possible machine failure. "
                "Review the applied load and inspect the drive "
                "system for abnormal operating conditions."
            )

        elif rpm < 1200:
            recommendation = (
                "The model predicts a possible machine failure. "
                "Investigate the rotational speed and inspect "
                "the motor and related components as appropriate."
            )

        elif process_temp > 315:
            recommendation = (
                "The model predicts a possible machine failure. "
                "Review the process temperature and inspect the "
                "cooling system if operating conditions warrant it."
            )

        else:
            recommendation = (
                "The model predicts a possible machine failure, "
                "but the available inputs do not establish a "
                "specific cause. Review the sensor readings and "
                "arrange an appropriate equipment inspection."
            )

        cause = (
            "Potential contributing condition identified from "
            "input thresholds; not a confirmed failure cause"
        )

        # Label the possible contributing condition.
        if tool_wear > 200:
            cause = "High Tool Wear - Potential Contributing Condition"

        elif torque > 60:
            cause = "High Torque - Potential Contributing Condition"

        elif rpm < 1200:
            cause = (
                "Low Rotational Speed - Potential "
                "Contributing Condition"
            )

        elif process_temp > 315:
            cause = (
                "High Process Temperature - Potential "
                "Contributing Condition"
            )

    else:

        # The model predicts Healthy.
        # Risk still influences the monitoring recommendation.

        if risk == "High Risk":

            recommendation = (
                "The model predicts a healthy machine, but its "
                "estimated failure probability is high. Verify "
                "the sensor readings, review operating conditions, "
                "and arrange an inspection promptly."
            )

            cause = (
                "No confirmed failure cause; elevated model "
                "failure probability requires review"
            )

        elif risk == "Medium Risk":

            recommendation = (
                "The model predicts a healthy machine, but the "
                "estimated failure probability warrants attention. "
                "Continue monitoring the sensor readings and "
                "investigate persistent abnormalities."
            )

            cause = (
                "No failure predicted; increased model uncertainty "
                "or estimated failure probability warrants monitoring"
            )

        else:

            recommendation = (
                "The model predicts a healthy machine with a low "
                "estimated failure probability. Continue routine "
                "monitoring and follow the scheduled maintenance plan."
            )

            cause = "No failure predicted by the model"

    # =====================================================
    # 9. SAVE PREDICTION TO SQLITE
    # =====================================================

    conn = sqlite3.connect("predictions.db")

    try:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO history
            (
                product_quality_type,
                process_temp,
                rpm,
                torque,
                tool_wear,
                cause,
                prediction,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            product_quality_type,
            process_temp,
            rpm,
            torque,
            tool_wear,
            cause,
            result,
            confidence
        ))

        conn.commit()

    finally:
        conn.close()

    # =====================================================
    # 10. DEBUG OUTPUT
    # =====================================================

    print("\n========== PREDICTION RESULTS ==========")
    print("Prediction:", result)
    print("Prediction Confidence:", confidence, "%")
    print("Healthy Probability:", healthy_probability, "%")
    print("Failure Probability:", failure_probability, "%")
    print("Risk Level:", risk)
    print("Model-Derived Health Indicator:", health_score)
    print("Machine Condition:", health_status)
    print("Maintenance Recommendation:", recommendation)
    print("Possible Contributing Condition:", cause)
    print("========================================\n")

    # =====================================================
    # 11. DISPLAY RESULT PAGE
    # =====================================================

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        failure_probability=failure_probability,
        risk=risk,
        recommendation=recommendation,
        health_score=health_score,
        health_status=health_status,
        cause=cause
    )
@app.route('/batch_upload', methods=['POST'])
def batch_upload():

    file = request.files.get('file')

    if not file or file.filename == '':
        return "Please select a CSV file.", 400

    try:
        df = pd.read_csv(file)

        # Remove unwanted CSV index columns
        df = df.loc[
            :,
            ~df.columns.astype(str).str.match(
                r'^Unnamed(?::\s*\d+)?$',
                case=False
            )
        ]

        # Get the exact features used during training
        expected_features = list(model.feature_names_in_)

        # Check required columns
        missing_features = [
            col for col in expected_features
            if col not in df.columns
        ]

        if missing_features:
            return (
                "Missing required columns: "
                + ", ".join(missing_features),
                400
            )

        # Keep original data for displaying results
        display_df = df.copy()

        # Prepare model input
        input_df = df[expected_features].copy()

        # Convert H, L, M into the model's numerical encoding
        if 'Type' in input_df.columns:
            input_df['Type'] = (
                input_df['Type']
                .astype(str)
                .str.strip()
                .str.upper()
            )

            type_mapping = {
                'H': 0,
                'L': 1,
                'M': 2
            }

            input_df['Type'] = input_df['Type'].map(type_mapping)

            if input_df['Type'].isna().any():
                return (
                    "Invalid Type value. Use only H, L, or M.",
                    400
                )

        # Convert model features to numeric values
        for col in expected_features:
            input_df[col] = pd.to_numeric(
                input_df[col],
                errors='coerce'
            )

        if input_df.isna().any().any():
            return (
                "Some required features contain missing or "
                "invalid values. Please check your CSV.",
                400
            )

        # Match the feature order used during training
        input_df = input_df[expected_features]

        # Generate predictions
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        # Display results alongside original H/L/M values
        display_df['Prediction'] = [
            "✅ Machine is Healthy" if p == 0
            else "⚠️ Machine Failure Predicted"
            for p in predictions
        ]

        display_df['Confidence (%)'] = [
            round(max(prob) * 100, 2)
            for prob in probabilities
        ]

        return render_template(
            "batch_result.html",
            table=display_df.to_html(
                classes="table table-striped table-bordered",
                index=False
            )
        )

    except Exception as e:
        app.logger.exception("Batch prediction failed")
        return f"Batch prediction failed: {str(e)}", 400
@app.route('/trend')
def trend():

    conn = sqlite3.connect("predictions.db")

    df = pd.read_sql_query("SELECT * FROM history", conn)

    conn.close()

    trend = df.groupby("prediction").size()

    return render_template(
        "trend.html",
        labels=trend.index.tolist(),
        values=trend.values.tolist()
    )
@app.route('/generate_report')
def generate_report():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT product_quality_type,
           
           process_temp,
           rpm,
           torque,
           tool_wear,
           prediction,
           confidence
    FROM history
    ORDER BY rowid DESC
    LIMIT 1
    """)

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return "No prediction available."

    product_quality_type,  process_temp, rpm, torque, tool_wear, prediction, confidence = row

    health_score = confidence if "Healthy" in prediction else 100 - confidence

    doc = SimpleDocTemplate("Machine_Report.pdf")
    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>AI-Based Predictive Maintenance Report</b>", styles['Title']))
    story.append(Paragraph("<br/>", styles['Normal']))

    story.append(Paragraph(f"<b>Product Quality Type:</b> {product_quality_type}", styles['Normal']))
    
    story.append(Paragraph(f"<b>Process Temperature:</b> {process_temp} K", styles['Normal']))
    story.append(Paragraph(f"<b>Rotational Speed:</b> {rpm} rpm", styles['Normal']))
    story.append(Paragraph(f"<b>Torque:</b> {torque} Nm", styles['Normal']))
    story.append(Paragraph(f"<b>Tool Wear:</b> {tool_wear} min", styles['Normal']))
    story.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence}%", styles['Normal']))
    story.append(Paragraph(f"<b>Health Score:</b> {round(health_score,2)}/100", styles['Normal']))

    doc.build(story)

    return send_file("Machine_Report.pdf", as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True)
