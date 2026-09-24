import joblib
import pandas as pd
import gradio as gr

bundle = joblib.load("penguin_model.joblib")
pipe_, cols, classes = bundle["pipeline"], bundle["feature_cols"], bundle["classes"]

def predict(bill_len, bill_depth, flipper, mass, island, sex):
    row = pd.DataFrame([{
        "bill_length_mm": bill_len, "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper, "body_mass_g": mass,
        "island_Dream": int(island == "Dream"),
        "island_Torgersen": int(island == "Torgersen"),
        "sex_MALE": int(sex == "Male")}])[cols]
    probs = pipe_.predict_proba(row)[0]
    return {c: float(p) for c, p in zip(classes, probs)}

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(value=45, label="Bill length (mm)"),
            gr.Number(value=17, label="Bill depth (mm)"),
            gr.Number(value=200, label="Flipper length (mm)"),
            gr.Number(value=4000, label="Body mass (g)"),
            gr.Dropdown(["Biscoe", "Dream", "Torgersen"], value="Biscoe", label="Island"),
            gr.Radio(["Female", "Male"], value="Female", label="Sex")],
    outputs=gr.Label(num_top_classes=3),
    title="Pygoscelis Penguin Species Identifier",
    description="Enter morphometric measurements to predict Adelie, Chinstrap or Gentoo.")

if __name__ == "__main__":
    demo.launch()