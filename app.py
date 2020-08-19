
from flask import Flask, render_template, request
from quoters import Quote
import pandas as pd
import pickle

app = Flask(
    __name__, 
    static_url_path="",
    static_folder="template/assets",
    template_folder="template"
)
model_path = "model/imbalance-handling.pkl"
model = pickle.load(open(model_path, "rb"))

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", data=Quote.print())
    else:
        dataset = pd.read_csv(
            request.files['dataset'],
            index_col=0
        )
        dataset['prediksi'] = model.predict(dataset)
        return dataset.to_html()
         
if __name__ == "__main__":
    app.run()