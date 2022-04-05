from app import app
from flask import render_template, request, jsonify
import pandas as pd
from config import Config
import dill

# @app.before_request
# def before_request():
#    pass

with open(f'{Config.dumps_dir}pipeline.dill', 'rb') as in_strm:
    model = dill.load(in_strm)

X_test = pd.read_csv(f'{Config.train_test_dir}X_test.csv')
y_test = pd.read_csv(f'{Config.train_test_dir}y_test.csv')


@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html', title='Suicide Prevention',  generations=pd.unique(X_test.generation),
                       age_cats=pd.unique(X_test.age), sex=['male', 'female'])


@app.route("/prediction", methods=["GET"])
def predict():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    # description, company_profile, benefits = "", "", ""
    # request_json = request.get_json()
    #
    # if request_json["description"]:
    #     description = request_json['description']
    # if request_json["company_profile"]:
    #     company_profile = request_json['company_profile']
    # if request_json["benefits"]:
    #     benefits = request_json['benefits']
    #
    # print(description)
    year = request.args.get('year', type=int)
    sex = request.args.get('sex', type=str)
    age = request.args.get('age', type=str)
    population = request.args.get('population', type=int)
    gdp_for_year = request.args.get('gdp_for_year', type=int)
    gdp_per_capita = request.args.get('gdp_per_capita', type=int)
    generation = request.args.get('generation', type=str)

    df = pd.DataFrame({"year": [year],
                        "sex": [sex],
                        "age": [age],
                        "population": [population],
                        "gdp_for_year": [gdp_for_year],
                        "gdp_per_capita": [gdp_per_capita],
                        "generation": [generation]
                        })

    df.to_csv(f"{Config.source_dir}test.csv")
    try:
        preds = model.predict(df)
    except Exception as e:
        data["predictions"] = "Error: ".join(e)
    else:
        data["predictions"] = str(preds['suicides_no'].values)
    # indicate that the request was a success
    data["success"] = True
    # return the data dictionary as a JSON response
    return jsonify(data)
