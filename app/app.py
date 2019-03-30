from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import model, time
import json

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
	return render_template('generic.html')

@app.route('/apply', methods=['GET', 'POST'])
def apply():
	return render_template('elements.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

	print(request.args)
	# print(request.url)

	if len(request.args) == 0:
		return 'argError:Zero'
	
	try:
		p1 = request.args.get('NAME_CONTRACT_TYPE')
		p2 = request.args.get('CODE_GENDER')
		p3 = request.args.get('FLAG_OWN_CAR')
		p4 = request.args.get('FLAG_OWN_REALTY')
		p5 = request.args.get('CNT_CHILDREN')
		p6 = request.args.get('AMT_INCOME_TOTAL')
		p7 = request.args.get('AMT_CREDIT')
		p8 = request.args.get('AMT_ANNUITY')
		p9 = request.args.get('NAME_TYPE_SUITE')
		p10 = request.args.get('NAME_INCOME_TYPE')
		p11 = request.args.get('NAME_EDUCATION_TYPE')
		p12 = request.args.get('NAME_FAMILY_STATUS')
		p13 = request.args.get('NAME_HOUSING_TYPE')
		p14 = request.args.get('DAYS_BIRTH')
		p15 = request.args.get('DAYS_EMPLOYED')
		p16 = request.args.get('OWN_CAR_AGE')
		p17 = request.args.get('OCCUPATION_TYPE')


		myJson = {
		'NAME_CONTRACT_TYPE': p1,
		'CODE_GENDER': p2,
		'FLAG_OWN_CAR': p3,
		'FLAG_OWN_REALTY': p4,
		'CNT_CHILDREN': p5,
		'AMT_INCOME_TOTAL': int(p6),
		'AMT_CREDIT': int(p7),
		'AMT_ANNUITY': int(p8),
		'NAME_TYPE_SUITE': p9,
		'NAME_INCOME_TYPE': p10,
		'NAME_EDUCATION_TYPE': p11,
		'NAME_FAMILY_STATUS': p12,
		'NAME_HOUSING_TYPE': p13,
		'DAYS_BIRTH': int(p14),
		'DAYS_EMPLOYED': int(p15),
		'OWN_CAR_AGE': int(p16),
		'OCCUPATION_TYPE': p17
		}

		print(myJson)
	
		myJson_modified = json.dumps(myJson)
		prob = model.predict(myJson)
		prob = str(prob)
		time.sleep(20)
		print(prob)
	
		return prob

	except Exception as e:
		print(e)
		print(len((request.args)))
		return 'argError:incorrect'

@app.route('/test', methods=['GET', 'POST'])
def test():
    return 'helloinnerve'


if __name__ == "__main__":
    app.run(debug=True)
