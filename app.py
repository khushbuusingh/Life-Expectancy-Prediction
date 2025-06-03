from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the data and model
df = pd.read_csv('life_prediction_data.csv',encoding='utf-8',sep=',') 
# Load the ML model
with open('model/Mxgbmodel.pkl', 'rb') as file:
    M_model = pickle.load(file)
with open('model/Fxgbmodel.pkl', 'rb') as file:
    F_model = pickle.load(file)
with open('model/Cxgbmodel.pkl', 'rb') as file:
    C_model = pickle.load(file)


#definig features for male, female and children(<15yr)
featuresF = df.drop(columns=['Life Expectancy','Country Name','Year','Urban Population Binned','Male Adult Mortality', 'Students', 'Obesity Children_TOTAL','male_pop'])
featuresM = df.drop(columns=['Life Expectancy','Country Name','Year','Urban Population Binned','Female Adult Mortality', 'Maternal Mortality', 'Students', 'Obesity Children_TOTAL','Female_pop'])
featuresC = df.drop(columns=['Life Expectancy','Country Name','Year','Urban Population Binned','Obesity Adults_TOTAL', 'daily_smoking', 'Alcohol Consumption','HIV Prevalence'])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

@app.route('/getstarted')
def getstarted():
    return render_template('getstarted.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    name = request.form.get('name')
    age = request.form.get('age')
    phone = request.form.get('phone')
    email = request.form.get('email')
    gender = request.form.get('gender')
    smoking = request.form.get('smoking')
    drinking = request.form.get('drinking')
    hiv_status = request.form.get('hiv_status')
    dtp3 = request.form.get('dtp3')
    mcv2 = request.form.get('mcv2')
    urban_rural = request.form.get('urban_rural')
    country = request.form.get('country')
    height = request.form.get('height')
    weight = request.form.get('weight')
    # bmi = request.form.get('bmi')
    electricity = request.form.get('electricity')
    employment_status = request.form.get('employment_status')

    # Convert height and weight to float
    height = int(height)
    weight = int(weight)

    bmi = weight // ((height / 100) ** 2)

    def create_test_case(input_dict):
        # Extract relevant information from input dictionary
        sex=input_dict['sex']
        country_name = input_dict['country']
        smoking = input_dict['smoking']
        alcohol = input_dict['alcohol']
        urban_rural = input_dict['urban_rural']
        dtp3 = input_dict['dtp3']
        mcv2 = input_dict['mcv2']
        electricity = input_dict['electricity']
        bmi = input_dict['bmi']
        hiv = input_dict['hiv']
        age= input_dict['age']
        unemployment=input_dict['unemployment']

        country_data = df[(df['Country Name'] == country_name) & (df['Year'] <= 2020) & (df['Year'] >= 2015)]
        if country_data.empty:
            raise ValueError(f"No data available for {country_name} from 2015 to 2020.")

        country_data = country_data.drop(columns=['Country Name'])

        test_case = country_data.copy()

        if smoking == 'Yes':
            test_case['daily_smoking'] = country_data['daily_smoking'].values[0]
        else:
            test_case['daily_smoking'] = 0

        if alcohol == 'Yes':
            test_case['Alcohol Consumption'] = country_data['Alcohol Consumption'].values[0]
        else:
            test_case['Alcohol Consumption'] = 0

        if urban_rural == 'Urban':
            test_case['Urban Population'] = country_data['Urban Population'].values[0]
        else:
            test_case['Urban Population'] = 100 - country_data['Urban Population'].values[0]

        if dtp3 == 'Yes':
            test_case['DTP3'] = country_data['DTP3'].values[0]
        else:
            test_case['DTP3'] = 0

        if mcv2 == 'Yes':
            test_case['MCV2_immunization'] = country_data['MCV2_immunization'].values[0]
        else:
            test_case['MCV2_immunization'] = 0

        if electricity == 'Yes':
            test_case['Access to Electricity'] = country_data['Access to Electricity'].values[0]
        else:
            test_case['Access to Electricity'] = 0

        if bmi > 25:
            if age > 18:
                test_case['Obesity Adults_TOTAL'] = country_data['Obesity Adults_TOTAL'].values[0]
            else:
                test_case['Obesity Children_TOTAL'] = country_data['Obesity Children_TOTAL'].values[0]
            test_case['Undernourishment'] = 0
        elif bmi < 18:
            test_case['Obesity Adults_TOTAL'] = 0
            test_case['Undernourishment'] = country_data['Undernourishment'].values[0]
        else:
            test_case['Obesity Adults_TOTAL'] = 0
            test_case['Undernourishment'] = 0

        if hiv == 'Yes':
            test_case['HIV Prevalence'] = country_data['HIV Prevalence'].values[0]
        else:
            test_case['HIV Prevalence'] = 0

        if unemployment == 'Yes':
            test_case['Unemployment'] = country_data['Unemployment'].values[0]
        else:
            test_case['Unemployment'] = 0

        features=None
        global featuresM, featuresF, featuresC

        if age>15:
            if sex=='Male':
                features=featuresM
            elif sex=='Female':
                features=featuresF
            else:
                raise ValueError("Invalid gender input: must be 'Male' or 'Female'.")
        else:
            features= featuresC

        
        required_columns = features.columns

        # test_case = test_case[required_columns]
        test_case = test_case[required_columns].apply(pd.to_numeric, errors='coerce')
        return test_case.mean().values.reshape(1, -1)

    # Prepare the data dictionary for the ML model
    input_dict = {
        'sex': gender,
        'country': country,
        'smoking': smoking,
        'alcohol': drinking,
        'urban_rural': urban_rural,
        'dtp3': dtp3,
        'mcv2': mcv2,
        'electricity': electricity,
        # 'electricity': 'Yes',
        'bmi': bmi,
        'hiv': hiv_status,
        'age': int(age),
        'unemployment': employment_status
    }

    test_case = create_test_case(input_dict)
    # print(test_case)
    
    life_expectancy = None  # Default initialization

    # life_expectancy = M_model.predict(test_case)[0]
        
    if input_dict['age']>15:
        if input_dict['sex']=='Male':
            life_expectancy = M_model.predict(test_case)[0]
        else:
            life_expectancy = F_model.predict(test_case)[0]
    else:
        life_expectancy = C_model.predict(test_case)[0]
    
    if life_expectancy is None:
        return "Error: Unable to predict life expectancy. Please check your inputs.", 400

    # Render the result page
    return render_template('result.html', prediction=life_expectancy, smoking=smoking, drinking=drinking, bmi=bmi, dtp3=dtp3, mcv2=mcv2)


if __name__ == '__main__':
    app.run(debug=True)
