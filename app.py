from flask import Flask, render_template, request, redirect, url_for
from utils.regression import reg_predict
from utils.classification import clssi_predict
from utils.nlp import nlp
import pandas as pd
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html'),200

@app.route('/ml')
def ml_options():
    return render_template('ml_options.html'),200

@app.route('/ml/select', methods=['POST'])
def ml_select():
    ml_type = request.form.get('ml_type')
    if ml_type == 'regression':
        return redirect(url_for('regression_form'))
    elif ml_type == 'classification':
        return redirect(url_for('classification_form'))
    else:
        return redirect(url_for('ml_options'))


@app.route('/ml/regression')
def regression_form():
    return render_template('regression_form.html'),200


@app.route('/ml/classification')
def classification_form():
    return render_template('classification_form.html'),200


@app.route('/ml/regression/predict', methods=['POST'])
def regression_predict():
    try:
        ml_algo= request.form['algorithm']
        date = request.form['date']
        lagging_kvarh = float(request.form['Lagging_Current_Reactive.Power_kVarh'])
        leading_kvarh = float(request.form['Leading_Current_Reactive_Power_kVarh'])
        co2 = float(request.form['CO2(tCO2)'])
        lag_pf = float(request.form['Lagging_Current_Power_Factor'])
        lead_pf = float(request.form['Leading_Current_Power_Factor'])
        nsm = float(request.form['NSM'])
        day_of_week = request.form['Day_of_week']
        load_type = request.form['Load_Type']
        if day_of_week== 'monday' or 'tuesday' or 'wednesday' or 'thusday' or 'friday' :
            week_status = 'Weekday'
        elif (day_of_week== 'saturday','sunday'):
            week_status="Weekend"
        else:
            return "write correct day of week"
    
        data = pd.DataFrame([{
        'date': date,
        'Lagging_Current_Reactive.Power_kVarh': lagging_kvarh,
        'Leading_Current_Reactive_Power_kVarh': leading_kvarh,
        'CO2(tCO2)': co2,
        'Lagging_Current_Power_Factor': lag_pf,
        'Leading_Current_Power_Factor': lead_pf,
        'NSM': nsm,
        'WeekStatus': week_status,
        'Day_of_week': day_of_week,
        'Load_Type': load_type }]) 
        result = reg_predict(data,ml_algo)
        return render_template('result.html', result=f"result of {ml_algo} is {result}")
    except Exception as e:
        return f"error occured in whole code {e}"

@app.route('/ml/classification/predict', methods=['POST'])
def classification_predict():
    try:
        ml_algo=request.form['algorithm']
        input_features = [
            int(request.form['battery_power']),
            int(request.form['blue']),
            float(request.form['clock_speed']),
            int(request.form['dual_sim']),
            int(request.form['fc']),
            int(request.form['four_g']),
            int(request.form['int_memory']),
            float(request.form['m_dep']),
            int(request.form['mobile_wt']),
            int(request.form['n_cores']),
            int(request.form['pc']),
            int(request.form['px_height']),
            int(request.form['px_width']),
            int(request.form['ram']),
            int(request.form['sc_h']),
            int(request.form['sc_w']),
            int(request.form['talk_time']),
            int(request.form['three_g']),
            int(request.form['touch_screen']),
            int(request.form['wifi'])
        ]
        required_fields = [
         'battery_power', 'clock_speed', 'dual_sim', 'fc', 'four_g',
        'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
        'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time',
        'touch_screen', 'algorithm'
         ]
        missing=[feild for feild in required_fields if not  input_features]
        
        if missing:
            return f"error missing feild {missing}"
        
        columns = [
        'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
        'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
        'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
        'touch_screen', 'wifi']
        
        df = pd.DataFrame([input_features], columns=columns)
        
        #dummy=[5000, 1, 2.3, 1, 1, 1, 64, 1.0, 130, 8, 123, 134, 123, 64, 123, 123, 233, 1, 0, 1]
    
        result = clssi_predict(ml_algo,df)
        return render_template('result.html', result=f"{ml_algo} classification model {result}")
    except Exception as e:
        return f"error occured in classification_predict {e}"


@app.route('/nlp')
def nlp_options():
    return render_template('nlp_options.html')

@app.route('/nlp/predict', methods=['POST'])
def nlp_predict():
    try:
        nlp_algo=request.form['embedding']
        text=request.form['text_input']
        result = nlp(nlp_algo,text)
        return render_template('result.html', result=result)
    except Exception as e:
        return f"error occured in nlp_predict func {e}"

if __name__ == '__main__':
    app.run(port=1212,debug=True)
